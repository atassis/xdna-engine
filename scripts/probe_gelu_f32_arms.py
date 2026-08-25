#!/usr/bin/env python3
"""Does a full-f32 GELU epilogue actually hang on device, and if it runs, what does it buy?

mm_silu_epilogue.cc records that a full-f32 polynomial GELU HANGS ("run_matmul8: kernel run did not
complete") and blames the cycle budget. That is a stated mechanism, not a demonstrated one, and this
tree has two other mechanisms that produce the SAME did-not-complete: a core frame overrunning the
stack reservation, and the mode-4 store path defect. This sweeps the two arms the note names
(GELU_F32_X = f32 cube+tail with the bf16 tanh output, GELU_F32_TANH = f32 everywhere) against the
stack reservation, so hang-vs-complete separates the first two.

Each dispatch runs in a SUBPROCESS under a hard timeout: a hang must be observed, not inherited, and
pyxrt's run.wait() takes no timeout. Timeout => HANG, which is the measurement, not a failure.

Precision, when an arm completes, is scored the way probe_gelu_epilogue.py does it: rtp[0]=0 is a
pure f32 copy, so the identity run yields bit-exactly the accumulator the GELU epilogue reads, and
scoring gelu against a host GELU *of that array* cancels the bfp16 mmul completely.

Run on a freed NPU:
  scripts/npu_lock.sh run -- .venv-iron/bin/python scripts/probe_gelu_f32_arms.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from ml_dtypes import bfloat16

WA = "mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"
PAD_M = int(os.environ.get("PAD_M", 512))
K_REAL, K_AUG, DFF = 768, 800, 3072
TIMEOUT_S = int(os.environ.get("ARM_TIMEOUT", 120))
SEED = 20260825

C0_EXACT = np.float64(0.7978845608028654)

# (label, artifact stem suffix after the mode tag). The baseline is the shipped bf16 body.
ARMS = [
    ("baseline  bf16 x, hw tanh", "modalgelu"),
    ("f32x      f32 x, hw tanh", "modalgelugx"),
    ("f32tanh   f32 x, f32 tanh", "modalgelugh"),
    ("f32x      f32 x, hw tanh   STACK=8192", "modalgelugxs8192"),
    ("f32tanh   f32 x, f32 tanh  STACK=8192", "modalgelughs8192"),
]

f32 = lambda x: np.asarray(x, np.float32)
bf = lambda x: f32(x).astype(bfloat16)


def gelu_ref(x64):
    return 0.5 * x64 * (1.0 + np.tanh(C0_EXACT * (x64 + 0.044715 * x64**3)))


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --- the child: one xclbin, one mode, one dispatch -------------------------------------------

CHILD = r'''
import os, sys, tempfile
import numpy as np
from ml_dtypes import bfloat16
import pyxrt

xclbin, insts, mode, npz, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
z = np.load(npz)
A_bf, B_bf = z["A"].view(bfloat16), z["B"].view(bfloat16)
PAD_M, DFF = int(z["pad_m"]), int(z["dff"])

# rtp_write lowers to one 6-word transaction per core: [.., addr, 0, VALUE, 24, ..].
RTP_IDX = np.arange(8, 8 + 6 * 32, 6)
if mode != 2:
    w = np.fromfile(insts, np.uint32)
    if not (np.all(w[RTP_IDX] == 2) and np.all(w[RTP_IDX + 1] == 24)
            and len(set(w[RTP_IDX - 2].tolist())) == 32):
        sys.exit("rtp mode words not where expected -- refusing to patch blind")
    w[RTP_IDX] = mode
    fd, insts = tempfile.mkstemp(suffix=".txt"); os.close(fd)
    w.tofile(insts)

TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
dev = pyxrt.device(0)
xb = pyxrt.xclbin(xclbin)
dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kk = pyxrt.kernel(ctx, xb.get_kernels()[0].get_name())

instr = np.fromfile(insts, np.uint32)
nout = PAD_M * DFF * 4
bi = pyxrt.bo(dev, instr.nbytes, pyxrt.bo.cacheable, kk.group_id(1))
bufs = []
for i, a in enumerate((A_bf, B_bf)):
    raw = np.ascontiguousarray(a).view(np.uint16)
    b = pyxrt.bo(dev, raw.nbytes, pyxrt.bo.host_only, kk.group_id(3 + i))
    b.write(raw.tobytes(), 0); b.sync(TO)
    bufs.append(b)
bc = pyxrt.bo(dev, nout, pyxrt.bo.host_only, kk.group_id(5))
bi.write(instr.tobytes(), 0); bi.sync(TO)
sys.stderr.write("dispatching\n"); sys.stderr.flush()
kk(3, bi, instr.size, *bufs, bc).wait()
bc.sync(FROM)
np.save(out, np.frombuffer(bc.read(nout, 0), np.float32).reshape(PAD_M, DFF))
sys.stderr.write("completed\n")
'''


def run_arm(xclbin, insts, mode, npz):
    """Returns (array, status). status is 'ok', 'hang', or an error string."""
    fd, out = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    child = os.path.join(tempfile.gettempdir(), "_gelu_arm_child.py")
    with open(child, "w") as fh:
        fh.write(CHILD)
    try:
        p = subprocess.run([sys.executable, child, xclbin, insts, str(mode), npz, out],
                           capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, f"HANG (no completion in {TIMEOUT_S}s)"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()
        return None, "ERROR: " + (tail[-1] if tail else f"rc={p.returncode}")
    return np.load(out), "ok"


def main():
    rng = np.random.default_rng(SEED)
    A = np.zeros((PAD_M, K_AUG), np.float32)
    A[:, :K_REAL] = rng.standard_normal((PAD_M, K_REAL))
    B = np.zeros((K_AUG, DFF), np.float32)
    B[:K_REAL, :] = rng.standard_normal((K_REAL, DFF)) / np.sqrt(K_REAL)
    fd, npz = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    np.savez(npz, A=bf(A).view(np.uint16), B=bf(B).view(np.uint16), pad_m=PAD_M, dff=DFF)

    print(f"PAD_M={PAD_M} K_AUG={K_AUG} DFF={DFF} timeout={TIMEOUT_S}s seed={SEED}\n")
    results = {}
    for label, tag in ARMS:
        stem = f"{PAD_M}x{K_AUG}x{DFF}_64x32x128_8c_{tag}"
        xclbin, insts = f"{WA}/final_{stem}.xclbin", f"{WA}/insts_{stem}.txt"
        if not (os.path.exists(xclbin) and os.path.exists(insts)):
            print(f"{label:44s}  SKIP (not built)")
            continue
        gel, st = run_arm(xclbin, insts, 2, npz)
        if st != "ok":
            print(f"{label:44s}  {st}")
            results[tag] = {"status": st}
            continue
        # identity: the exact f32 accumulator the epilogue reads, so the GEMM error cancels
        ident, st_i = run_arm(xclbin, insts, 0, npz)
        if st_i != "ok":
            print(f"{label:44s}  gelu ok, identity {st_i}")
            results[tag] = {"status": "ok", "identity": st_i}
            continue
        x = f32(ident)
        r = rel_l2(gel, gelu_ref(np.float64(x)))
        finite = np.isfinite(gel).all()
        print(f"{label:44s}  COMPLETES  rel-L2={r:.4e}  finite={finite}")
        results[tag] = {"status": "ok", "rel_l2": r, "finite": bool(finite)}

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
