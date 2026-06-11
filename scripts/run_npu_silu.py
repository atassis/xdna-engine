#!/usr/bin/env python3
"""Run + validate SiLU/Swish (x*sigmoid(x)) on the XDNA2 NPU via pyxrt.

The kernel (mlir-aie/aie_kernels/aie2p/silu.cc, silu_bf16) is a pure elementwise
bf16 op: out[i] = silu(in[i]). It does NOT compute sigmoid exactly — it uses a
tanh identity, sigmoid(x) = 0.5*(1 + tanh(x/2)), with the AIE bf16 tanh
approximation. So the device result differs from a clean numpy reference by both
the tanh approximation error and bf16 rounding. We compare with a relative
tolerance (~1-2%) rather than a few-ulp bound (cf. dwconv1d which is exact math).

GigaAM Conformer needs two sizes per block:
  400*768  = 307200  (after the first pointwise)
  400*3072 = 1228800 (FFN hidden / GLU expansion)
Each has its own xclbin+insts (the example writes build/final.xclbin, so we
copied per-length artifacts to final_<length>.xclbin / insts_<length>.bin).

IRON host ABI (from silu/test.cpp): opcode=3, single input -> single output:
  kernel(opcode, instr[gid1,cacheable], n_instr, X[gid3], Y[gid4])

Usage:
  .venv-iron/bin/python scripts/run_npu_silu.py --length 307200 --dry   # no NPU
  .venv-iron/bin/python scripts/run_npu_silu.py --length 307200         # REAL run
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

EX = "mlir-aie/programming_examples/ml/silu/build"
SIZES = (307200, 1228800)  # 400*768, 400*3072


def silu_ref_fp32(x_bf16):
    """Clean SiLU in fp32 (input is the bf16 values actually sent).
    x_bf16: [N] bf16. returns [N] fp32 = x*sigmoid(x)."""
    x = x_bf16.astype(np.float32)
    return x / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=307200)
    ap.add_argument("--xclbin", default=None)
    ap.add_argument("--insts", default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    if a.length % 1024 != 0:
        sys.exit(f"length {a.length} must be a multiple of 1024")
    xclbin = a.xclbin or f"{EX}/final_{a.length}.xclbin"
    insts = a.insts or f"{EX}/insts_{a.length}.bin"

    rng = np.random.RandomState(0)
    x = rng.uniform(-6.0, 6.0, size=(a.length,)).astype(bfloat16)

    ref_f = silu_ref_fp32(x)                 # clean silu in fp32
    ref_b = ref_f.astype(bfloat16)           # bf16-rounded clean truth
    print(f"[ref] silu x[{a.length}] bf16 -> out[{a.length}] (fp32 truth, x in [-6,6])")

    X = np.ascontiguousarray(x).reshape(-1)
    if a.dry:
        print(f"[dry] xclbin={xclbin} insts={insts}")
        for p in (xclbin, insts):
            print(f"[dry]   {'OK ' if os.path.exists(p) else 'MISSING'} {p}")
        print(f"[dry] X={X.nbytes}B Y={X.nbytes}B; ref_fp32[:5]={ref_f[:5]}")
        print("[dry] not touching the NPU.")
        return 0

    for p in (xclbin, insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build: source scripts/iron_env.sh && "
                     f"make -C mlir-aie/programming_examples/ml/silu NPU2=1 length={a.length} "
                     f"&& cp {EX}/final.xclbin {xclbin} && cp {EX}/insts.bin {insts}")
    instr = np.fromfile(insts, dtype=np.uint32)

    import pyxrt
    xb = pyxrt.xclbin(xclbin)
    kname = xb.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}")

    d = pyxrt.device(0)
    d.register_xclbin(xb)
    ctx = pyxrt.hw_context(d, xb.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_y = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(4))

    bo_instr.write(instr.tobytes(), 0);              bo_instr.sync(TO)
    bo_x.write(X.view(np.uint16).tobytes(), 0);      bo_x.sync(TO)

    def once():
        r = k(3, bo_instr, instr.size, bo_x, bo_y)
        r.wait()
    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters

    bo_y.sync(FROM)
    Y = np.frombuffer(bo_y.read(X.nbytes, 0), dtype=np.uint16).view(bfloat16).reshape(a.length)

    yf = Y.astype(np.float32)
    adiff = np.abs(yf - ref_f)                       # NPU vs clean fp32 silu
    denom = np.maximum(np.abs(ref_f), 1e-2)
    rel = adiff / denom
    N = a.length

    print(f"[run] device time/iter: {dt*1e3:.3f} ms  ({N} elems)")
    print(f"[run] vs fp32 truth:  max|Δ|={adiff.max():.5f}  mean|Δ|={adiff.mean():.6f}  "
          f"max_rel={rel.max():.4f}  mean_rel={rel.mean():.5f}")
    print(f"[run] Y[:5]={yf[:5]}  ref={ref_f[:5]}")
    # PASS: the kernel uses a tanh approximation + bf16 rounding, so we allow ~2%
    # relative error (matching the example's 0.04 abs tol spirit). A gross
    # data-movement/dtype bug blows way past this.
    ok = (rel.mean() < 0.02) and (np.percentile(rel, 99) < 0.05) and (adiff.max() < 0.2)
    print(f"[run] SiLU on NPU (tanh approx): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
