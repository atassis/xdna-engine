#!/usr/bin/env python3
"""On-device validation of the A3 fused SiLU epilogue (conformer_epilogues.cc).

Runs conformer_silu_epilogue_f32_bf16 (EPI_M=1, EPI_N=1024) over T=64 rows on the
XDNA2 NPU via pyxrt and compares the bf16 readback against the f64 host SiLU
reference (out = x * sigmoid(x)), matching scripts/parakeet_ref_encoder.py / the
A3 numpy golden.

Gate: rel-L2 <= 0.08 AND corr >= 0.99 vs the f32 golden.

IRON host ABI: opcode=3; kernel(opcode, instr[gid1,cacheable], n_instr, X[gid3], Y[gid4]).
X = [T*N] float32, Y = [T*N] bf16.
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

N, T = 1024, 64
BUILD = "/tmp/scratch"


def silu_ref(x):
    x = x.astype(np.float64)
    return x / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{BUILD}/final.xclbin")
    ap.add_argument("--insts", default=f"{BUILD}/insts.bin")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((T, N)).astype(np.float32) * 3.0)
    ref = silu_ref(x)                                  # f64 golden

    X = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)

    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    instr = np.fromfile(a.insts, dtype=np.uint32)

    import pyxrt
    xclbin = pyxrt.xclbin(a.xclbin)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}")

    d = pyxrt.device(0)
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    Ybytes = T * N * 2
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_y = pyxrt.bo(d, Ybytes, pyxrt.bo.host_only, k.group_id(4))

    bo_instr.write(instr.tobytes(), 0); bo_instr.sync(TO)
    bo_x.write(X.tobytes(), 0); bo_x.sync(TO)

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
    Y = np.frombuffer(bo_y.read(Ybytes, 0), dtype=np.uint16).view(bfloat16).reshape(T, N)
    yf = Y.astype(np.float32)

    a_flat, r_flat = yf.ravel(), ref.ravel().astype(np.float32)
    rel_l2 = float(np.linalg.norm(a_flat - r_flat) / (np.linalg.norm(r_flat) + 1e-12))
    corr = float(np.corrcoef(a_flat, r_flat)[0, 1])
    adiff = np.abs(yf - ref.astype(np.float32))
    print(f"[run] device time/iter: {dt*1e3:.3f} ms  (T={T} x N={N} SiLU)")
    print(f"[run] rel-L2={rel_l2:.5e}  corr={corr:.6f}  max|d|={adiff.max():.4f}  mean|d|={adiff.mean():.6f}")
    print(f"[run] Y[0,:5]={yf[0,:5]}  ref={ref[0,:5]}")
    ok = (rel_l2 <= 0.08) and (corr >= 0.99)
    print(f"[run] SiLU epilogue on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
