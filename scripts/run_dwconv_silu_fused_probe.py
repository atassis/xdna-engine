#!/usr/bin/env python3
"""Validate the FUSED dwconv->silu xclbin (final_dwconv_silu_1024x400) standalone on device.

Structural check for roadmap 5-A rung: dwconv (f32 out) -> on-chip f32 fifo -> silu, one
xclbin, two separate cores. The alt-channel per-tile-loop miscompile corrupts EVEN channels
when a HEAVY epilogue rides one loop; here the two ops are separate simple cores, so EVEN and
ODD must BOTH be clean. We compare vs an exact-f32 silu(dwconv(x)) golden (the NPU silu is
bf16-tanh, ~7e-3 rel-L2 floor) and split the per-channel error EVEN vs ODD.

  in bf16 [C=1024,T=400], w bf16 [C,16] (taps[0..8] + BN-bias[9]); out f32 [C,T].
  ABI (dwconv1d.py): kernel(3, instr, n_instr, X[gid3], W[gid4], Y[gid5]).

Usage: .venv-iron/bin/python scratchpad/run_dwconv_silu_fused_probe.py   (NPU must be free)
"""
import os, sys, time
import numpy as np
from ml_dtypes import bfloat16

C, T, KW, K, P = 1024, 400, 16, 9, 4
EX = "mlir-aie/programming_examples/ml/dwconv1d/build"
XCLBIN = f"{EX}/final_dwconv_silu_1024x400.xclbin"
INSTS = f"{EX}/insts_dwconv_silu_1024x400.txt"


def dwconv_ref_fp32(x_bf16, w_bf16):
    """k=9 'same' depthwise conv in fp32 (bf16 values sent), + BN bias @ w[:,9]."""
    x = x_bf16.astype(np.float32)
    w = w_bf16.astype(np.float32)
    out = np.zeros((C, T), dtype=np.float32)
    for t in range(T):
        acc = w[:, K].copy()  # bias
        for i in range(K):
            idx = t - P + i
            if 0 <= idx < T:
                acc = acc + w[:, i] * x[:, idx]
        out[:, t] = acc
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    rng = np.random.RandomState(0)
    x = rng.standard_normal(size=(C, T)).astype(bfloat16)          # x ~ N(0,1)
    taps = rng.uniform(-0.5, 0.5, size=(C, K)).astype(np.float32)
    bias = rng.uniform(-0.2, 0.2, size=(C,)).astype(np.float32)
    w = np.zeros((C, KW), dtype=np.float32)
    w[:, :K] = taps
    w[:, K] = bias
    w = w.astype(bfloat16)

    dw_f = dwconv_ref_fp32(x, w)          # exact f32 dwconv
    ref_f = silu(dw_f)                    # exact f32 silu(dwconv) golden
    print(f"[ref] x{list(x.shape)} w{list(w.shape)} -> silu(dwconv) f32 golden")

    X = np.ascontiguousarray(x).reshape(-1)
    W = np.ascontiguousarray(w).reshape(-1)
    for p in (XCLBIN, INSTS):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build: make -f Makefile.dwsilu NPU2=1 cols=8 build/final_dwconv_silu_1024x400.xclbin")
    instr = np.fromfile(INSTS, dtype=np.uint32)

    import pyxrt
    xclbin = pyxrt.xclbin(XCLBIN)
    kname = xclbin.get_kernels()[0].get_name()
    print(f"[artifacts] kernel='{kname}' instr_words={instr.size}")
    d = pyxrt.device(0)
    d.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(d, xclbin.get_uuid())
    k = pyxrt.kernel(ctx, kname)
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

    ybytes = C * T * 4  # f32 out
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_w = pyxrt.bo(d, W.nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_y = pyxrt.bo(d, ybytes, pyxrt.bo.host_only, k.group_id(5))
    bo_instr.write(instr.tobytes(), 0);           bo_instr.sync(TO)
    bo_x.write(X.view(np.uint16).tobytes(), 0);   bo_x.sync(TO)
    bo_w.write(W.view(np.uint16).tobytes(), 0);   bo_w.sync(TO)

    def once():
        r = k(3, bo_instr, instr.size, bo_x, bo_w, bo_y)
        r.wait()
    once()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        once()
    dt = (time.perf_counter() - t0) / iters

    bo_y.sync(FROM)
    Y = np.frombuffer(bo_y.read(ybytes, 0), dtype=np.float32).reshape(C, T)

    adiff = np.abs(Y - ref_f)
    denom = np.maximum(np.abs(ref_f), 1e-3)
    rel = adiff / denom
    per_ch = adiff.mean(axis=1)            # mean abs err per channel
    relL2 = np.linalg.norm(Y - ref_f) / np.linalg.norm(ref_f)
    thr = 0.05                             # >> bf16-tanh ~7e-3 noise, << O(1) miscompile
    even_bad = int((per_ch[0::2] > thr).sum())
    odd_bad = int((per_ch[1::2] > thr).sum())
    print(f"[run] device time/iter: {dt*1e3:.3f} ms   ({C}ch x {T}, k=9, fused dwconv+silu)")
    print(f"[run] rel-L2 vs exact f32 silu(dwconv): {relL2:.4e}   max|Δ|={adiff.max():.4f}")
    print(f"[run] per-channel mean|Δ|>{thr}:  EVEN {even_bad}/{C//2}   ODD {odd_bad}/{C//2}  (0/0 => no alt-channel miscompile)")
    print(f"[run] Y[0,:4]={Y[0,:4]}  ref={ref_f[0,:4]}")
    ok = (even_bad == 0 and odd_bad == 0 and relL2 < 0.03)
    print(f"[run] FUSED dwconv+silu on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
