#!/usr/bin/env python3
"""Validate the TIME-MAJOR fused dwconv->silu xclbin (final_dwconv_silu_t_1024x400) standalone.

Conv step 3b (transpose-dissolving): [T,D] dwconv (dwconv1d_k9_tmajor, f32 out) -> on-chip f32 fifo ->
silu core, ONE xclbin, two separate cores. Mirrors run_dwconv_silu_fused_probe.py but in [T,D] layout
(the channel-major probe is [C,T]). The alt-channel per-tile-loop miscompile corrupts EVEN iterations
when a HEAVY epilogue rides one loop; here the two ops are separate simple cores, so BOTH parities --
per TIME-ROW and per CHANNEL -- must be clean. Golden = exact-f32 silu(dwconv_k9_same_timeaxis(x));
the NPU silu is bf16-tanh (~1.2e-2 rel-L2 floor, matching the channel-major brick).

  Host tensors sent to the xclbin (ABI: kernel(3, instr, n_instr, X[g3], W[g4], Y[g5])):
    X = padded input [T+2P, D]=[408,1024] bf16 (P=4 zero rows top+bottom; overlapping halo tiles).
    W = TAP-MAJOR weights [K+1, D]=[10,1024] bf16 (row p = tap p per-channel; row K = BN bias).
    Y = output [T, D]=[400,1024] f32.

Usage: .venv-iron/bin/python scripts/run_dwconv_silu_tmajor_probe.py   (NPU must be free)
"""
import os, sys, time
import numpy as np
from ml_dtypes import bfloat16

D, T, K, P = 1024, 400, 9, 4
Kp1 = K + 1
EX = "mlir-aie/programming_examples/ml/dwconv1d/build"
XCLBIN = f"{EX}/final_dwconv_silu_t_1024x400.xclbin"
INSTS = f"{EX}/insts_dwconv_silu_t_1024x400.txt"


def dwconv_ref_fp32_td(x_bf16, taps_bf16, bias_bf16):
    """k=9 'same' depthwise conv along TIME in [T,D], fp32 (bf16 values), + BN bias per-channel."""
    x = x_bf16.astype(np.float32)            # [T, D]
    taps = taps_bf16.astype(np.float32)      # [D, K]
    bias = bias_bf16.astype(np.float32)      # [D]
    out = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        acc = bias.copy()                    # [D]
        for p in range(K):
            idx = t - P + p
            if 0 <= idx < T:
                acc = acc + taps[:, p] * x[idx, :]
        out[t, :] = acc
    return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    rng = np.random.RandomState(0)
    x = rng.standard_normal(size=(T, D)).astype(bfloat16)             # x ~ N(0,1), [T,D]
    taps = rng.uniform(-0.5, 0.5, size=(D, K)).astype(bfloat16)       # [D, K] per-channel taps
    bias = rng.uniform(-0.2, 0.2, size=(D,)).astype(bfloat16)         # [D] BN-folded bias

    dw_f = dwconv_ref_fp32_td(x, taps, bias)     # exact f32 dwconv [T,D]
    ref_f = silu(dw_f)                           # exact f32 silu(dwconv) golden
    print(f"[ref] x[{T},{D}] taps[{D},{K}] -> silu(dwconv) f32 golden [T,D]")

    # --- host packing: padded input [T+2P, D] bf16 + tap-major weights [K+1, D] bf16 ---
    xpad = np.zeros((T + 2 * P, D), dtype=bfloat16)
    xpad[P:P + T, :] = x                          # 4 zero rows top + bottom (== 'same' end pad)
    w_tm = np.zeros((Kp1, D), dtype=np.float32)
    w_tm[:K, :] = taps.astype(np.float32).T       # row p = tap p across all D
    w_tm[K, :] = bias.astype(np.float32)          # row K = per-channel bias
    w_tm = w_tm.astype(bfloat16)

    X = np.ascontiguousarray(xpad).reshape(-1)
    W = np.ascontiguousarray(w_tm).reshape(-1)
    for p in (XCLBIN, INSTS):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- build: make -f Makefile.dwsilu_t NPU2=1 cols=8 build/final_dwconv_silu_t_1024x400.xclbin")
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

    ybytes = T * D * 4  # f32 out
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
    Y = np.frombuffer(bo_y.read(ybytes, 0), dtype=np.float32).reshape(T, D)

    adiff = np.abs(Y - ref_f)
    relL2 = np.linalg.norm(Y - ref_f) / np.linalg.norm(ref_f)
    thr = 0.05                              # >> bf16-tanh ~1.2e-2 noise, << O(1) miscompile
    # per TIME-ROW even/odd (the dwconv tile-loop parity) AND per CHANNEL even/odd (the D-lane parity)
    per_row = adiff.mean(axis=1)            # [T]
    per_ch = adiff.mean(axis=0)             # [D]
    row_even_bad = int((per_row[0::2] > thr).sum()); row_odd_bad = int((per_row[1::2] > thr).sum())
    ch_even_bad = int((per_ch[0::2] > thr).sum());   ch_odd_bad = int((per_ch[1::2] > thr).sum())
    print(f"[run] device time/iter: {dt*1e3:.3f} ms   ([T={T},D={D}], k=9, TIME-MAJOR fused dwconv+silu)")
    print(f"[run] rel-L2 vs exact f32 silu(dwconv): {relL2:.4e}   max|Δ|={adiff.max():.4f}")
    print(f"[run] per-ROW  mean|Δ|>{thr}:  EVEN {row_even_bad}/{T//2}   ODD {row_odd_bad}/{T - T//2}")
    print(f"[run] per-CHAN mean|Δ|>{thr}:  EVEN {ch_even_bad}/{D//2}   ODD {ch_odd_bad}/{D//2}  (0/0 all => no miscompile)")
    print(f"[run] Y[0,:4]={Y[0,:4]}  ref={ref_f[0,:4]}")
    ok = (row_even_bad == 0 and row_odd_bad == 0 and ch_even_bad == 0 and ch_odd_bad == 0 and relL2 < 0.03)
    print(f"[run] TIME-MAJOR fused dwconv+silu on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
