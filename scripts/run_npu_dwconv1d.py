#!/usr/bin/env python3
"""Run + validate depthwise-conv1d (k=5, 'same' pad) on the XDNA2 NPU via pyxrt.

The kernel processes one channel per ObjectFifo tile: in[400] bf16, w[16] bf16
(first 5 = taps), out[400] bf16, accumulating in fp32 and storing one bf16 round.
We mirror that exactly in numpy (upcast bf16->fp32, sequential 5-tap MAC, single
bf16 round) so the comparison is ~1 ULP, not a loose tolerance.

By default uses the REAL GigaAM-v3 block-0 depthwise weights (from the ONNX) on a
random input, which is the actual op the encoder needs. Use --random-weights for
a synthetic check.

IRON host ABI (from eltwise_mul/test.cpp): opcode=3;
  kernel(opcode, instr[gid1,cacheable], n_instr, X[gid3], W[gid4], Y[gid5])

Usage:
  .venv-iron/bin/python scripts/run_npu_dwconv1d.py --dry   # validate refs, no NPU
  .venv-iron/bin/python scripts/run_npu_dwconv1d.py         # REAL run (NPU must be free)
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

C, T, KW, K, P = 768, 400, 16, 5, 2
EX = "mlir-aie/programming_examples/ml/dwconv1d/build"
ONNX = "models/gigaam_v3_encoder_static.onnx"


def dwconv_ref_fp32(x_bf16, w_bf16):
    """True conv math in fp32 (inputs/weights are the bf16 values actually sent).
    x_bf16: [C,T] bf16, w_bf16: [C,KW] bf16 (first 5 taps used). returns [C,T] fp32."""
    x = x_bf16.astype(np.float32)
    w = w_bf16.astype(np.float32)
    out = np.zeros((C, T), dtype=np.float32)
    for t in range(T):
        acc = np.zeros(C, dtype=np.float32)
        for i in range(K):
            idx = t - P + i
            if 0 <= idx < T:
                acc = acc + w[:, i] * x[:, idx]   # per-channel scalar tap
        out[:, t] = acc
    return out


def bf16_ulp_distance(a_bf16, b_bf16):
    """Number of representable bf16 steps between two bf16 arrays (sign-magnitude
    ordered to a monotonic int key, then abs difference)."""
    def key(v):
        u = v.view(np.uint16).astype(np.int32)
        neg = (u & 0x8000) != 0
        # map to a monotonically increasing integer across the bf16 number line
        return np.where(neg, 0x8000 - (u & 0x7FFF), 0x8000 + (u & 0x7FFF))
    return np.abs(key(a_bf16) - key(b_bf16))


WEIGHT_NPY = "artifacts/block0_dwconv_weight.npy"


def load_real_weights():
    # extracted from the ONNX with .venv (which has onnx); .venv-iron has no onnx.
    if not os.path.exists(WEIGHT_NPY):
        sys.exit(f"missing {WEIGHT_NPY} — run: .venv/bin/python scripts/extract_block0.py")
    return np.load(WEIGHT_NPY).reshape(C, K).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts.bin")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    rng = np.random.RandomState(0)
    x = rng.uniform(-2.0, 2.0, size=(C, T)).astype(bfloat16)
    if a.random_weights:
        taps = rng.uniform(-0.5, 0.5, size=(C, K)).astype(np.float32)
        wsrc = "random"
    else:
        taps = load_real_weights()
        wsrc = "GigaAM-v3 block-0 depthwise_conv.weight"
    w = np.zeros((C, KW), dtype=np.float32)
    w[:, :K] = taps
    w = w.astype(bfloat16)

    ref_f = dwconv_ref_fp32(x, w)           # true conv in fp32
    ref_b = ref_f.astype(bfloat16)          # bf16-rounded truth
    print(f"[ref] weights={wsrc}  x{list(x.shape)} w{list(w.shape)} -> out[768,400] (fp32 truth)")

    X = np.ascontiguousarray(x).reshape(-1)
    W = np.ascontiguousarray(w).reshape(-1)
    if a.dry:
        print(f"[dry] X={X.nbytes}B W={W.nbytes}B Y={X.nbytes}B; ref_fp32[0,:5]={ref_f[0,:5]}")
        print("[dry] not touching the NPU.")
        return 0

    for p in (a.xclbin, a.insts):
        if not os.path.exists(p):
            sys.exit(f"missing {p} — build: (cd {os.path.dirname(EX)} && make NPU2=1)")
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

    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_w = pyxrt.bo(d, W.nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_y = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(5))

    bo_instr.write(instr.tobytes(), 0);              bo_instr.sync(TO)
    bo_x.write(X.view(np.uint16).tobytes(), 0);      bo_x.sync(TO)
    bo_w.write(W.view(np.uint16).tobytes(), 0);      bo_w.sync(TO)

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
    Y = np.frombuffer(bo_y.read(X.nbytes, 0), dtype=np.uint16).view(bfloat16).reshape(C, T)

    yf = Y.astype(np.float32)
    adiff = np.abs(yf - ref_f)                       # NPU vs fp32 truth
    denom = np.maximum(np.abs(ref_f), 1e-3)
    rel = adiff / denom
    ulp = bf16_ulp_distance(Y, ref_b)               # NPU vs bf16-rounded truth
    exact = int((ulp == 0).sum())
    within1 = int((ulp <= 1).sum())
    within2 = int((ulp <= 2).sum())
    N = C * T

    # per-channel check: a data-movement bug (wrong weights to a channel) shows as
    # a few channels with large error; pure rounding is spread ~uniformly.
    per_ch_max = adiff.max(axis=1)
    bad_ch = int((per_ch_max > 0.1).sum())

    print(f"[run] device time/iter: {dt*1e3:.3f} ms  ({C} channels x {T} steps, k={K})")
    print(f"[run] vs fp32 truth:  max|Δ|={adiff.max():.5f}  mean|Δ|={adiff.mean():.6f}  max_rel={rel.max():.4f}")
    print(f"[run] vs bf16 truth:  exact={exact}/{N} ({100*exact/N:.1f}%)  ≤1ulp={100*within1/N:.2f}%  ≤2ulp={100*within2/N:.2f}%  maxulp={int(ulp.max())}")
    print(f"[run] per-channel:    channels with max|Δ|>0.1: {bad_ch}/{C}  (0 => no mis-fed channel)")
    print(f"[run] Y[0,:5]={yf[0,:5]}  ref={ref_f[0,:5]}")
    # PASS: matches fp32 truth to within bf16 rounding (≤2 ulp everywhere) and no
    # systematically-wrong channel. Loose abs guard catches gross errors too.
    ok = (int(ulp.max()) <= 2) and (bad_ch == 0) and (adiff.max() < 0.05)
    print(f"[run] depthwise-conv1d k=5 on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
