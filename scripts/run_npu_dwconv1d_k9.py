#!/usr/bin/env python3
"""On-device validation of the k=9 vectorized depthwise-conv1d (A1, sliding_mul).

Runs the dwconv1d_k9_bf16 IRON design (C=1024, T=400, KW=16) on the XDNA2 NPU
via pyxrt and compares the readback against the fp32 host reference from
route_b_kernels/dwconv1d/golden_dwconv1d_k9.py (the Parakeet ConvModule depthwise
conv: out[c,t] = sum_{j=0..8} dw[c,j]*pad(x)[c,t+j] + bias[c]).

Gate: rel-L2 <= 0.08 AND corr >= 0.99 vs the fp32 golden.

IRON host ABI (eltwise_mul/test.cpp): opcode=3;
  kernel(opcode, instr[gid1,cacheable], n_instr, X[gid3], W[gid4], Y[gid5])
X = [C*T] bf16, W = [C*KW] bf16 (taps[0..8] + bias@9), Y = [C*T] bf16.
"""
import argparse, os, sys, time
import numpy as np
from ml_dtypes import bfloat16

C, T, KW, K, P = 1024, 400, 16, 9, 4
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(REPO, "mlir-aie/programming_examples/ml/dwconv1d/build")
ENC = os.path.join(REPO, "artifacts/parakeet/encoder")


def bf16(x):
    return x.astype(bfloat16).astype(np.float32)


def host_reference(x_f32, dw_f32, b_f32):
    hp = np.pad(x_f32, ((0, 0), (P, P)))
    out = np.zeros((C, T), dtype=np.float32)
    for j in range(K):
        out += dw_f32[:, j:j + 1] * hp[:, j:j + T]
    out += b_f32[:, None]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", default=f"{EX}/final.xclbin")
    ap.add_argument("--insts", default=f"{EX}/insts.bin")
    ap.add_argument("--block", type=int, default=0)
    a = ap.parse_args()

    dw = np.load(f"{ENC}/L{a.block}/conv.depthwise_conv.weight.npy")[:, 0, :].astype(np.float32)
    bias = np.load(f"{ENC}/L{a.block}/conv.depthwise_conv.bias.npy").astype(np.float32)
    assert dw.shape == (C, K), dw.shape
    assert bias.shape == (C,), bias.shape

    rng = np.random.default_rng(0)
    x = rng.standard_normal((C, T)).astype(np.float32)

    ref_f = host_reference(x, dw, bias)               # fp32 golden

    # Build device inputs (bf16). Weight tile: taps[0..8], bias@9, rest 0.
    xb = bf16(x).astype(bfloat16)
    w = np.zeros((C, KW), dtype=np.float32)
    w[:, :K] = dw
    w[:, K] = bias
    wb = w.astype(bfloat16)

    X = np.ascontiguousarray(xb).reshape(-1)
    W = np.ascontiguousarray(wb).reshape(-1)

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

    Ybytes = C * T * 2
    bo_instr = pyxrt.bo(d, instr.nbytes, pyxrt.bo.cacheable, k.group_id(1))
    bo_x = pyxrt.bo(d, X.nbytes, pyxrt.bo.host_only, k.group_id(3))
    bo_w = pyxrt.bo(d, W.nbytes, pyxrt.bo.host_only, k.group_id(4))
    bo_y = pyxrt.bo(d, Ybytes, pyxrt.bo.host_only, k.group_id(5))

    bo_instr.write(instr.tobytes(), 0);            bo_instr.sync(TO)
    bo_x.write(X.view(np.uint16).tobytes(), 0);    bo_x.sync(TO)
    bo_w.write(W.view(np.uint16).tobytes(), 0);    bo_w.sync(TO)

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
    Y = np.frombuffer(bo_y.read(Ybytes, 0), dtype=np.uint16).view(bfloat16).reshape(C, T)
    yf = Y.astype(np.float32)

    # Diagnostics: standard, flipped-tap (conv vs corr), and no-bias references.
    def rel(a, b):
        return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))
    ref_flip = host_reference(x, dw[:, ::-1].copy(), bias)
    ref_nobias = host_reference(x, dw, np.zeros_like(bias))
    ref_flip_nobias = host_reference(x, dw[:, ::-1].copy(), np.zeros_like(bias))
    print(f"[diag] rel-L2 vs standard      = {rel(yf, ref_f):.5e}")
    print(f"[diag] rel-L2 vs flipped-tap   = {rel(yf, ref_flip):.5e}")
    print(f"[diag] rel-L2 vs no-bias       = {rel(yf, ref_nobias):.5e}")
    print(f"[diag] rel-L2 vs flip+no-bias  = {rel(yf, ref_flip_nobias):.5e}")
    np.save("$REPO/scratchpad_dwconv_Y.npy", yf)

    a_flat, r_flat = yf.ravel(), ref_f.ravel()
    rel_l2 = float(np.linalg.norm(a_flat - r_flat) / (np.linalg.norm(r_flat) + 1e-12))
    corr = float(np.corrcoef(a_flat, r_flat)[0, 1])
    adiff = np.abs(yf - ref_f)
    per_ch_max = adiff.max(axis=1)
    bad_ch = int((per_ch_max > 0.1 * (np.abs(ref_f).max(axis=1) + 1e-6)).sum())

    print(f"[run] device time/iter: {dt*1e3:.3f} ms  (C={C} x T={T}, k={K})")
    print(f"[run] rel-L2={rel_l2:.5e}  corr={corr:.6f}  max|d|={adiff.max():.4f}  mean|d|={adiff.mean():.6f}")
    print(f"[run] per-channel >10% rel: {bad_ch}/{C}")
    print(f"[run] Y[0,:5]={yf[0,:5]}  ref={ref_f[0,:5]}")
    ok = (rel_l2 <= 0.08) and (corr >= 0.99)
    print(f"[run] dwconv1d k=9 (sliding_mul) on NPU: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
