#!/usr/bin/env python3
"""Numpy golden for the vectorized depthwise-conv1d k=9 kernel (dwconv1d.cc).

Node: the Parakeet-TDT FastConformer ConvModule depthwise conv1d (k=9, 'same',
per-channel) + BatchNorm-folded bias -- exactly the `out` of conv_module before
the SiLU, in scripts/parakeet_ref_encoder.py:

    ht = h.T ; hp = pad(ht, (4,4))
    out[c,t] = sum_{j=0..8} dw[c,j] * hp[c, t+j] + dwb[c]

GATE: rel-L2( kernel-model(bf16) , host-reference(fp32) ) <= 0.08.

  - host-reference: full fp32 depthwise conv on fp32 activations (the golden).
  - kernel-model:   what dwconv1d.cc computes -- bf16 in/weights, fp32 (accfloat)
    accumulate via the sliding FIR, bf16 round on store. Mirrors the on-chip
    sliding_mul path (incl. the L=32 chunking; mathematically identical to a
    direct per-t window, so we compute the window form and the bf16 rounding).

Uses the REAL Parakeet L0 depthwise weights+bias (artifacts/parakeet/encoder/L0)
so the test exercises the actual tap distribution. Channels = 1024, T = 400
(the frame count baked into dwconv1d_k9_bf16).

Also emits a single-channel tile (in[400], w[16]=taps+bias@9, out[400]) to the
scratch dir for the later on-NPU node harness (Tier B).
"""
import os
import sys
import numpy as np
from ml_dtypes import bfloat16

C, T, K, P = 1024, 400, 9, 4
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENC = os.path.join(REPO, "artifacts/parakeet/encoder")


def bf16(x):
    return x.astype(bfloat16).astype(np.float32)


def host_reference(x_f32, dw_f32, b_f32):
    """fp32 depthwise conv 'same' (correlation), bias added -- the golden."""
    hp = np.pad(x_f32, ((0, 0), (P, P)))  # [C, T+2P]
    out = np.zeros((C, T), dtype=np.float32)
    for j in range(K):
        out += dw_f32[:, j:j + 1] * hp[:, j:j + T]
    out += b_f32[:, None]
    return out


def kernel_model(x_f32, dw_f32, b_f32):
    """Bit-faithful model of dwconv1d.cc: bf16 in/weights, fp32 accumulate,
    bf16 output. Bias folded into the epilogue (carried in w[9] on chip)."""
    xb = bf16(x_f32)            # input arrives as bf16
    wb = bf16(dw_f32)           # taps stored bf16
    bf = b_f32.astype(np.float32)  # bias carried bf16 in w[9]; round it too:
    bf = bf16(b_f32)
    hp = np.pad(xb, ((0, 0), (P, P)))
    acc = np.zeros((C, T), dtype=np.float32)   # accfloat (fp32) accumulator
    for j in range(K):
        acc += wb[:, j:j + 1].astype(np.float32) * hp[:, j:j + T].astype(np.float32)
    acc = acc + bf[:, None]
    return bf16(acc)            # single bf16 round on store


def rel_l2(a, b):
    return float(np.linalg.norm((a - b).ravel()) / (np.linalg.norm(b.ravel()) + 1e-12))


def main():
    blk = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dw = np.load(f"{ENC}/L{blk}/conv.depthwise_conv.weight.npy")[:, 0, :].astype(np.float32)  # [C,9]
    bias = np.load(f"{ENC}/L{blk}/conv.depthwise_conv.bias.npy").astype(np.float32)           # [C]
    assert dw.shape == (C, K), dw.shape
    assert bias.shape == (C,), bias.shape

    rng = np.random.default_rng(0)
    # Representative ConvModule input scale (post-GLU activations are ~O(1)).
    x = rng.standard_normal((C, T)).astype(np.float32)

    ref = host_reference(x, dw, bias)
    ker = kernel_model(x, dw, bias)

    err = rel_l2(ker, ref)
    gate = 0.08
    ok = err <= gate
    print(f"[dwconv1d k=9] block L{blk}  C={C} T={T} K={K} P={P}")
    print(f"  rel-L2(kernel_bf16, host_fp32) = {err:.5e}   gate <= {gate}")
    print(f"  max|abs| ref={np.abs(ref).max():.4f}  ker={np.abs(ker).max():.4f}")
    print("  RESULT:", "PASS" if ok else "FAIL")

    # Emit a single-channel tile for the on-NPU node harness (Tier B).
    sp = os.environ.get("DWCONV_TILE_OUT")
    if sp:
        os.makedirs(sp, exist_ok=True)
        ch = 0
        wt = np.zeros(16, dtype=np.float32)
        wt[:K] = dw[ch]
        wt[K] = bias[ch]
        bf16(x[ch]).astype(bfloat16).tofile(os.path.join(sp, "in.bin"))
        wt.astype(bfloat16).tofile(os.path.join(sp, "w.bin"))
        ref[ch].astype(bfloat16).tofile(os.path.join(sp, "out_ref.bin"))
        print(f"  wrote channel-{ch} tile (in/w/out_ref .bin) -> {sp}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
