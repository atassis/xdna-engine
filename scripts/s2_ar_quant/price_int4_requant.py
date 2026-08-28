#!/usr/bin/env python3
"""Price int4 requantization of the S2 AR transformer's weights -- the number the
epic (tts-s2-npu-mvp) names as an unmet precondition ("until that number exists
further AR brick work is unpriced").

The AR weight set (layers.*/fast_layers.*/embeddings/output in the real GGUF) is
already Q6_K on disk, not bf16 -- gguf_shapes.py confirms zero f32/bf16 among the
4,561,852,416 AR params. So "int4 vs bf16" is the wrong comparison; the real
question is int4 vs the Q6_K IT WOULD REPLACE, at whatever additional error int4
adds on top of Q6_K's own (already-lossy) rounding.

    python3 price_int4_requant.py [--gguf PATH] [--layers N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from q6k_dequant import QK_K, load_q6k_from_index, read_gguf_index  # noqa: E402

DEFAULT_GGUF = HERE.parents[2] / "s2.cpp" / "models" / "s2-pro-q6_k.gguf"

# The AR transformer's own tensor-name prefixes, as opposed to the codec sub-tree's
# "c.*" (build_transformer's separate call site in s2_codec.cpp) -- see s2_model.h.
AR_NAME_PREFIXES = ("layers.", "fast_layers.", "embeddings.weight", "fast_embeddings.weight",
                    "codebook_embeddings.weight", "norm.weight", "fast_norm.weight",
                    "fast_output.weight")

# GROUP=64 symmetric, matching route_b_kernels/bricks/dequant-int4-group.cc's
# default instantiation (dequant_int4_group_row<16,64>, HAS_ZP=0).
GROUP = 64


def quantize_group(x, bits, group=GROUP):
    """Symmetric per-group absmax quantization to `bits`, zp=0 -- generalizes
    dequant_int4_group's contract (dequant(q) = q * scale) to any bit width, so
    int8 can run alongside int4 as a sanity control on the same tensors."""
    qmax = 2 ** (bits - 1) - 1
    n = x.shape[0]
    pad = (-n) % group
    xp = np.pad(x, (0, pad)) if pad else x
    g = xp.reshape(-1, group)
    scale = np.abs(g).max(axis=1, keepdims=True) / qmax
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(g / scale), -qmax - 1, qmax)
    recon = (q * scale).reshape(-1)[:n]
    return recon.astype(np.float32)


def rel_l2(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    den = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / den) if den else float(np.linalg.norm(a - b))


AR_TENSOR_KINDS = [
    "attention.wqkv.weight", "attention.wo.weight",
    "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=str(DEFAULT_GGUF))
    ap.add_argument("--layers", type=int, default=8,
                     help="main-transformer layers to sample (0..N-1); fast_layers sampled in full (4)")
    args = ap.parse_args()

    f, data_start, infos = read_gguf_index(args.gguf)

    # Real weight-set size, derived from this GGUF's own tensor table rather than
    # a copied-in constant (cross-checks against the epic's own "4.56B params").
    ar_params = sum(np.prod(dims) for name, (dims, ty, _) in infos.items()
                    if name.startswith(AR_NAME_PREFIXES))
    print(f"AR weight set (from this GGUF's tensor table): {ar_params:,} params")

    rows = []  # (name, n_params, rel_l2_int4, rel_l2_int8)
    names = []
    for layer in range(args.layers):
        names += [f"layers.{layer}.{k}" for k in AR_TENSOR_KINDS]
    for layer in range(4):
        names += [f"fast_layers.{layer}.{k}" for k in AR_TENSOR_KINDS]
    names += ["embeddings.weight", "fast_embeddings.weight",
              "codebook_embeddings.weight", "fast_output.weight"]

    total_params_sampled = 0
    weighted_num4 = 0.0
    weighted_num8 = 0.0
    for name in names:
        if name not in infos or infos[name][1] != 14:  # 14 = GGUF Q6_K; skip f16 norms etc.
            continue
        q6k, n = load_q6k_from_index(f, data_start, infos, name)
        rl2_4 = rel_l2(quantize_group(q6k, bits=4), q6k)
        rl2_8 = rel_l2(quantize_group(q6k, bits=8), q6k)
        rows.append((name, n, rl2_4, rl2_8))
        total_params_sampled += n
        weighted_num4 += rl2_4 * n
        weighted_num8 += rl2_8 * n

    print(f"{'tensor':45s} {'params':>12s}  {'rel-L2 int4':>12s}  {'rel-L2 int8':>12s}")
    for name, n, rl2_4, rl2_8 in rows:
        print(f"{name:45s} {n:12,d}  {rl2_4:12.4e}  {rl2_8:12.4e}")

    agg4 = weighted_num4 / total_params_sampled
    agg8 = weighted_num8 / total_params_sampled
    worst = max(rows, key=lambda r: r[2])
    print(f"\nsampled {len(rows)} tensors, {total_params_sampled:,} params "
          f"({100*total_params_sampled/ar_params:.1f}% of the full AR weight set)")
    print(f"param-weighted mean rel-L2 vs q6_k, GROUP={GROUP} symmetric absmax:")
    print(f"  int4: {agg4:.4e}   int8: {agg8:.4e}   (ratio {agg4/agg8:.1f}x)")
    print(f"worst int4 tensor: {worst[0]} at {worst[2]:.4e}")

    # Byte-budget consequence over the FULL AR weight set. Exact current on-disk size
    # (mixed Q6_K + a few f16 norm vectors, per this GGUF's own tensor table) vs the two
    # requant targets applied uniformly (both bricks operate on any dequantized f32/bf16
    # input, so the tiny f16 slice would requantize the same as the Q6_K majority).
    TYPE_BYTES_PER_ELEM = {1: 2.0, 14: 210.0 / QK_K}
    q6k_bytes = sum(np.prod(dims) * TYPE_BYTES_PER_ELEM[ty]
                    for name, (dims, ty, _) in infos.items() if name.startswith(AR_NAME_PREFIXES))
    int4_bytes = ar_params * (4.25 / 8.0)   # 4 bit + 16-bit f16 scale per 64 -> 4.25 bit/w
    int8_bytes = ar_params * (8.25 / 8.0)   # 8 bit + 16-bit f16 scale per 64 -> 8.25 bit/w
    print(f"\nfull AR weight set: {ar_params:,} params")
    print(f"  current (Q6_K + f16 norms, on disk today): {q6k_bytes/1e9:.3f} GB")
    print(f"  int4 GROUP={GROUP} f16-scale (this brick): {int4_bytes/1e9:.3f} GB "
          f"({q6k_bytes/int4_bytes:.2f}x smaller than Q6_K -- NOT 4x, because Q6_K is "
          f"already ~6.56 bit/weight, not bf16's 16)")
    print(f"  int8 GROUP={GROUP} f16-scale:               {int8_bytes/1e9:.3f} GB "
          f"({int8_bytes/q6k_bytes:.2f}x LARGER than Q6_K -- int8 buys accuracy, not bytes, here)")
    for label, bw in [("measured pure-read (npu-lpddr-read-scaling-and-peak)", 52.7),
                       ("Infinity Fabric read ceiling (the-wall-is-the-fabric-not-lpddr)", 62.7)]:
        print(f"  streamed once at {bw} GB/s [{label}]:")
        print(f"    Q6_K : {1000*q6k_bytes/bw/1e9:.1f} ms")
        print(f"    int4 : {1000*int4_bytes/bw/1e9:.1f} ms")


if __name__ == "__main__":
    main()
