#!/usr/bin/env python3
"""Golden (host numpy reference) for the gemm-bf16xbfp16 brick.

BRICK: gemm-bf16xbfp16  [group: matmul]  -- mixed bf16-activation x
bfp16ebs8-weight GEMM (aie::mmul<8,8,8,bfloat16,bfp16ebs8,accauto>).
WEIGHT-COMPRESSION only: the activation (A) is modeled as plain bf16
(round-trip through bf16, no block-floating-point quantization); only the
weight (B) is quantized to bfp16ebs8 blocks along the shared K dimension --
this predicts the rel-L2 the CPU-verified kernel above should hit once run
on-device, and is what the later DEVICE pass checks its NPU output against.

Generic: shapes are parameters (T activation rows, K contraction dim, N
output dim), not hard-coded to one model. Any caller (FFN fc1/fc2, attention
proj, decode gate proj, ...) reuses this same golden by passing its own
(T, K, N).

Usage:
    python3 golden.py                      # runs the built-in shape sweep
    python3 golden.py --T 64 --K 1024 --N 4096

Gate: rel-L2 <= 3e-2 (weight-only bfp16ebs8 quantization is materially
tighter than the bf16xbf16-emulated-through-bfp16 case in ffn_bfp16/
golden_ffn_gemm.py, since the activation operand is NOT block-quantized here
-- only compressed weights are).
"""
import argparse
import numpy as np

EBS = 8          # elements-per-block-share-exponent (bfp16ebs8)
MANT_BITS = 8     # signed mantissa bits incl. sign, per bfp16ebs8 element

DEFAULT_REL_L2_GATE = 3e-2


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bfloat16 -> fp32 (truncate low 16 mantissa
    bits with RNE). Models the activation operand, which stays plain bf16
    (no block-floating-point step) in this mixed brick."""
    u = x.astype(np.float32).view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    u = (u + rounding_bias) & 0xFFFF0000
    return u.view(np.float32)


def to_bfp16ebs8_blocks(x: np.ndarray, axis: int) -> np.ndarray:
    """Quantize x to bfp16ebs8 along `axis`: every EBS consecutive elements
    share one 8-bit exponent (the block max-abs), mantissas rounded to
    MANT_BITS (signed). Returns fp32 (dequantized) values -- this is the
    weight-compression datapath the packed bfp16ebs8 buffer represents once
    unpacked by the systolic array's block-vector loader."""
    x = x.astype(np.float32)
    x = np.moveaxis(x, axis, -1)
    shp = x.shape
    assert shp[-1] % EBS == 0, f"dim={shp[-1]} not divisible by EBS={EBS}"
    xb = x.reshape(*shp[:-1], shp[-1] // EBS, EBS)
    maxabs = np.max(np.abs(xb), axis=-1, keepdims=True)
    maxabs = np.where(maxabs == 0, 1.0, maxabs)
    shared_exp = np.floor(np.log2(maxabs))
    scale = 2.0 ** (shared_exp - (MANT_BITS - 2))
    q = np.round(xb / scale) * scale
    q = q.reshape(*shp)
    return np.moveaxis(q, -1, axis)


def gemm_fp32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full-precision reference GEMM: C = A @ B."""
    return a.astype(np.float32) @ b.astype(np.float32)


def gemm_bf16xbfp16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Model of the gemm_bf16xbfp16 kernel's datapath:
      A -> bf16 (activation, NOT block-quantized)
      B -> bf16 -> bfp16ebs8 blocks along K (rows of B)   (weight, compressed)
      C = A_bf16 @ B_bfp16ebs8, accumulated in fp32 (accfloat), as the mmul's
          accauto accumulator does on-core.
    """
    a_bf16 = to_bf16(a)
    b_q = to_bfp16ebs8_blocks(to_bf16(b), axis=0)  # block along K (rows of B)
    return a_bf16 @ b_q  # fp32 accumulate


def rel_l2(ref: np.ndarray, got: np.ndarray) -> float:
    return float(np.linalg.norm((got - ref).ravel()) /
                 (np.linalg.norm(ref.ravel()) + 1e-12))


def run_case(name: str, T: int, K: int, N: int, seed: int,
             gate: float = DEFAULT_REL_L2_GATE) -> float:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((T, K)).astype(np.float32)
    b = (rng.standard_normal((K, N)).astype(np.float32) * (1.0 / np.sqrt(K)))
    ref = gemm_fp32(a, b)
    got = gemm_bf16xbfp16(a, b)
    r = rel_l2(ref, got)
    print(f"  {name:14s} [{T},{K}]x[{K},{N}]  rel-L2={r:.5f}  "
          f"{'PASS' if r <= gate else 'FAIL'} (<={gate})")
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--T", type=int, default=None, help="activation rows")
    ap.add_argument("--K", type=int, default=None, help="contraction dim")
    ap.add_argument("--N", type=int, default=None, help="output dim")
    ap.add_argument("--gate", type=float, default=DEFAULT_REL_L2_GATE)
    args = ap.parse_args()

    print("gemm-bf16xbfp16 brick: bf16-activation x bfp16ebs8-weight GEMM "
          "datapath sim (CPU golden, predicts NPU rel-L2)\n")

    if args.T and args.K and args.N:
        r = run_case(f"T{args.T}", args.T, args.K, args.N, seed=1,
                     gate=args.gate)
        print(f"\n rel-L2 = {r:.5f} -> "
              f"{'WITHIN' if r <= args.gate else 'OUTSIDE'} the {args.gate} gate")
        return

    # Built-in generic shape sweep: small dims (exercise the 8x8x8 tile
    # boundary directly) + representative transformer-scale FFN/proj dims.
    worst = 0.0
    for T in (8, 64, 256):
        print(f" T={T}:")
        worst = max(worst, run_case("gemm.8x8x8", 8, 8, 8, seed=T * 10 + 0, gate=args.gate))
        worst = max(worst, run_case("ffn.fc1", T, 1024, 4096, seed=T * 10 + 1, gate=args.gate))
        worst = max(worst, run_case("ffn.fc2", T, 4096, 1024, seed=T * 10 + 2, gate=args.gate))
        worst = max(worst, run_case("attn.proj", T, 1024, 1024, seed=T * 10 + 3, gate=args.gate))

    print(f"\n worst rel-L2 = {worst:.5f}  -> gemm-bf16xbfp16 is "
          f"{'WITHIN' if worst <= args.gate else 'OUTSIDE'} the {args.gate} gate")
    print(" (this is the FORMAT/datapath accuracy prediction for the CPU-compiled "
          "kernel; device pass checks the actual NPU output against this same "
          "golden with the same gate)")


if __name__ == "__main__":
    main()
