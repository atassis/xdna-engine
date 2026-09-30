#!/usr/bin/env python3
"""Golden for the Parakeet FFN/proj GEMMs (A2): numpy fp32 GEMM reference + a
bfp16ebs8 block-floating-point DATAPATH simulation, to predict the rel-L2 the
bfp16 8x8x8 mmul kernel WOULD achieve once the Peano #847 ICE/miscompile is
fixed. CPU-only; no NPU.

Parakeet-tdt-0.6b-v3 FastConformer: d_model=1024, d_ff=4096 (rust/npu-parakeet/
src/config.rs). FFN has two GEMMs:
  fc1: [T,1024] @ [1024,4096]
  fc2: [T,4096] @ [4096,1024]

bfp16ebs8 = the TRUE AIE2P systolic format: blocks of 8 elements along the
shared (K) dimension share ONE 8-bit exponent; each element keeps an 8-bit
signed mantissa. This sim mirrors the `vconv.bfp16ebs8.fp32` the kernel emits.
The mmul accumulates in fp32 (accfloat), so only the A/B operands are blocked.

Gate: rel-L2 <= 0.08 (spec tight-GEMM node wants <0.02 but bfp16 is WER-gated).
"""
import numpy as np

EBS = 8                     # elements-per-block-share-exponent (bfp16ebs8)
MANT_BITS = 8               # signed mantissa incl. sign for bfp16ebs8 operand


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bfloat16 -> fp32 (truncate low 16 mantissa bits w/ RNE)."""
    u = x.astype(np.float32).view(np.uint32)
    # round-to-nearest-even on the 16 dropped bits
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    u = (u + rounding_bias) & 0xFFFF0000
    return u.view(np.float32)


def to_bfp16_blocks(x: np.ndarray, axis: int) -> np.ndarray:
    """Quantize x to bfp16ebs8 along `axis`: every EBS consecutive elements
    share the max exponent, mantissas rounded to MANT_BITS. Returns fp32."""
    x = x.astype(np.float32)
    x = np.moveaxis(x, axis, -1)
    shp = x.shape
    assert shp[-1] % EBS == 0, f"K={shp[-1]} not divisible by EBS={EBS}"
    xb = x.reshape(*shp[:-1], shp[-1] // EBS, EBS)
    # shared exponent per block = exponent of the block max-abs
    maxabs = np.max(np.abs(xb), axis=-1, keepdims=True)
    maxabs = np.where(maxabs == 0, 1.0, maxabs)
    shared_exp = np.floor(np.log2(maxabs))               # block scale
    scale = 2.0 ** (shared_exp - (MANT_BITS - 2))        # quantum per block
    q = np.round(xb / scale) * scale                      # RNE to mantissa grid
    q = q.reshape(*shp)
    return np.moveaxis(q, -1, axis)


def gemm_fp32(a, b):
    return a.astype(np.float32) @ b.astype(np.float32)


def gemm_bfp16(a, b):
    """Model the kernel: A,B -> bf16 -> bfp16ebs8 blocks along K, mac in fp32."""
    a_q = to_bfp16_blocks(to_bf16(a), axis=1)            # block along K (cols of A)
    b_q = to_bfp16_blocks(to_bf16(b), axis=0)            # block along K (rows of B)
    return a_q @ b_q                                      # fp32 accumulate


def relL2(ref, got):
    return float(np.linalg.norm((got - ref).ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12))


def run_case(name, T, K, N, seed):
    rng = np.random.default_rng(seed)
    # encoder activations ~ unit-ish; weights small (post-LN scale)
    a = rng.standard_normal((T, K)).astype(np.float32)
    b = (rng.standard_normal((K, N)).astype(np.float32) * (1.0 / np.sqrt(K)))
    ref = gemm_fp32(a, b)
    got = gemm_bfp16(a, b)
    r = relL2(ref, got)
    print(f"  {name:14s} [{T},{K}]x[{K},{N}]  rel-L2={r:.5f}  "
          f"{'PASS' if r <= 0.08 else 'FAIL'} (<=0.08)"
          f"{'  tight-PASS' if r <= 0.02 else ''}")
    return r


def main():
    print("Parakeet FFN/proj GEMM bfp16ebs8 datapath sim (CPU golden, predicts NPU rel-L2)")
    print("d_model=1024 d_ff=4096; T=encoder frames (compute-bound M>=8 regime)\n")
    worst = 0.0
    for T in (8, 64, 256):
        print(f" T={T}:")
        worst = max(worst, run_case("FFN.fc1", T, 1024, 4096, seed=T * 10 + 1))
        worst = max(worst, run_case("FFN.fc2", T, 4096, 1024, seed=T * 10 + 2))
        # proj GEMMs (Q/K/V/out) at d_model square
        worst = max(worst, run_case("attn.proj", T, 1024, 1024, seed=T * 10 + 3))
    print(f"\n worst rel-L2 = {worst:.5f}  -> bfp16 format is "
          f"{'WITHIN' if worst <= 0.08 else 'OUTSIDE'} the 0.08 gate")
    print(" (this is the FORMAT accuracy; the kernel cannot run until Peano #847 is fixed)")


if __name__ == "__main__":
    main()
