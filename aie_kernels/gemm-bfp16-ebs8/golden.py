#!/usr/bin/env python3
"""Golden (host, CPU-only, no NPU) reference for the gemm-bfp16-ebs8 brick.

Brick: C[M,N] (bf16, fp32-accumulated) = A[M,K] (bf16) @ B[K,N] (bf16), M,K,N
multiples of 8, matching the AIE2P TRUE-SYSTOLIC tile
`aie::block_vector<bfp16ebs8,64>` operands run through the free intrinsics
`mul_8x8_8x8T` / `mac_8x8_8x8T` (see mlir-aie/aie_kernels/aie2p/mm_bfp.cc,
the AMD reference this brick's on-device kernel follows) -- NOT the
bfloat16-emulated-via-bfp16 path that trips the OPEN llvm-aie #847 ICE.

Datapath being modeled (per-8x8-tile, matches the on-chip kernel):
  1. bf16 A,B tiles loaded from the resident [tile,D] stream.
  2. EACH operand is independently quantized to bfp16ebs8: blocks of 8
     elements along the shared (K) dimension share one 8-bit exponent (the
     block max-abs, floor(log2)), each element keeps an 8-bit signed
     mantissa, rounded to the mantissa grid ROUND-TO-NEAREST-EVEN (the
     project's banked convention for every bf16/bfp16 cast on the WER-gated
     path; the on-chip kernel sets `aie::rounding_mode::conv_even` before
     any bf16->bfp16ebs8 conversion).
  3. The 8x8x8 systolic tile MACs the two quantized operands, accumulating
     in fp32 (accfloat) -- only A/B are block-quantized, not the accumulator.
  4. Result is cast back to bf16 for the resident stream contract's C output.

This is the SAME bfp16ebs8 block-float quantization model already validated
in route_b_kernels/ffn_bfp16/golden_ffn_gemm.py (that script predicted the
rel-L2 a bfp16 mmul WOULD achieve once #847 is fixed); this brick's kernel
uses bfp16ebs8 directly (not the bf16-emulated-via-#847-macro path), so this
golden is the DIRECT host reference for the on-device rel-L2 gate, not just
a format-accuracy prediction.
"""
import numpy as np

EBS = 8          # elements-per-block-share-exponent (bfp16ebs8)
MANT_BITS = 8     # signed mantissa bits (incl. sign) for a bfp16ebs8 element


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bfloat16 -> fp32 (truncate low 16 mantissa bits w/ RNE)."""
    u = x.astype(np.float32).view(np.uint32)
    rounding_bias = ((u >> 16) & 1) + 0x7FFF
    u = (u + rounding_bias) & 0xFFFF0000
    return u.view(np.float32)


def to_bfp16ebs8_blocks(x: np.ndarray, axis: int) -> np.ndarray:
    """Quantize x to bfp16ebs8 along `axis`: every EBS consecutive elements
    share the block max-abs exponent, mantissas rounded to MANT_BITS (signed,
    round-to-nearest-even). Returns fp32 (dequantized) for a numpy-native
    matmul reference."""
    x = x.astype(np.float32)
    x = np.moveaxis(x, axis, -1)
    shp = x.shape
    assert shp[-1] % EBS == 0, f"K={shp[-1]} not divisible by EBS={EBS}"
    xb = x.reshape(*shp[:-1], shp[-1] // EBS, EBS)
    maxabs = np.max(np.abs(xb), axis=-1, keepdims=True)
    maxabs = np.where(maxabs == 0, 1.0, maxabs)
    shared_exp = np.floor(np.log2(maxabs))          # one 8-bit exponent/block
    scale = 2.0 ** (shared_exp - (MANT_BITS - 2))    # mantissa quantum
    q = np.round(xb / scale) * scale                 # RNE onto the grid
    q = q.reshape(*shp)
    return np.moveaxis(q, -1, axis)


def gemm_fp32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) @ b.astype(np.float32)


def gemm_bfp16_ebs8_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Host reference for the on-device kernel: A,B -> bf16 -> bfp16ebs8
    (blocked along K for BOTH operands, matching the systolic tile's shared
    contraction dimension) -> fp32-accumulate matmul -> bf16 output."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % 8 == 0 and K % 8 == 0 and N % 8 == 0, \
        "gemm-bfp16-ebs8 brick requires M,K,N multiples of 8 (native mmul<8,8,8> tile)"
    a_q = to_bfp16ebs8_blocks(to_bf16(a), axis=1)   # block along K (A's cols)
    b_q = to_bfp16ebs8_blocks(to_bf16(b), axis=0)   # block along K (B's rows)
    acc_fp32 = a_q @ b_q                             # fp32 accumulate
    return to_bf16(acc_fp32)                         # bf16 stream-contract output


def rel_l2(ref: np.ndarray, got: np.ndarray) -> float:
    ref64 = ref.astype(np.float64).ravel()
    got64 = got.astype(np.float64).ravel()
    return float(np.linalg.norm(got64 - ref64) / (np.linalg.norm(ref64) + 1e-12))


def run_case(name: str, M: int, K: int, N: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = (rng.standard_normal((K, N)).astype(np.float32) * (1.0 / np.sqrt(K)))
    ref_fp32 = gemm_fp32(a, b)
    got_bfp16 = gemm_bfp16_ebs8_ref(a, b)
    r = rel_l2(ref_fp32, got_bfp16)
    print(f"  {name:16s} [{M},{K}]x[{K},{N}]  rel-L2 vs fp32={r:.5f}  "
          f"{'PASS' if r <= 0.08 else 'FAIL'} (<=0.08)")
    return r


def main():
    print("gemm-bfp16-ebs8 brick golden (CPU-only, no NPU)")
    print("bf16 in/out, TRUE bfp16ebs8 x bfp16ebs8 systolic-tile datapath sim\n")
    worst = 0.0
    for (M, K, N) in [(8, 8, 8), (64, 64, 64), (128, 128, 128), (8, 1024, 4096)]:
        worst = max(worst, run_case(f"M{M}K{K}N{N}", M, K, N,
                                     seed=M * 1000 + K * 10 + N % 97))
    print(f"\n worst rel-L2 (vs fp32 reference) = {worst:.5f}")
    print(" On-device verify pass: feed the SAME A,B (row-major bf16, A:[M,K],")
    print(" B:[K,N]) into gemm_bfp16_ebs8_<M>x<K>x<N>() and compare device bf16")
    print(" C against `gemm_bfp16_ebs8_ref(a, b)` here (NOT against fp32 `ref`,")
    print(" which only bounds the format's own accuracy loss).")
    print(" Gate: rel-L2(device, gemm_bfp16_ebs8_ref) <= 3e-2 -- this brick's")
    print(" quantization is deterministic (same block-exponent + RNE-mantissa")
    print(" rule on both sides), so device-vs-this-golden should be MUCH tighter")
    print(" than the vs-fp32 format-loss bound printed above; a device rel-L2")
    print(" near that fp32 bound (not near 0) would indicate the RNE/conv_even")
    print(" rounding mode is wrong or a real kernel bug, not a rounding nuance.")


if __name__ == "__main__":
    main()
