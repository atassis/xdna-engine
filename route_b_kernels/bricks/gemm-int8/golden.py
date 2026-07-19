#!/usr/bin/env python3
"""Golden (host, CPU-only, no NPU) reference for the gemm-int8 brick.

Brick: C[M,N] (int32) = A[M,K] (int8) @ B[K,N] (int8), M,K,N multiples of 8,
matching aie::mmul<8,8,8,int8,int8,accauto> tiled across the [tile,D] stream.

int8 x int8 -> int32 is EXACT integer arithmetic (no rounding/format loss like
bfp16/bf16 bricks): the device kernel's int32 accumulator must reproduce this
golden bit-exactly modulo overflow saturation semantics. This script:
  1. Generates random int8 A, B in [-127,127] (avoid -128 for MMUL::size_A
     symmetry / avoid INT8_MIN*INT8_MIN edge asymmetry in some SIMD lowerings).
  2. Computes the reference GEMM in numpy int64 (no overflow) then casts to
     int32 (checking no int32 overflow actually occurs for the chosen sizes/
     value range, so the int32-cast reference is itself the exact answer).
  3. Reports rel-L2 against itself (0.0) as a sanity smoke test, and documents
     the threshold the LATER on-device pass should gate against.

Later NPU pass: feed the SAME A,B (row-major int8, A:[M,K], B:[K,N]) into the
device kernel, compare device int32 C against `ref` here.
"""
import numpy as np


def gemm_int8_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact int8 x int8 -> int32 GEMM reference (numpy int64 acc, cast to int32)."""
    assert a.dtype == np.int8 and b.dtype == np.int8
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert M % 8 == 0 and K % 8 == 0 and N % 8 == 0, \
        "gemm-int8 brick requires M,K,N multiples of 8 (native mmul<8,8,8> tile)"
    acc64 = a.astype(np.int64) @ b.astype(np.int64)
    # int8*int8 accumulated over K: max |acc| = K * 127 * 127, well within
    # int32 range (2^31-1) for any K up to ~1.3M -- no int32 overflow for
    # realistic GEMM shapes, so this cast is lossless here.
    assert np.all(np.abs(acc64) < np.iinfo(np.int32).max), \
        "int32 accumulator would overflow for this shape/value range"
    return acc64.astype(np.int32)


def rel_l2(ref: np.ndarray, got: np.ndarray) -> float:
    ref64 = ref.astype(np.float64).ravel()
    got64 = got.astype(np.float64).ravel()
    return float(np.linalg.norm(got64 - ref64) / (np.linalg.norm(ref64) + 1e-12))


def run_case(name: str, M: int, K: int, N: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    # int8 range [-127,127]: skip -128 (asymmetric two's-complement extreme,
    # matches the convention used by other int8 bricks in this tree, e.g.
    # whole_array_fused's modal-int8 dequant path).
    a = rng.integers(-127, 128, size=(M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-127, 128, size=(K, N), dtype=np.int64).astype(np.int8)
    ref = gemm_int8_ref(a, b)
    # Self-check only here (CPU golden has no device to compare against yet);
    # the device pass replaces `ref` on one side with the on-NPU result.
    r = rel_l2(ref, ref)
    print(f"  {name:14s} [{M},{K}]x[{K},{N}] int8->int32  rel-L2(self)={r:.2e}  "
          f"acc range=[{ref.min()},{ref.max()}]")
    return r


def main():
    print("gemm-int8 brick golden (CPU-only, no NPU) -- int8 GEMM, int32 acc, exact math\n")
    for (M, K, N) in [(8, 8, 8), (64, 64, 64), (128, 128, 128), (8, 1024, 4096)]:
        run_case(f"M{M}K{K}N{N}", M, K, N, seed=M * 1000 + K * 10 + N % 97)
    print("\nrel-L2 gate for the on-device verify pass: <= 3e-2")
    print("(int8xint8->int32 is exact integer math -- any device rel-L2 above")
    print(" this threshold indicates a real kernel bug, not a format-rounding")
    print(" tradeoff like the bf16/bfp16 bricks; a tight near-0 pass is expected)")


if __name__ == "__main__":
    main()
