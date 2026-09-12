#!/usr/bin/env python3
"""Golden for mv_scores_mmul.cc's sc_matvec_mmul_bf16_bf16.

Gates the TILE ORDER and the loop structure by walking the kernel's own index arithmetic in
numpy. It does NOT gate device arithmetic: the emulated mmul quantizes both operands to bfp16
(shared exponent per 8), which this does not model -- that needs the device.

Reference: c[p] = sum_d a[p][d] * b[d], f32 accumulate over bf16 inputs, which is what the
shipped matvec_vectorized_rtk computes.
"""
import numpy as np
from ml_dtypes import bfloat16

MM_M = MM_K = MM_N = 8


def pack_a(a):
    """[m, k] row-major -> [k/8][m/8][8 d][8 p], the order the kernel's pA walk reads.

    Offset of (p, d) = (d//8)*m*8 + (p//8)*64 + (d%8)*8 + (p%8).
    """
    m, k = a.shape
    assert m % MM_N == 0 and k % MM_K == 0, "K008: m and k must tile"
    # [m/8, 8p, k/8, 8d] -> transpose to [k/8, m/8, 8d, 8p]
    return (a.reshape(m // MM_N, MM_N, k // MM_K, MM_K)
             .transpose(2, 0, 3, 1)
             .reshape(-1))


def emulate(packed, b, m, k):
    """Walk exactly what the kernel does: pA = a + pb*64, stride pt*64 per d-chunk; B operand is
    the 64 contiguous elements read as [K=8][N=8] row-major; A operand is b's 8-wide d-chunk
    replicated across all 8 rows."""
    pt, kt = m // MM_N, k // MM_K
    c = np.zeros(m, dtype=np.float32)
    for pb in range(pt):
        acc = np.zeros((MM_M, MM_N), dtype=np.float32)
        for d in range(kt):
            off = d * pt * MM_K * MM_N + pb * MM_K * MM_N
            B = packed[off:off + MM_K * MM_N].astype(np.float32).reshape(MM_K, MM_N)
            A = np.tile(b[d * MM_K:(d + 1) * MM_K].astype(np.float32), (MM_M, 1))
            acc += A @ B
        c[pb * MM_N:(pb + 1) * MM_N] = acc[0]      # every row is the same query
    return c


def rel_of(x, y):
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))


def main():
    rng = np.random.default_rng(0)
    fails = 0
    for m, k in ((256, 128), (64, 128), (512, 64), (128, 256)):
        a = rng.standard_normal((m, k)).astype(bfloat16)
        b = rng.standard_normal(k).astype(bfloat16)
        # Order-matched reference: the kernel accumulates in MM_K-wide chunks, so a single
        # np.dot differs by f32 reassociation (~1e-7) and would make a bit-exact gate test
        # associativity rather than layout. Sum in the kernel's order to isolate the layout.
        af, bf = a.astype(np.float32), b.astype(np.float32)
        ref = np.zeros(m, dtype=np.float32)
        for d in range(k // MM_K):
            ref += af[:, d * MM_K:(d + 1) * MM_K] @ bf[d * MM_K:(d + 1) * MM_K]
        naive = af @ bf
        packed = pack_a(a)

        # 1. the packing is a bijection - no element lost or duplicated
        back = (packed.reshape(k // MM_K, m // MM_N, MM_K, MM_N)
                      .transpose(1, 3, 0, 2).reshape(m, k))
        bij = np.array_equal(back.astype(np.float32), a.astype(np.float32))

        # 2. the kernel's index walk reproduces the reference
        got = emulate(packed, b, m, k)
        # The BLOCKING gate is the bijection above: it is exact and it is what tests the tile
        # order. Bit-exactness between `got` and `ref` is NOT a layout gate -- both are numpy
        # expression trees over the same f32 values and differ by BLAS's own summation choices,
        # so demanding it would test numpy. rel-L2 is a note here, never the blocking gate.
        exact = rel_of(got, ref) < 1e-6
        rel = np.linalg.norm(got - naive) / max(np.linalg.norm(naive), 1e-30)

        ok = bij and exact
        fails += not ok
        print(f"m={m:4d} k={k:4d}  bijection={'PASS' if bij else 'FAIL'}  "
              f"vs-reference={'PASS' if exact else 'FAIL'}  "
              f"rel-L2-vs-order-matched={rel_of(got, ref):.3e}  rel-L2-vs-naive={rel:.3e}")
    print("GOLDEN PASS" if not fails else f"GOLDEN FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
