#!/usr/bin/env python3
"""Host numpy golden for the gemv-int8 brick (gemv_int8.cc).

Generic int8 x int8 -> int32 matvec, M=1 padded to GEMV_M rows for the mmul
r-dim (row 0 = live query, rows 1..M-1 are zero padding). Provides both the
raw i32 accumulator (matches gemv_i8_i8_i32) and the fused-dequant f32 output
(matches gemv_i8_i8_f32, S = scale_a * w_scale).

This is the HOST reference the later DEVICE pass checks rel-L2 against -- it
does not touch the device or the compiled .o, it only mirrors the kernel's
arithmetic (int8xint8 -> int32 accumulate, then optional *S dequant to f32).

Usage (self-check when run directly):
    python3 golden.py
"""
from __future__ import annotations

import numpy as np

# Same default tile shape as the kernel's compile-time macros.
GEMV_M = 8   # padded row count (r); only row 0 is the "real" query
GEMV_K = 768
GEMV_N = 768


def gemv_int8_i32(a_i8: np.ndarray, b_i8: np.ndarray) -> np.ndarray:
    """Raw i32 accumulator, mirrors gemv_i8_i8_i32.

    a_i8: [M, K] int8, row-major (row 0 = live query row, rest zero padding).
    b_i8: [K, N] int8, row-major (resident weight tile).
    returns: [M, N] int32.
    """
    assert a_i8.dtype == np.int8 and b_i8.dtype == np.int8
    m, k = a_i8.shape
    k2, n = b_i8.shape
    assert k == k2, f"K mismatch: A has {k}, B has {k2}"
    # Widen to i32 before the multiply-accumulate -- this is exactly the
    # int8*int8 -> int32 reduction the AIE2P mmul intrinsic performs; numpy's
    # matmul would silently overflow if left in int8.
    return (a_i8.astype(np.int32) @ b_i8.astype(np.int32)).astype(np.int32)


def gemv_int8_f32(a_i8: np.ndarray, b_i8: np.ndarray, scale: float) -> np.ndarray:
    """Fused-dequant f32 output, mirrors gemv_i8_i8_f32 (S = scale_a * w_scale)."""
    acc_i32 = gemv_int8_i32(a_i8, b_i8)
    return acc_i32.astype(np.float32) * np.float32(scale)


def make_padded_query(row_i8: np.ndarray, m_pad: int = GEMV_M) -> np.ndarray:
    """Pad a single [K] int8 query row to [m_pad, K] with zero rows, matching
    the kernel's M=1-padded-to-r convention (row 0 = live row)."""
    k = row_i8.shape[0]
    out = np.zeros((m_pad, k), dtype=np.int8)
    out[0, :] = row_i8
    return out


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    query = rng.integers(-127, 128, size=(GEMV_K,), dtype=np.int8)
    a = make_padded_query(query, GEMV_M)
    b = rng.integers(-127, 128, size=(GEMV_K, GEMV_N), dtype=np.int8)

    c_i32 = gemv_int8_i32(a, b)
    assert c_i32.shape == (GEMV_M, GEMV_N)
    # Row 0 is the only live row; rows 1.. must be exactly zero (zero query
    # rows dotted with any weight column are always 0).
    assert np.all(c_i32[1:, :] == 0), "padding rows must reduce to exactly 0"

    scale_a = 0.03
    w_scale = 0.05
    S = scale_a * w_scale
    c_f32 = gemv_int8_f32(a, b, S)
    expected_f32 = c_i32.astype(np.float64) * S
    err = rel_l2(c_f32, expected_f32)
    print(f"self-check rel-L2 (i32 vs scaled f32 reference): {err:.3e}")
    assert err < 1e-6

    print("golden.py self-check OK: "
          f"A{a.shape} int8 @ B{b.shape} int8 -> C{c_i32.shape} int32, "
          f"row0 sample out[0,:4]={c_i32[0, :4]}")
