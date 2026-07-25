#!/usr/bin/env python3
"""Golden (host/numpy) reference for the dequant-int4-group brick.

Mirrors dequant_int4_group.cc exactly:
  - int4 packed 2 values/byte, LOW nibble = even column, HIGH nibble = odd column,
    signed two's-complement in [-8, 7].
  - GROUP contiguous columns share one f32 scale + optional int8 zero-point:
        dequant(q) = (q - zp) * scale
  - narrow to bf16 with round-to-nearest-even (matches ::aie::rounding_mode::conv_even,
    NOT numpy/ml_dtypes truncation default).

Usage as a library: `dequant_int4_group_golden(packed, scale, zp, cols, group)`.
Run directly to self-check pack/unpack round-trip and print a couple of example rows.

This is the reference the LATER device pass will diff its on-device output against
via rel-L2 (see verify_spec in the task return / this brick's README-less contract):
  threshold: rel_L2 < 3e-2  (bf16 dequant + group-scale rounding budget)
"""
import numpy as np


def pack_int4_row(q_row: np.ndarray) -> np.ndarray:
    """Pack a row of signed int4 values (range [-8,7], even length) into bytes.
    Even index -> low nibble, odd index -> high nibble (matches the .cc contract).
    """
    assert q_row.ndim == 1
    assert q_row.shape[0] % 2 == 0
    q = q_row.astype(np.int16)
    assert np.all((q >= -8) & (q <= 7)), "int4 values must be in [-8, 7]"
    nibbles = (q & 0xF).astype(np.uint8)
    lo = nibbles[0::2]
    hi = nibbles[1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4_row(packed: np.ndarray, cols: int) -> np.ndarray:
    """Inverse of pack_int4_row: bytes -> signed int4 values in [-8,7], length cols."""
    out = np.empty(cols, dtype=np.int16)
    lo = (packed & 0x0F).astype(np.int16)
    hi = ((packed >> 4) & 0x0F).astype(np.int16)
    # sign-extend 4-bit -> int16
    lo = np.where(lo >= 8, lo - 16, lo)
    hi = np.where(hi >= 8, hi - 16, hi)
    out[0::2] = lo
    out[1::2] = hi
    return out


def _round_to_bf16_nearest_even(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 (represented back as f32) with round-to-nearest-even, matching
    ::aie::rounding_mode::conv_even (NOT truncation)."""
    x = x.astype(np.float32)
    bits = x.view(np.uint32)
    # round-to-nearest-even on the low 16 mantissa bits being dropped
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    rounded = (bits.astype(np.uint64) + rounding_bias) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def dequant_int4_group_golden(packed: np.ndarray, scale: np.ndarray,
                               zp: np.ndarray | None, cols: int, group: int) -> np.ndarray:
    """Host reference for dequant_int4_group_row<N,GROUP>(...).

    packed: uint8[ceil(cols/2)]
    scale:  float32[cols // group]
    zp:     int8[cols // group] or None (symmetric, zp == 0)
    returns: float32[cols], each value already rounded through the bf16 grid
             (round-to-nearest-even), i.e. directly comparable to the on-device
             bf16 output cast to float32.
    """
    assert cols % group == 0, "cols must be a multiple of group"
    q = unpack_int4_row(packed, cols).astype(np.float32)
    ngroups = cols // group
    group_idx = np.repeat(np.arange(ngroups), group)
    s = scale[group_idx]
    z = (zp[group_idx].astype(np.float32) if zp is not None
         else np.zeros(cols, dtype=np.float32))
    y = (q - z) * s
    return _round_to_bf16_nearest_even(y)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + 1e-12
    return float(num / den)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cols, group = 256, 64
    ngroups = cols // group

    # round-trip self-check: pack -> unpack -> identical int4 values
    q_true = rng.integers(-8, 8, size=cols, dtype=np.int16)
    packed = pack_int4_row(q_true)
    q_back = unpack_int4_row(packed, cols)
    assert np.array_equal(q_true, q_back), "pack/unpack round-trip mismatch"

    scale = (0.01 + rng.random(ngroups, dtype=np.float64)).astype(np.float32)
    zp = rng.integers(-2, 3, size=ngroups, dtype=np.int8)  # optional asymmetric case

    # symmetric (HAS_ZP=0 kernel build)
    y_sym = dequant_int4_group_golden(packed, scale, None, cols, group)
    # asymmetric (HAS_ZP=1 kernel build)
    y_asym = dequant_int4_group_golden(packed, scale, zp, cols, group)

    print(f"cols={cols} group={group} ngroups={ngroups}")
    print("q[:8]      =", q_true[:8])
    print("packed[:4] =", packed[:4])
    print("scale[:2]  =", scale[:2])
    print("zp[:2]     =", zp[:2])
    print("y_sym[:8]  =", y_sym[:8])
    print("y_asym[:8] =", y_asym[:8])

    # sanity: self-vs-self rel-L2 must be exactly 0
    assert rel_l2(y_sym, y_sym) == 0.0
    print("self-check OK (pack/unpack round-trip + rel_l2(x,x)==0)")
