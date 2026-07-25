#!/usr/bin/env python3
"""Host numpy golden reference for the cast-quant-bf16-int8 format brick.

Mirrors cast_quant_bf16_int8.cc exactly (same rounding + clamp conventions),
for the later DEVICE pass to check the on-chip xclbin output against via
rel-L2. Four functions, one per kernel entry point:

  quantize_bf16_to_int8(x_bf16, scale)   -> int8   (q = clamp(round(x/scale), -128, 127))
  dequantize_int8_to_bf16(q_int8, scale) -> bfloat16-rounded f32 (x = q * scale)
  cast_bf16_to_f32(x_bf16)               -> f32 (exact widen, bit-identical)
  cast_f32_to_bf16(x_f32)                -> bfloat16-rounded f32 (round-to-nearest-even)

No torch/ml_dtypes dependency: bf16 is emulated in plain numpy by truncating
an f32's mantissa to 7 bits, round-to-nearest-even at the bf16 boundary (this
matches ::aie::rounding_mode::conv_even used on-device, and the host
npu_xrt::pack_f32_to_bf16 convention referenced by the sibling cast kernels).
"""
import numpy as np


def _round_f32_to_bf16_bits(x_f32: np.ndarray) -> np.ndarray:
    """Round f32 -> bf16 (round-to-nearest-even), return as uint32 bits with
    the low 16 bits zeroed (i.e. the value the bf16 pattern represents, still
    stored in an f32 slot). This is the numeric quantity a real bf16 register
    would carry after narrowing."""
    x_f32 = np.ascontiguousarray(x_f32, dtype=np.float32)
    bits = x_f32.view(np.uint32)
    # round-to-nearest-even at the 16-bit truncation boundary
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    rounded = (bits.astype(np.uint64) + rounding_bias).astype(np.uint32)
    # handle NaN: force the canonical quiet-NaN pattern to avoid rounding
    # a NaN's mantissa into an unexpected Inf.
    is_nan = np.isnan(x_f32)
    rounded_bits = rounded & np.uint32(0xFFFF0000)
    rounded_bits = np.where(is_nan, np.uint32(0x7FC00000), rounded_bits)
    return rounded_bits


def to_bf16_rounded_f32(x_f32: np.ndarray) -> np.ndarray:
    """f32 -> bf16 -> back to f32 (the value a bf16 register holds, in an f32
    numpy array so it stays directly usable/comparable in numpy)."""
    bits = _round_f32_to_bf16_bits(x_f32)
    return bits.view(np.float32).copy()


def cast_f32_to_bf16(x_f32: np.ndarray) -> np.ndarray:
    """f32 -> bf16 narrow (round-to-nearest-even). Returned as f32 carrying
    the exact bf16-representable value (see module docstring)."""
    return to_bf16_rounded_f32(x_f32)


def cast_bf16_to_f32(x_bf16_as_f32: np.ndarray) -> np.ndarray:
    """bf16 -> f32 widen. Exact / bit-identical: bf16 is already the top 16
    bits of an f32, so this is a no-op once the input is bf16-rounded."""
    return np.ascontiguousarray(x_bf16_as_f32, dtype=np.float32).copy()


def quantize_bf16_to_int8(x_bf16_as_f32: np.ndarray, scale: float) -> np.ndarray:
    """bf16 -> int8, symmetric per-tensor scale:
    q = clamp(round(x / scale), -128, 127), round-half-to-even (matches the
    accumulator's shift-round-saturate to_vector<int8_t> on-device)."""
    x = np.ascontiguousarray(x_bf16_as_f32, dtype=np.float32)
    scaled = x.astype(np.float64) / float(scale)
    q = np.round(scaled)  # numpy round-half-to-even, same as conv_even
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)


def dequantize_int8_to_bf16(q_int8: np.ndarray, scale: float) -> np.ndarray:
    """int8 -> bf16: x = q * scale, then narrowed to bf16 (round-to-nearest-
    even) since the device kernel's accumulator emits a bf16 vector."""
    q = np.ascontiguousarray(q_int8, dtype=np.int8).astype(np.float32)
    x = q * np.float32(scale)
    return to_bf16_rounded_f32(x)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64).ravel()
    b = np.ascontiguousarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cols = 64  # multiple of N=16, matches the kernel's cols % N == 0 contract

    # --- bf16 <-> f32 round trip sanity ---
    x_f32 = rng.normal(scale=3.0, size=cols).astype(np.float32)
    x_bf16 = cast_f32_to_bf16(x_f32)
    x_back = cast_bf16_to_f32(x_bf16)
    assert np.array_equal(x_bf16, x_back), "bf16->f32 widen must be exact"
    print("cast_f32_to_bf16 / cast_bf16_to_f32 round-trip: exact match, OK")

    # --- bf16 <-> int8 quant round trip sanity ---
    scale = float(np.max(np.abs(x_bf16)) / 127.0)
    q = quantize_bf16_to_int8(x_bf16, scale)
    x_dq = dequantize_int8_to_bf16(q, scale)
    err = rel_l2(x_dq, x_bf16)
    print(f"quantize/dequantize rel-L2 vs bf16 input: {err:.4e} "
          f"(int8 quant error, expected to be well above 0 but bounded)")

    print("golden.py self-check OK")
