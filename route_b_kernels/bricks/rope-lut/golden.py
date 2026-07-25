#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Host reference (numpy) for the rope-lut brick (route_b_kernels/bricks/rope-lut).

Two references are produced:
  - `rope_reference_exact()`   : ideal float64 RoPE (np.sin/np.cos), the
                                  textbook/HF definition -- what a correctness
                                  review should compare *tolerance intent* against.
  - `rope_reference_quantized()`: bit-faithful model of what the device kernel
                                  actually computes -- same 256-entry int8-keyed
                                  sin/cos LUT (bias=128, key = round(theta*128/pi)
                                  after wrapping to [-pi,pi)) that rope_lut.cc
                                  embeds in rope_lut_tables.inc. This is the one
                                  the device pass should rel-L2 the kernel output
                                  against (it isolates "did the device execute the
                                  same LUT program" from "how coarse is a 256-bin
                                  LUT", which is a modeling choice, not a bug).

Both share the same generic parameterization as the kernel: D (head_dim),
ROT (rotary width, ROT <= D, partial rotary), base (theta base), and an
optional NTK-style linear position scale (effective_pos = pos / scale).

inv_freq is the "host-folded" tensor the kernel receives as a resident
constant buffer (see rope_lut.cc's block comment) -- it is emitted here too.
"""
import numpy as np

LUT_N = 256  # must match rope_lut_tables.inc generation (gen used by this file)


def build_inv_freq(rot: int, base: float = 10000.0, scale: float = 1.0) -> np.ndarray:
    """inv_freq[i] = base^(-2i/rot) / scale, i in [0, rot/2). Fixed given
    (rot, base, scale) -- this is exactly what golden host-side folding
    computes once at model-load time and the kernel receives as `inv_freq`."""
    assert rot % 2 == 0
    half = rot // 2
    exponents = np.arange(half, dtype=np.float64) * (2.0 / rot)
    return (base ** (-exponents)) / scale


def _bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 precision (round-to-nearest-even truncation),
    matching the mantissa truncation rope_lut.cc's LUT literals were
    generated with (see gen_lut logic mirrored below)."""
    x32 = x.astype(np.float32)
    bits = x32.view(np.uint32)
    rounding_bias = 0x7fff + ((bits >> 16) & 1)
    bits2 = (bits + rounding_bias) & 0xffff0000
    return bits2.view(np.float32)


def sincos_lut_tables() -> tuple[np.ndarray, np.ndarray]:
    """The 256-entry logical (un-bank-duplicated) sin/cos tables, bf16-rounded,
    indexed by key+128 for key in [-128,127] -- i.e. table[idx] with
    idx = key + 128. This is the *logical* content of kSinLut*/kCosLut*
    in rope_lut_tables.inc (that .inc additionally bank-duplicates each
    half for the aie::lut<4,bfloat16> hardware layout; this flat form is
    what matters for numerics)."""
    key = np.arange(LUT_N, dtype=np.float64) - 128.0
    theta = key * (np.pi / 128.0)
    sin_tab = _bf16_round(np.sin(theta))
    cos_tab = _bf16_round(np.cos(theta))
    return sin_tab, cos_tab


def _quantize_key(theta: np.ndarray) -> np.ndarray:
    """theta -> int8 key in [-128,127], matching rope_lut.cc's wrap+quantize
    (manual round-half-away-from-zero, no libm roundf)."""
    k = np.floor(theta / (2.0 * np.pi) + 0.5).astype(np.int64)
    # note: np.floor(x+0.5) matches away-from-zero rounding used on-device
    # for both positive and negative x (kernel's `(k_wrap>=0)?+0.5:-0.5` trick)
    wrapped = theta - k * (2.0 * np.pi)
    q = wrapped * (128.0 / np.pi)
    key = np.floor(q + np.where(q >= 0, 0.5, -0.5)).astype(np.int64)
    return np.clip(key, -128, 127)


def rope_reference_quantized(qk: np.ndarray, pos: np.ndarray, rot: int,
                              base: float = 10000.0, scale: float = 1.0) -> np.ndarray:
    """qk: [M, D] float32 (bf16-rounded by caller if comparing against a bf16
    device buffer); pos: [M] int; returns rotated [M, D], using the SAME
    256-entry gather-LUT the kernel uses (bit-faithful device model)."""
    m, d = qk.shape
    assert rot <= d and rot % 2 == 0
    half = rot // 2
    inv_freq = build_inv_freq(rot, base, scale)
    sin_tab, cos_tab = sincos_lut_tables()

    out = qk.astype(np.float64).copy()
    for row in range(m):
        theta = pos[row] * inv_freq  # [half]
        key = _quantize_key(theta)
        idx = key + 128
        s = sin_tab[idx]
        c = cos_tab[idx]
        x1 = qk[row, 0:half].astype(np.float64)
        x2 = qk[row, half:rot].astype(np.float64)
        out[row, 0:half] = x1 * c - x2 * s
        out[row, half:rot] = x1 * s + x2 * c
        # out[row, rot:] left unchanged (partial rotary pass-through)
    return out


def rope_reference_exact(qk: np.ndarray, pos: np.ndarray, rot: int,
                          base: float = 10000.0, scale: float = 1.0) -> np.ndarray:
    """Ideal RoPE, full-precision np.sin/np.cos -- the textbook definition,
    for judging how much error the 256-bin LUT quantization itself costs
    (independent of any device bug)."""
    m, d = qk.shape
    half = rot // 2
    inv_freq = build_inv_freq(rot, base, scale)
    out = qk.astype(np.float64).copy()
    for row in range(m):
        theta = pos[row] * inv_freq
        s, c = np.sin(theta), np.cos(theta)
        x1 = qk[row, 0:half].astype(np.float64)
        x2 = qk[row, half:rot].astype(np.float64)
        out[row, 0:half] = x1 * c - x2 * s
        out[row, half:rot] = x1 * s + x2 * c
    return out


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    denom = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom > 0 else float(np.linalg.norm(a - b))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D, ROT, M = 128, 128, 64
    qk = rng.standard_normal((M, D)).astype(np.float32)
    pos = np.arange(M, dtype=np.int64)  # decode-style monotonically increasing KV offset

    quant = rope_reference_quantized(qk, pos, ROT)
    exact = rope_reference_exact(qk, pos, ROT)
    print("LUT-quantization-only rel-L2 (quantized vs exact, no device involved):",
          rel_l2(quant, exact))

    # partial-rotary sanity check
    ROT2 = 64
    quant2 = rope_reference_quantized(qk, pos, ROT2)
    assert np.allclose(quant2[:, ROT2:], qk[:, ROT2:]), "pass-through dims must be untouched"
    print("partial-rotary pass-through check: OK")
