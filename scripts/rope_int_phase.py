#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The CHOSEN on-chip RoPE scheme for kv_skip_v's inverse rotation -- single resident uint32 per
component (a 0.32 fixed-point turn fraction), exact integer phase accumulation, no reduction step.
See scripts/measure_rope_poly.py for the exploration that picked this over the fp32/double-single
schemes it keeps as the losing-scheme record.

Kept separate from that exploration script so the production oracle (gate_llm_reference.py) never
imports an exploration script -- this is the only module either of them, or a kernel test, needs.
"""
import numpy as np

TWO_PI = 2.0 * np.pi
U32_MASK = np.uint64(0xFFFFFFFF)

# A uint32 turn's top 2 bits select the quadrant (4 quadrants of width 2**30 each); HALF_QUADRANT
# is added before the shift so the split rounds to the NEAREST quadrant boundary instead of
# truncating towards zero -- the standard round-half-up-via-bias trick, at quadrant granularity.
QUADRANT_BITS = 30
QUADRANT_WIDTH = 1 << QUADRANT_BITS  # 2**30
HALF_QUADRANT = QUADRANT_WIDTH >> 1  # 2**29


def poly_sincos(y):
    """Degree-9/8 Taylor for sin/cos on |y|<=pi/4 -- ~1e-9 there, negligible next to int_phase's
    own ~1.9e-4 bound (the F quantization -- see phase_cs)."""
    y2 = y * y
    sin_y = y * (1.0 + y2 * (-1.0 / 6 + y2 * (1.0 / 120 + y2 * (-1.0 / 5040))))
    cos_y = 1.0 + y2 * (-1.0 / 2 + y2 * (1.0 / 24 + y2 * (-1.0 / 720 + y2 * (1.0 / 40320))))
    return sin_y.astype(np.float32), cos_y.astype(np.float32)


def inv_freq_to_turns_u32(inv64):
    """Resident constant: inv_freq (rad/position, float64) -> uint32 0.32 fixed-point turn
    fraction. inv_freq <= 1 (theta >= 1, exponent in [0,1)) so the turn fraction is < 1 and fits
    without an integer part."""
    return (np.round(np.asarray(inv64, np.float64) / TWO_PI * (2.0 ** 32)).astype(np.uint64)
            & U32_MASK)


def phase_cs(ph_u32):
    """ph (uint32 turns*2**32) -> (sin, cos), float32. Quadrant + residual are both exact integer
    ops (no reduction rounding); only the final residual->radian scale and the poly are float32."""
    t = (ph_u32.astype(np.uint64) + np.uint64(HALF_QUADRANT)) & U32_MASK
    q = (t >> np.uint64(QUADRANT_BITS)).astype(np.int64)
    r = (t & np.uint64(QUADRANT_WIDTH - 1)).astype(np.int64) - HALF_QUADRANT
    y = r.astype(np.float32) * np.float32(np.pi / 2 / QUADRANT_WIDTH)
    sin_y, cos_y = poly_sincos(y)
    sin_r = np.select([q == 0, q == 1, q == 2, q == 3], [sin_y, cos_y, -sin_y, -cos_y])
    cos_r = np.select([q == 0, q == 1, q == 2, q == 3], [cos_y, -sin_y, -cos_y, sin_y])
    return sin_r.astype(np.float32), cos_r.astype(np.float32)
