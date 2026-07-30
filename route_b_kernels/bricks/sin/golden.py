#!/usr/bin/env python3
"""Host numpy golden for route_b_kernels/bricks/sin/sin.cc.

The kernel computes sin(x) for unbounded f32 x as: range-reduce to r = x - 2*pi*round(x/(2*pi))
so r in [-pi, pi], then evaluate via aie::linear_approx over a bfloat16 LUT of one period.
Primitive choice is measured, not assumed: linear_approx beats parallel_lookup on sin by 3.5x at
512 B and 65x at 8 KB at matched memory (kb/lut-primitive-choice-lookup-vs-linear-approx).

`sin_lut_emulated` below is a bit-faithful transcription of what the DEVICE does, including the
bfloat16 narrowing of the reduced argument and the piecewise-linear interpolation. It is the
prediction the device gate is checked against; `np.sin` is the ground truth both are measured on.

Usage: python3 golden.py    # prints max abs err of the emulation vs np.sin, and emits the .inc
"""
import numpy as np

TWO_PI = 2.0 * np.pi
# 256 entries * 24 B/entry (4 B float offset + 2 B bf16 slope, x4 copies) = 6 KB, inside the 8 KB
# budget the KB note measured. Entry count is bounded ABOVE by bf16 input resolution, not by memory:
# linear_approx indexes off the bf16 input, and bf16 has an 8-bit mantissa, so at |u| ~ 128 the input
# spacing is 0.5 entries. Pushing to 512 entries would quantize the input to ~1 entry and throw away
# the interpolation the primitive exists for.
LUT_ENTRIES = 256
LUT_LO = -np.pi
LUT_HI = np.pi
STEP = (LUT_HI - LUT_LO) / LUT_ENTRIES
# The kernel pre-scales the reduced argument into ENTRY UNITS and calls compute() with step_bits=0:
# hardware computes index = int(u) and returns offset[e] + slope[e]*u against the FULL u, so the
# table stores y-INTERCEPTS in u coordinates, not within-cell values.
SCALE = 1.0 / STEP                     # r -> u,  u in [-128, 128)
BIAS = LUT_ENTRIES // 2                # entry 0 <-> u = -128


def bf16(x):
    """Round f32 -> bfloat16 -> f32, round-to-nearest-even (what the device does on narrowing)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint32)
    lsb = (u >> 16) & 1
    rounded = ((u + 0x7FFF + lsb) & 0xFFFF0000).astype(np.uint32)
    return rounded.view(np.float32)


def range_reduce(x):
    """r = x - 2*pi*round(x / 2*pi), giving r in [-pi, pi]."""
    x = np.asarray(x, dtype=np.float32)
    k = np.rint(x / TWO_PI)
    return (x - TWO_PI * k).astype(np.float32)


def sin_ref(x):
    """Ground truth."""
    return np.sin(np.asarray(x, dtype=np.float64)).astype(np.float32)


def lut_tables():
    """(offset, slope) per entry in U COORDINATES, matching what the hardware evaluates.

    aie2 linear_approx computes  offset[e] + slope[e] * u  where u is the (delayed) FULL input,
    so offset must be the line's y-INTERCEPT, not the cell's left-edge value.

    The slope is stored as bfloat16, and the intercept is then computed FROM THE ROUNDED SLOPE.
    That matters more than it looks: the intercept is O(1) while slope*u is O(pi), so pairing an
    exact intercept with a rounded slope leaves an uncancelled error of eps_bf16 * |slope*u| ~ 1e-2.
    Deriving the intercept from the rounded slope makes the residual O(slope * cell_width) instead.
    """
    u_edges = np.arange(LUT_ENTRIES + 1, dtype=np.float64) - BIAS      # left edges in u units
    r_edges = u_edges / SCALE
    vals = np.sin(r_edges)
    slope = ((vals[1:] - vals[:-1]) / 1.0).astype(np.float32)          # per UNIT of u
    slope_b = bf16(slope)                                              # what the table really holds
    offset = (vals[:-1] - slope_b.astype(np.float64) * u_edges[:-1]).astype(np.float32)
    return offset, slope_b


def sin_lut_emulated(x):
    """Bit-faithful emulation of the device path: reduce, scale to u, narrow to bf16, interpolate."""
    r = range_reduce(x)
    u = bf16(np.asarray(r, dtype=np.float32) * np.float32(SCALE))      # compute() takes bf16 input
    offset, slope_b = lut_tables()
    idx = np.clip((np.floor(u).astype(np.int32) + BIAS), 0, LUT_ENTRIES - 1)
    # offset is float in the table; slope is bf16; the mac accumulates in f32
    return (offset[idx] + slope_b[idx].astype(np.float32) * u).astype(np.float32)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


def emit_inc(path):
    """Emit the aie::lut<4,float,bfloat16> ab/cd tables.

    !! THE ab/cd PACKING BELOW IS UNVERIFIED. !!
    The values (intercept + bf16 slope, above) are correct and self-checked; how they must be
    INTERLEAVED across the ab/cd banks is not. aie2's linear_approx reads them with
    `load_lut_2x_float(LUT_ab_, LUT_cd_, index, coeff0, coeff1)` and then splits the pair via
    `shuffle(coeff0, coeff1, T32_16x2_hi)` for the f32 offset and `shuffle(..., T16_16x4_lo)` for the
    bf16 slope (aie_api/detail/aie2/linear_approx.hpp:346-420). That packing is expressed only
    through those shuffle patterns, and there is NO working linear_approx call site anywhere in this
    repo to copy -- bricks/rope-lut uses parallel_lookup, a different primitive, and
    _verify/probe_lut_ab_cd.py probes parallel_lookup's banks, not these.

    So the even/odd split written here is a GUESS and must be established empirically before the
    kernel can be trusted. See the probe task in the Lane C stage-1 plan; model it on
    probe_lut_ab_cd.py (discriminating tables + known keys, so each lane reveals which bank and
    which entry it actually read).
    """
    offset, slope = lut_tables()
    def fmt(vals):
        return ",\n  ".join(", ".join(f"{v:.8e}f" for v in vals[i:i + 8]) for i in range(0, len(vals), 8))
    inter_ab, inter_cd = [], []
    for i in range(LUT_ENTRIES):
        (inter_ab if i % 2 == 0 else inter_cd).extend([offset[i], slope[i]])
    body = f"""// AUTO-GENERATED by bricks/sin/golden.py -- do not hand-edit.
// aie::lut<4,float,bfloat16> tables for sin over [{LUT_LO:.8f}, {LUT_HI:.8f}), {LUT_ENTRIES} entries.
#pragma once
static const float kSinApproxAb[] = {{
  {fmt(inter_ab * 4)}
}};
static const float kSinApproxCd[] = {{
  {fmt(inter_cd * 4)}
}};
static constexpr int kSinLutEntries = {LUT_ENTRIES};
static constexpr float kSinLutLo = {LUT_LO:.8f}f;
static constexpr float kSinLutStep = {STEP:.10f}f;
"""
    path.write_text(body)


if __name__ == "__main__":
    from pathlib import Path
    rng = np.random.default_rng(0)
    # Snake's argument is alpha*x: alpha is a learned per-channel scale, x a post-conv activation.
    # Cover +/-64 (about 10 periods) so range reduction is actually exercised.
    x = (rng.random(1 << 16, dtype=np.float32) * 128.0 - 64.0).astype(np.float32)
    got = sin_lut_emulated(x)
    ref = sin_ref(x)
    print("max abs err:", float(np.max(np.abs(got - ref))))
    print("rel-L2     :", rel_l2(got, ref))
    assert rel_l2(got, ref) <= 3e-2, "emulated LUT path misses the brick gate"
    emit_inc(Path(__file__).parent / "sin_lut_tables.inc")
    print("wrote sin_lut_tables.inc")
