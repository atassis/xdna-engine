#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Task 1 (kv_skip_v phase 1): can an on-chip fp32 sin/cos polynomial reproduce RoPE's cos/sin
table accurately enough to derive V from K by inverting K's RoPE rotation, for every past position
0..262144? Pure numpy exploration -- no kernel code. Ground truth is scripts/gate_llm_reference.py's
`rope()`, computed in float64.

Finding: the polynomial itself is not the error source (degree-9 Taylor on |y|<=pi/4 is ~1e-9,
negligible). Two operand-precision effects dominate, and BOTH must be fixed together:

1. inv_freq stored as a single fp32 has a ~2^-24 relative error; multiplied by a position up to
   2^18, that is an absolute angle error up to ~0.03 rad, independent of the range-reduction
   scheme -- confirmed: even the best single-fp32-inv_freq scheme here (Dekker two-product +
   3-limb Cody-Waite) still fails at 6.6e-3..7.6e-3 vs the gate. bf16 inv_freq is far worse (a
   relative error on the UNREDUCED angle, which runs up to 2.6e5): max error saturates at ~2.0,
   i.e. essentially random phase.
2. FIX: store inv_freq resident as a double-single fp32 PAIR (hi, lo -- ~48 bits combined, no
   fp64 hardware needed, 2x the storage of one fp32 per component) and form the angle via a
   Dekker two-product against each limb, combined with >=2-limb Cody-Waite double-single
   reduction mod 2*pi. That passes at 3.6e-7..4.8e-7 -- 4 orders of magnitude under the gate,
   effectively float32-epsilon-limited. Reduction alone (double-single inv, single-limb
   reduction) still fails at 6.7e-3..7.3e-3: BOTH the operand split and the >=2-limb reduction
   are required, neither alone is sufficient.

CHOSEN SCHEME: int_phase. Resident is a single uint32 per component -- F = round(inv_freq /
(2*pi) * 2**32), a 0.32 fixed-point turn fraction -- instead of the fp32 pair above. Phase for
position p is ph = (p * F) mod 2**32, an exact uint32 multiply: the wraparound IS the fractional
turn, no reduction step, no accumulated rounding across positions. Error is exactly the one
rounding of F: |dF| <= 2**-33 turns, so at p_max = 2**18 the phase error is bounded by
p_max * 2**-33 turns * 2*pi rad/turn ~= 1.92e-4 rad -- matches the measured 1.908e-4 max|dcos| /
1.907e-4 max|dsin| below. Passes the gate by ~30x with half the resident storage of the
double-single scheme and no multi-limb reduction arithmetic -- the right tradeoff for an on-chip
kernel with no fp64.

The chosen scheme itself (inv_freq_to_turns_u32, phase_cs) lives in scripts/rope_int_phase.py, not
here, so the production oracle (gate_llm_reference.py) never has to import this exploration
script -- this module imports it back for scheme_int_phase below and the losing-scheme comparison.

Run: .venv-iron/bin/python scripts/measure_rope_poly.py
"""
import numpy as np
import ml_dtypes

from rope_int_phase import (  # noqa: E402 -- chosen scheme lives there; see its module docstring
    TWO_PI, U32_MASK, poly_sincos, inv_freq_to_turns_u32, phase_cs,
)

BF16 = ml_dtypes.bfloat16
PI_OVER_2 = np.pi / 2.0

# bf16 gate: this repo's own convention (scripts/gate_numeric.py) is one bf16 quantum, 2**-8, at
# the tensor's magnitude -- cos/sin sit at RMS ~0.7, so the floor is ~2**-8*0.7 ~= 2.7e-3. The
# plan's independent derivation (8 mantissa bits -> unit roundoff 2**-9 ~= 3.9e-3 relative at
# magnitude ~1) gives the same order. Gate = 3x that unit roundoff, at magnitude 1 (cos/sin peak).
BF16_ULP_AT_1 = 2.0 ** -9
GATE_ABS = 3.0 * BF16_ULP_AT_1  # 5.859e-3


def rope_inv_freq(hd, theta, partial):
    """Verbatim formula from scripts/gate_llm_reference.py:rope() (float64)."""
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64)[: hd // 2] / hd))
    if partial is not None:
        inv[int(partial * hd // 2):] = 0.0
    return inv


def to_bf16_f32(x):
    return np.asarray(x, np.float32).astype(BF16).astype(np.float32)


def veltkamp_split_f32(a):
    """Split fp32 a into hi+lo, each with <=12 significant bits, hi+lo == a exactly."""
    C = np.float32(4097.0)  # 2**12 + 1, splits a 24-bit mantissa in half
    big = np.float32(a) * C
    hi = (big - (big - a)).astype(np.float32)
    lo = (a - hi).astype(np.float32)
    return hi, lo


def twoprod_f32(a, b):
    """Dekker's error-free fp32 product: hi+lo == a*b exactly (no FMA assumed)."""
    hi = (np.float32(a) * np.float32(b)).astype(np.float32)
    a_hi, a_lo = veltkamp_split_f32(a)
    b_hi, b_lo = veltkamp_split_f32(b)
    err = (((a_hi * b_hi - hi) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo).astype(np.float32)
    return hi, err


# Cody-Waite limbs of 2*pi, each an exact fp32 value, decreasing magnitude, so k*limb rounds with
# a smaller absolute error than one fp32 multiply by the full constant would.
_TWO_PI_F64 = np.float64(TWO_PI)
CW_LIMBS = []
_rem = _TWO_PI_F64
for _ in range(4):
    lim = np.float32(_rem)
    CW_LIMBS.append(lim)
    _rem -= np.float64(lim)
CW_LIMBS = np.array(CW_LIMBS, dtype=np.float32)


def quadrant_select(k, sin_y, cos_y):
    q = (np.asarray(k, np.int64) % 4 + 4) % 4
    sin_r = np.select([q == 0, q == 1, q == 2, q == 3], [sin_y, cos_y, -sin_y, -cos_y])
    cos_r = np.select([q == 0, q == 1, q == 2, q == 3], [cos_y, -sin_y, -cos_y, sin_y])
    return sin_r.astype(np.float32), cos_r.astype(np.float32)


def scheme_naive_fp32(pos_f32, inv_f32):
    """Plain fp32 product, single-fp32-constant pi/2 reduction. What a naive port would write."""
    angle = (pos_f32 * inv_f32).astype(np.float32)
    k = np.round(angle / np.float32(PI_OVER_2)).astype(np.float32)
    r = (angle - k * np.float32(PI_OVER_2)).astype(np.float32)
    sin_y, cos_y = poly_sincos(r)
    return quadrant_select(k, sin_y, cos_y)


def scheme_codywaite_fp32(pos_f32, inv_f32, nlimbs=2):
    """Plain fp32 product (single rounding), multi-limb Cody-Waite pi/2 reduction."""
    angle = (pos_f32 * inv_f32).astype(np.float32)
    k = np.round(angle / np.float32(PI_OVER_2)).astype(np.float32)
    limbs = []
    _rem = np.float64(PI_OVER_2)
    for _ in range(nlimbs):
        lim = np.float32(_rem)
        limbs.append(lim)
        _rem -= np.float64(lim)
    r = angle
    for lim in limbs:
        r = (r - k * lim).astype(np.float32)
    sin_y, cos_y = poly_sincos(r)
    return quadrant_select(k, sin_y, cos_y)


def two_sum(a, b):
    """Knuth's exact fp32 sum: s+e == a+b exactly, no ordering assumption on |a|,|b|."""
    s = (a + b).astype(np.float32)
    bb = (s - a).astype(np.float32)
    e = ((a - (s - bb)) + (b - bb)).astype(np.float32)
    return s, e


def dd_add(hi1, lo1, hi2, lo2):
    """Double-single (fp32 pair) add, renormalized: (hi,lo) with hi+lo == hi1+lo1+hi2+lo2 to
    within one fp32 ulp of the sum's magnitude."""
    s, e = two_sum(hi1, hi2)
    e = (e + lo1 + lo2).astype(np.float32)
    s2, e2 = two_sum(s, e)
    return s2, e2


def scheme_twoprod_codywaite(pos_f32, inv_f32, nlimbs=3):
    """Dekker two-product for pos*inv_freq (hi+lo, fp32 only, EXACT: hi+lo == pos*inv_freq with
    no rounding beyond the two fp32 outputs), then double-single reduction against `nlimbs`
    Cody-Waite limbs of 2*pi -- the argument-reduction scheme this task's gate needs (see module
    docstring). k is picked from `hi` alone in plain fp32: hi already carries the angle's full
    magnitude, so a fractional error there of ~1e-7 relative is far short of the 0.5 needed to
    round to the wrong integer multiple of 2*pi."""
    hi, lo = twoprod_f32(pos_f32, inv_f32)
    k32 = np.round(hi / np.float32(TWO_PI)).astype(np.float32)
    r_hi, r_lo = hi, lo
    for lim in CW_LIMBS[:nlimbs]:
        p_hi, p_lo = twoprod_f32(k32, lim)
        r_hi, r_lo = dd_add(r_hi, r_lo, -p_hi, -p_lo)
    r = (r_hi + r_lo).astype(np.float32)
    # r is in [-pi, pi); one more fold to [-pi/4, pi/4] -- small magnitude now, a single fp32
    # constant costs negligible precision here versus doing it at the original 2.6e5 scale.
    k2 = np.round(r / np.float32(PI_OVER_2)).astype(np.float32)
    r2 = (r - k2 * np.float32(PI_OVER_2)).astype(np.float32)
    sin_y, cos_y = poly_sincos(r2)
    return quadrant_select(k2, sin_y, cos_y)  # k32's 2*pi cycles contribute 0 mod 4


def split_f64_to_f32_pair(x64):
    """Resident double-single storage for inv_freq: hi = fp32(x), lo = fp32(x - f64(hi)).
    hi+lo recovers ~48 bits of x64's mantissa versus a single fp32's 24 -- the fix for the
    dominant error found below (a single fp32 inv_freq's ~2^-24 relative error times a position up
    to 2^18 is an absolute angle error of ~2^-6 rad, independent of range-reduction scheme)."""
    hi = x64.astype(np.float32)
    lo = (x64 - hi.astype(np.float64)).astype(np.float32)
    return hi, lo


def scheme_twoprod_codywaite_ds_inv(pos_f32, inv_hi, inv_lo, nlimbs=3):
    """Like scheme_twoprod_codywaite, but the operand itself is double-single: angle (exact
    double-single) = pos*inv_hi (twoprod) dd_add pos*inv_lo (twoprod), before the same Cody-Waite
    reduction. Needs inv_freq resident as an (hi, lo) fp32 pair, not one fp32 value."""
    h1, l1 = twoprod_f32(pos_f32, inv_hi)
    h2, l2 = twoprod_f32(pos_f32, inv_lo)
    r_hi, r_lo = dd_add(h1, l1, h2, l2)
    k32 = np.round(r_hi / np.float32(TWO_PI)).astype(np.float32)
    for lim in CW_LIMBS[:nlimbs]:
        p_hi, p_lo = twoprod_f32(k32, lim)
        r_hi, r_lo = dd_add(r_hi, r_lo, -p_hi, -p_lo)
    r = (r_hi + r_lo).astype(np.float32)
    k2 = np.round(r / np.float32(PI_OVER_2)).astype(np.float32)
    r2 = (r - k2 * np.float32(PI_OVER_2)).astype(np.float32)
    sin_y, cos_y = poly_sincos(r2)
    return quadrant_select(k2, sin_y, cos_y)


# int_phase (the chosen scheme, see rope_int_phase.py) isn't in SCHEMES: every entry there takes
# (pos_f32, inv_f32) so measure() can sweep the same fp32/bf16 inv_freq-precision axis over all of
# them, but int_phase's resident constant is a uint32 turn fraction, not an fp32 inv_freq -- a
# different signature -- so measure() calls it separately, once per (geometry, position grid).
SCHEMES = {
    "naive_fp32": scheme_naive_fp32,
    "codywaite2_fp32": lambda p, i: scheme_codywaite_fp32(p, i, nlimbs=2),
    "codywaite3_fp32": lambda p, i: scheme_codywaite_fp32(p, i, nlimbs=3),
    "twoprod_codywaite3": lambda p, i: scheme_twoprod_codywaite(p, i, nlimbs=3),
}


def scheme_int_phase(pos_u32, F_u32):
    """pos and F both uint32 (or safely-widenable); ph = p*F mod 2**32 is an exact fractional-turn
    product -- no reduction step, no rounding beyond the resident F itself (see phase_cs)."""
    ph = (pos_u32.astype(np.uint64) * F_u32.astype(np.uint64)) & U32_MASK
    return phase_cs(ph)


def position_samples(max_pos=262144):
    dense = np.arange(0, 2049, dtype=np.float64)
    log_sparse = np.unique(np.round(np.geomspace(2049, max_pos, 400)).astype(np.int64))
    pos = np.unique(np.concatenate([dense, log_sparse.astype(np.float64), [float(max_pos)]]))
    return pos


def measure(name, hd, theta, partial, max_pos=262144):
    inv64 = rope_inv_freq(hd, theta, partial)
    nonzero = inv64 != 0.0
    inv64_nz = inv64[nonzero]
    pos = position_samples(max_pos)

    P, F = np.meshgrid(pos, inv64_nz, indexing="ij")
    angle64 = P * F
    ref_cos = np.cos(angle64).astype(np.float32)
    ref_sin = np.sin(angle64).astype(np.float32)

    pos_f32 = P.astype(np.float32)
    results = {}
    for inv_prec, inv_f32 in (("fp32", F.astype(np.float32)), ("bf16", to_bf16_f32(F))):
        for scheme_name, fn in SCHEMES.items():
            sin_r, cos_r = fn(pos_f32, inv_f32)
            abs_c = np.abs(cos_r.astype(np.float64) - ref_cos.astype(np.float64))
            abs_s = np.abs(sin_r.astype(np.float64) - ref_sin.astype(np.float64))
            key = f"{scheme_name}/inv={inv_prec}"
            results[key] = (float(abs_c.max()), float(abs_s.max()))

    inv_hi, inv_lo = split_f64_to_f32_pair(F)
    for nlimbs in (1, 2, 3):
        sin_r, cos_r = scheme_twoprod_codywaite_ds_inv(pos_f32, inv_hi, inv_lo, nlimbs=nlimbs)
        abs_c = np.abs(cos_r.astype(np.float64) - ref_cos.astype(np.float64))
        abs_s = np.abs(sin_r.astype(np.float64) - ref_sin.astype(np.float64))
        key = f"twoprod_codywaite{nlimbs}_ds_inv/inv=fp32x2"
        results[key] = (float(abs_c.max()), float(abs_s.max()))

    F_u32 = inv_freq_to_turns_u32(F)
    sin_r, cos_r = scheme_int_phase(P.astype(np.uint64), F_u32)
    abs_c = np.abs(cos_r.astype(np.float64) - ref_cos.astype(np.float64))
    abs_s = np.abs(sin_r.astype(np.float64) - ref_sin.astype(np.float64))
    results["int_phase (chosen)/inv=u32turns"] = (float(abs_c.max()), float(abs_s.max()))
    return results, inv64_nz.size


def main():
    print(f"gate: abs error <= {GATE_ABS:.4e} (3x bf16 unit roundoff at magnitude 1, "
          f"cf. gate_numeric.py's 2**-8*rms(ref) floor convention)\n")

    geoms = [
        ("global (hd=512, theta=1e6, partial=0.25)", 512, 1_000_000.0, 0.25),
        ("local  (hd=256, theta=1e4, partial=None)", 256, 10_000.0, None),
    ]
    any_pass = {}
    for label, hd, theta, partial in geoms:
        results, n_active = measure(label, hd, theta, partial)
        print(f"=== {label} -- {n_active} active inv_freq components, positions 0..262144 ===")
        for key, (max_c, max_s) in sorted(results.items(), key=lambda kv: max(kv[1])):
            worst = max(max_c, max_s)
            verdict = "PASS" if worst <= GATE_ABS else "FAIL"
            any_pass[(label, key)] = verdict == "PASS"
            print(f"  {key:32s} max|d cos|={max_c:.4e}  max|d sin|={max_s:.4e}  [{verdict}]")
        print()

    best_fp32 = [k for (lab, k), ok in any_pass.items() if ok and "inv=fp32" in k]
    best_bf16 = [k for (lab, k), ok in any_pass.items() if ok and "inv=bf16" in k]
    print(f"schemes passing with fp32 inv_freq: {sorted(set(best_fp32))}")
    print(f"schemes passing with bf16 inv_freq: {sorted(set(best_bf16))}")


if __name__ == "__main__":
    main()
