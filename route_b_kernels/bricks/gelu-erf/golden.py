#!/usr/bin/env python3
"""Host numpy golden for route_b_kernels/bricks/gelu-erf/gelu_erf.cc.

Target curve, matching ggml_vec_gelu_erf_f32 / scripts/codec_quantizer_ref.py::gelu_erf:
    y = 0.5 * x * (1 + erf(x / sqrt(2)))

Differs from route_b_kernels/bricks/geglu/geglu.cc on two independent counts: geglu.cc is (a) the
TANH approximation (torch gelu(approximate="tanh")), not erf -- a different curve, not just a
different implementation of the same one -- and (b) a GATED op (GeGLU: gelu(gate)*up, two [up|gate]
operands, one output). This brick is the plain ungated GELU: one input row, one output row.
Consumer: scripts/codec_quantizer_ref.py::convnext_block (~line 292), applied elementwise to the
[4096, T] pwconv1 output.

`gelu_erf_ref` reproduces codec_quantizer_ref.gelu_erf/_erf bit-for-bit (same A&S 7.1.26 erf,
same coefficients, max abs error 1.5e-7 vs true erf) -- this IS the ground truth the device gate
checks against, because it is the oracle the rest of this repo's quantizer/WER pipeline already
uses. Reproduced locally rather than imported, matching every other brick's golden.py in this tree.

`gelu_erf_hiprec` cross-checks that oracle against scipy.special.erf (float64) purely to confirm
the A&S approximation error is where it claims to be -- see __main__.

RELIABILITY WARNING -- THREE STRIKES, READ BEFORE TRUSTING gelu_erf_emulated (2026-07-30/31).
v1's emulation of an A&S-erf kernel (aie::inv reciprocal + aie::exp2<bfloat16>) predicted rel-L2
1.171e-05; device measured 0.70. v2 added a device-evidence clamp before the exp2 argument plus an
unconditional final invariant clamp; predicted numbers barely moved, device measured 0.7043 --
ESSENTIALLY UNCHANGED, and the final clamp made the failure UNDIAGNOSABLE by mapping a wrong raw
value at the diagnostic point (x=12) to exactly 0.0. Root-caused via a device probe (oneshot vs
rowwise rails, same kernel: both red, row0 identical -> ruled out a rail/codegen defect, proved the
arithmetic wrong broadly): aie::exp2<bfloat16> was confirmed producing wrong output for this
kernel's operand distribution. v3 removed exp2 AND aie::inv entirely (see below) -- landmarks were
then all correct on device, INCLUDING the predicted constant tail offset, so the maths were right --
but a device probe isolating abs/min/mul/add/`0.5*(x+abs(x))` found all four bit-exact while
NON-landmark points still failed (one read exactly 0.0, later chunks blew up to -1.85e5). That is
textbook spill/alias codegen (sin.cc's own header describes the identical symptom -- "zeros
interleaved" -- from an earlier session; sin remains red today for the same reason). The formula
was never wrong in v3; the compiled Horner chain (10 coefficients, 5 live vectors) was too long.

v4 (current) is v3's IDENTICAL formula with two purely mechanical changes: the D(ax) fit degree
cut from 9 to 5 (accuracy margin was ~3000x more than the 3e-2 gate needs), and `s=0.5*(x+ax)`
computed BEFORE the Horner loop so `x`/`ax` are dead during it -- down to 3 live vectors (axc, p,
s) across the polynomial, from v3's 5. gelu_erf_emulated below is v4's formula, bit-for-bit,
reordered to match. It uses zero primitives shown to mismatch device (only aie::abs/aie::min/
aie::mul/aie::add/aie::sub, and specifically abs/min/`0.5*(x+abs(x))` were independently device-
confirmed bit-exact) -- but per the three strikes above, a clean host number has NEVER been
sufficient evidence on this brick. It says nothing about spill/alias codegen, which is what broke
v3 despite a numerically-fine host prediction. Report gelu_erf_emulated's numbers as fit quality
only (e.g. "expected raw value at x=12"), never as evidence the brick will pass.

Usage: python3 golden.py
"""
import numpy as np

# D(ax) = ax - GELU_ref(ax), degree-5 Horner fit over ax in [0, 5], highest degree first. Must
# match gelu_erf.cc's literals exactly -- this IS the kernel's formula, not an independent model.
# Cut from v3's degree 9 (see module docstring): degree 4 measured rel-L2 2.389e-03 / max-abs
# 3.064e-02 (too close to the 3e-2 gate itself); degree 5 measured rel-L2 5.422e-04 / max-abs
# 6.701e-03 (~55x / ~4.5x margin) and was kept.
_D_COEFFS = [
    0.002835879616732109,     # x^5
    -0.043260768673460806,    # x^4
    0.24472919119017997,      # x^3
    -0.6094418780222849,      # x^2
    0.5663938552943448,       # x^1
    -0.004514140008753372,    # x^0
]
_D_CLAMP = 5.0


def _erf_as_and_s(x):
    """Abramowitz & Stegun 7.1.26, same formula/coefficients as
    scripts/codec_quantizer_ref.py::_erf (~line 249). Max abs error 1.5e-7."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592) * t
    return sign * (1.0 - poly * np.exp(-ax * ax))


def gelu_erf_ref(x):
    """Ground truth: bit-structurally identical to codec_quantizer_ref.gelu_erf (A&S 7.1.26 erf).
    This is what verify_gelu_erf.py gates the device kernel against, and what gelu_erf.cc's D(ax)
    polynomial was fit against."""
    xf = np.asarray(x, dtype=np.float64)
    return (0.5 * xf * (1.0 + _erf_as_and_s(xf / np.sqrt(2.0)))).astype(np.float32)


def _erf_hiprec(x):
    """True erf, float64, via scipy.special.erf (vectorized)."""
    from scipy.special import erf as _scipy_erf
    return _scipy_erf(np.asarray(x, dtype=np.float64))


def gelu_erf_hiprec(x):
    """High-precision reference: exact erf (scipy), not the A&S approximation."""
    xf = np.asarray(x, dtype=np.float64)
    return 0.5 * xf * (1.0 + _erf_hiprec(xf / np.sqrt(2.0)))


def gelu_erf_emulated(x):
    """Predicts the device: gelu_erf.cc's v4 formula, bit-for-bit, f32 throughout, reordered so
    `s=0.5*(x+ax)` is computed before the degree-5 Horner loop in axc (matching the kernel's
    register-pressure fix -- see module docstring). See the RELIABILITY WARNING above before
    trusting this against the real device: it is a formula/fit-quality sanity check, not a
    validated numeric predictor; a clean host number has been insufficient on this brick before."""
    x = np.asarray(x, dtype=np.float32)
    ax = np.abs(x).astype(np.float32)
    axc = np.minimum(ax, np.float32(_D_CLAMP)).astype(np.float32)
    s = (x + ax).astype(np.float32)
    s = (s * np.float32(0.5)).astype(np.float32)  # x, ax no longer referenced past this point
    coeffs = [np.float32(c) for c in _D_COEFFS]
    p = np.full_like(axc, coeffs[0])
    for c in coeffs[1:]:
        p = (p * axc).astype(np.float32)
        p = (p + c).astype(np.float32)
    out = (s - p).astype(np.float32)  # NO final clamp -- see module docstring
    return out


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.concatenate([
        np.linspace(-30.0, 30.0, 200001, dtype=np.float32),
        (rng.standard_normal(200000).astype(np.float32) * 3.0),
    ])

    ref = gelu_erf_ref(x)
    hip = gelu_erf_hiprec(x)
    emu = gelu_erf_emulated(x)

    print(f"shape {x.shape}")
    print(f"ref (A&S, oracle)  vs hiprec (true erf): rel-L2 {rel_l2(ref, hip):.3e}  "
          f"max-abs {float(np.max(np.abs(ref.astype(np.float64) - hip))):.3e}")
    print(f"emulated (v4 poly) vs ref (A&S, oracle): rel-L2 {rel_l2(emu, ref):.3e}  "
          f"max-abs {float(np.max(np.abs(emu.astype(np.float64) - ref.astype(np.float64)))):.3e}  "
          f"[FIT QUALITY ONLY -- see RELIABILITY WARNING above, NOT evidence of a device pass]")
    print(f"emulated (v4 poly) vs hiprec (true erf): rel-L2 {rel_l2(emu, hip):.3e}  "
          f"max-abs {float(np.max(np.abs(emu.astype(np.float64) - hip))):.3e}")

    # Tails: GELU must saturate to 0 for x << 0 and to x for x >> 0. v4 has NO final clamp, so
    # these are the RAW polynomial values -- expect a small FIXED offset (~6.7e-3, from D(5)
    # evaluated at the clamp boundary), not exact 0 / exact x, and NOT growing with |x|.
    tails = np.array([-1000., -100., -50., -20., -12., -10., -6., -5., -1., 0.,
                      1., 5., 6., 10., 12., 20., 50., 100., 1000.], dtype=np.float32)
    print("tail x -> ref / emulated (v4, raw, no clamp):")
    for xv, rv, ev in zip(tails, gelu_erf_ref(tails), gelu_erf_emulated(tails)):
        print(f"  x={xv:+9.2f}  ref={rv:+12.6f}  emulated={ev:+12.6f}  diff={abs(float(ev)-float(rv)):.3e}")
    assert np.all(np.isfinite(gelu_erf_emulated(tails))), "emulated kernel must never produce NaN/Inf"
    # Bound (not exact-zero/exact-x, since v4 has no final clamp): the fixed D(5) offset only,
    # generously bounded at 2e-2 (D(5)~6.7e-3) to allow for f32 Horner rounding differences.
    assert abs(float(gelu_erf_emulated(np.array([-1000.], np.float32))[0])) < 2e-2, \
        "GELU must stay near 0 for x << 0 (within the fixed D(5) offset)"
    assert abs(float(gelu_erf_emulated(np.array([1000.], np.float32))[0]) - 1000.0) < 2e-2, \
        "GELU must stay near x for x >> 0 (within the fixed D(5) offset)"
    print("tail saturation checks OK (bounded, non-growing offset -- see comment above)")
    print(f"\nEXPECTED RAW DEVICE VALUE at x=11.9882 (the coordinator's v3 diagnostic point): "
          f"{float(gelu_erf_emulated(np.array([11.9882], np.float32))[0]):+.6f} "
          f"(golden={float(gelu_erf_ref(np.array([11.9882], np.float32))[0]):+.6f}, "
          f"host prediction only, NOT a device guarantee)")
