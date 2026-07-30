#!/usr/bin/env python3
"""Host numpy golden for route_b_kernels/bricks/sin/sin.cc.

APPROACH (changed 2026-07-30, see the Lane C plan): polynomial, not a LUT.

The measured KB verdict (kb/lut-primitive-choice-lookup-vs-linear-approx) is that aie::linear_approx
beats a table lookup on sin by a wide margin -- but that assumes the lut<4> layout WORKS, and it
currently does not: the ab/cd duplication layout is an OPEN question in this repo
(log/2026-07/2026-07-25-rope-lut-root-cause-bank-granularity.md carries a same-day retraction and ends
"The layout question is OPEN"), it already blocks the rope-lut brick, and _verify/probe_linear_approx_abcd.py
measured that our best guess is still permuted. Snake needs ~1e-2, which a polynomial clears easily, so
the codec does not gate on that open question. Solving the layout is its own task.

The kernel therefore does: fold the argument into [-pi/2, pi/2] with f32 arithmetic (precision matters
for large |x|), then evaluate an odd Taylor polynomial, all in f32. mm_silu_epilogue's header warns that an all-f32
transcendental can exceed the per-tile cycle budget and hang, but that was a sigmoid/erf-class body;
4 multiplies + 3 adds is far cheaper, and the device gate is what settles it.

Usage: python3 golden.py
"""
import numpy as np

TWO_PI = 2.0 * np.pi
PI = np.pi
HALF_PI = 0.5 * np.pi

# Odd Taylor coefficients for sin(r) = r*(1 + c1 r^2 + c2 r^4 + c3 r^6) on [-pi/2, pi/2].
C1 = -1.0 / 6.0
C2 = 1.0 / 120.0
C3 = -1.0 / 5040.0


def bf16(x):
    """Round f32 -> bfloat16 -> f32, round-to-nearest-even (what the device does on narrowing)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint32)
    lsb = (u >> 16) & 1
    rounded = ((u + 0x7FFF + lsb) & 0xFFFF0000).astype(np.uint32)
    return rounded.view(np.float32)


def fold(x):
    """Fold x into [-pi/2, pi/2] preserving sin, using only ops the kernel has.

    Three stages, each a compare+select in the kernel:
      1. subtract 2*pi*round(x/2*pi)      -> [-pi, pi]   (the kernel rounds in pure f32; no int
                                                          round-trip, aie::to_float does not
                                                          instantiate on aie2p)
      2. wrap by +/-2*pi                  -> [-pi, pi]   (safety net)
      3. reflect about +/-pi/2 via sin(pi - t) = sin(t)  -> [-pi/2, pi/2]
    """
    x = np.asarray(x, dtype=np.float32)
    k = np.rint(x / TWO_PI).astype(np.float32)   # kernel rounds via (q + 2^23) - 2^23
    r = (x - TWO_PI * k).astype(np.float32)
    r = np.where(r > PI, r - TWO_PI, r).astype(np.float32)
    r = np.where(r < -PI, r + TWO_PI, r).astype(np.float32)
    r = np.where(r > HALF_PI, PI - r, r).astype(np.float32)
    r = np.where(r < -HALF_PI, -PI - r, r).astype(np.float32)
    return r


def sin_ref(x):
    """Ground truth."""
    return np.sin(np.asarray(x, dtype=np.float64)).astype(np.float32)


def sin_poly_emulated(x):
    """Bit-faithful emulation of the kernel: f32 fold, then an f32 Horner polynomial.

    The kernel stays f32 end to end (4 mults + 3 adds is well short of the per-tile budget that hangs
    an all-f32 SiLU), so this emulation must be f32 too -- narrowing here would make the golden
    disagree with the thing it is supposed to predict.
    """
    r = fold(x).astype(np.float32)
    r2 = (r * r).astype(np.float32)
    p = (np.float32(C3) * r2 + np.float32(C2)).astype(np.float32)
    p = (p * r2 + np.float32(C1)).astype(np.float32)
    p = (p * r2 + np.float32(1.0)).astype(np.float32)
    return (p * r).astype(np.float32)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Snake's argument is alpha*x: alpha a learned per-channel scale, x a post-conv activation.
    # Cover +/-64 (about 10 periods) so the fold is actually exercised.
    x = (rng.random(1 << 16, dtype=np.float32) * 128.0 - 64.0).astype(np.float32)
    got = sin_poly_emulated(x)
    ref = sin_ref(x)
    print("max abs err:", float(np.max(np.abs(got - ref))))
    print("rel-L2     :", rel_l2(got, ref))
    assert rel_l2(got, ref) <= 3e-2, "emulated poly path misses the brick gate"
    # The fold must actually land in [-pi/2, pi/2]; a fold bug is the likeliest failure and would
    # otherwise show up only as a mysterious device mismatch at large |x|.
    r = fold(x)
    assert np.all(np.abs(r) <= HALF_PI + 1e-5), f"fold escaped range: max |r| = {np.max(np.abs(r))}"
    print("fold range OK, max |r| =", float(np.max(np.abs(r))))
