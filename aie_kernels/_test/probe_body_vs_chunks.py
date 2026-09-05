#!/usr/bin/env python3
"""Why does snake's body survive a multi-iteration loop when gelu's does not?

This is the ONE question blocking an upstream report on the loop defect. Everything else is settled
and measured (see kb/kernel-internal-loops-miscompile-put-volume-in-the-worker):

  gelu body, compile-time bound, 1 chunk    PASS 1.497e-07
  gelu body, compile-time bound, 2 chunks   FAIL 1.112e+01   <- error grows with degree AND count
  gelu body, ANY runtime bound              FAIL 7.097e-01
  abs / min / mul / add, 1..8 chunks        rel-L2 EXACTLY 0 (probe_absmin_chunks.py)
  Horner degree 2..9, 1 chunk               all green (probe_horner_degree.py)

BUT `snake` ships a FOUR-chunk loop with a RUNTIME bound and is device-green at 5.143e-06, twice
re-verified. Its body is not obviously lighter: `snake_core` does mul -> sin_v -> mul -> mul -> add,
and `sin_v` itself contains an argument fold plus a SIX-term Horner. So "kernel loops miscompile" is
refutable in one line by a maintainer pointing at snake, and live-vector count does not predict it.

So: run both bodies through ONE harness at 1/2/4 chunks, compile-time bounds throughout, and bisect
the structural difference between them.

  snake     mul, sin_v (fold + 6-term Horner), mul, mul, add     -- expected GREEN (it ships)
  gelu      abs, min, add, mul, 5-term Horner, sub               -- expected RED at 2+
  gelu_noclamp   gelu with abs+min REMOVED (Horner eats x directly, s = x*0.5)
  gelu_absonly   gelu with min removed, abs kept
  gelu_minonly   gelu with abs removed, min kept

`probe_absmin_chunks.py` already showed abs and min are individually fine multi-chunk in a TRIVIAL
body. What it could not show is whether they are fine multi-chunk when COMBINED with a Horner chain.
If gelu_noclamp is green at 2+ while gelu is red, that combination is the trigger and the report
writes itself. If gelu_noclamp is ALSO red, the clamp is exonerated and the difference from snake is
elsewhere -- the next candidate being sin_v's argument fold, which snake has and gelu does not.

Every body is gated against its own float64 reference, so a green here means the KERNEL is right,
not that the reference is lenient.
"""
import time
from pathlib import Path

import numpy as np

import bricklib

GEN = bricklib.GEN
ROWS = 32
GATE = 3e-2
CHUNK_COUNTS = [1, 2, 4]

ALPHA, INV_ALPHA = 1.75, 1.0 / 1.75
GCS = [1.0 / (k + 1) for k in range(6)]          # degree-5, benign, same scale as gelu's real fit
SIN_C = [-2.50521083854417188e-08, 2.75573192239858907e-06, -1.98412698412698413e-04,
         8.33333333333333322e-03, -1.66666666666666657e-01, 1.0]   # sin.cc's own coefficients

rng = np.random.default_rng(31)
POOL = np.concatenate([
    rng.uniform(-12.0, -4.0, 8192),
    rng.uniform(-3.0, 3.0, 8192),
    rng.uniform(4.0, 12.0, 8192),
]).astype(np.float32)
rng.shuffle(POOL)

f32 = np.float32


def horner(v, cs):
    p = np.full_like(v, cs[0])
    for c in cs[1:]:
        p = p * v + c
    return p


def _hb(cs, arg):
    """Emit a Horner chain over `arg` into vector `p`."""
    out = [f"::aie::vector<float,16> p = ::aie::broadcast<float,16>({cs[0]:.9e}f);"]
    for c in cs[1:]:
        out.append(f"p = ::aie::mul(p, {arg});")
        out.append(f"p = ::aie::add(p, ::aie::broadcast<float,16>({c:.9e}f));")
    return out


def sin_ref(r):
    r = np.asarray(r, np.float64)
    k = np.round(r / (2 * np.pi))
    r = r - k * 2 * np.pi
    return horner(r * r, SIN_C) * r


BODIES = {
    # snake_core's exact shape, sin_v inlined. Uses 1.5*2^23 (the corrected constant) so the fold
    # actually rounds on host; on device the fold is inert either way, which is why the reference
    # below models it as an EXACT round -- see kb/magic-number-rounding-is-a-noop-on-aie2p.
    "snake": (
        ["::aie::vector<float,16> ax = ::aie::mul(v, ::aie::broadcast<float,16>(%.9ef));" % ALPHA,
         "::aie::vector<float,16> k = ::aie::mul(ax, ::aie::broadcast<float,16>(0.15915494309189535f));",
         "k = ::aie::add(k, ::aie::broadcast<float,16>(12582912.0f));",
         "k = ::aie::sub(k, ::aie::broadcast<float,16>(12582912.0f));",
         "::aie::vector<float,16> kt = ::aie::mul(k, ::aie::broadcast<float,16>(6.28318530717958648f));",
         "::aie::vector<float,16> r = ::aie::sub(ax, kt);",
         "::aie::vector<float,16> r2 = ::aie::mul(r, r);"]
        + _hb(SIN_C, "r2")
        + ["::aie::vector<float,16> s = ::aie::mul(p, r);",
           "::aie::vector<float,16> s2 = ::aie::mul(s, s);",
           "::aie::vector<float,16> q = ::aie::mul(s2, ::aie::broadcast<float,16>(%.9ef));" % INV_ALPHA,
           "::aie::store_v(out + i*16, ::aie::add(v, q));"],
        lambda x: x.astype(np.float64) + sin_ref(x.astype(np.float64) * ALPHA) ** 2 * INV_ALPHA),

    "gelu": (
        ["::aie::vector<float,16> ax = ::aie::abs(v);",
         "::aie::vector<float,16> axc = ::aie::min(ax, ::aie::broadcast<float,16>(5.0f));",
         "::aie::vector<float,16> s = ::aie::add(v, ax);",
         "s = ::aie::mul(s, ::aie::broadcast<float,16>(0.5f));"]
        + _hb(GCS, "axc")
        + ["::aie::store_v(out + i*16, ::aie::sub(s, p));"],
        lambda x: 0.5 * (x.astype(np.float64) + np.abs(x.astype(np.float64)))
                  - horner(np.minimum(np.abs(x.astype(np.float64)), 5.0), GCS)),

    "gelu_noclamp": (
        ["::aie::vector<float,16> s = ::aie::mul(v, ::aie::broadcast<float,16>(0.5f));"]
        + _hb(GCS, "v")
        + ["::aie::store_v(out + i*16, ::aie::sub(s, p));"],
        lambda x: 0.5 * x.astype(np.float64) - horner(x.astype(np.float64), GCS)),

    "gelu_absonly": (
        ["::aie::vector<float,16> ax = ::aie::abs(v);",
         "::aie::vector<float,16> s = ::aie::add(v, ax);",
         "s = ::aie::mul(s, ::aie::broadcast<float,16>(0.5f));"]
        + _hb(GCS, "ax")
        + ["::aie::store_v(out + i*16, ::aie::sub(s, p));"],
        lambda x: 0.5 * (x.astype(np.float64) + np.abs(x.astype(np.float64)))
                  - horner(np.abs(x.astype(np.float64)), GCS)),

    "gelu_minonly": (
        ["::aie::vector<float,16> axc = ::aie::min(v, ::aie::broadcast<float,16>(5.0f));",
         "::aie::vector<float,16> s = ::aie::mul(v, ::aie::broadcast<float,16>(0.5f));"]
        + _hb(GCS, "axc")
        + ["::aie::store_v(out + i*16, ::aie::sub(s, p));"],
        lambda x: 0.5 * x.astype(np.float64)
                  - horner(np.minimum(x.astype(np.float64), 5.0), GCS)),
}

print(f"{'body':14s} " + " ".join(f"{'c=' + str(c):>15s}" for c in CHUNK_COUNTS))
table = {}
for name, (lines, ref_fn) in BODIES.items():
    cells, row = [], []
    for nch in CHUNK_COUNTS:
        cols = nch * 16
        x = POOL[:ROWS * cols].reshape(ROWS, cols)
        ref = ref_fn(x)
        cc = GEN / f"bvc_{name}_{nch}.cc"
        cc.write_text(
            "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
            f"// cachebust {int(time.time()*1000) % 10**9}\n"
            f'extern "C" void bvc_{name}_{nch}(float *inp, float *out) {{\n'
            "  event0();\n  #pragma clang loop unroll(disable)\n"
            f"  for (int i = 0; i < {nch}; i++) {{\n"
            "    ::aie::vector<float,16> v = ::aie::load_v<16>(inp + i*16);\n    "
            + "\n    ".join(lines) + "\n"
            "  }\n  event1();\n}\n")
        res = bricklib.verify_rowwise(
            name=f"bvc_{name}_{nch}", brick_cc=cc, shim_body="", symbol=f"bvc_{name}_{nch}",
            m=ROWS, in_cols=cols, out_cols=cols, x=x, expected=ref, gate=GATE)
        row.append((nch, res["ok"], res["rel_l2"]))
        cells.append(f"{res['rel_l2']:>12.3e}{'   ' if res['ok'] else ' ! '}")
    table[name] = row
    print(f"{name:14s} " + " ".join(cells))

print()
def ok_at(n, c):
    return next(o for ch, o, _ in table[n] if ch == c)

snake_multi = ok_at("snake", 2) and ok_at("snake", 4)
gelu_multi = ok_at("gelu", 2)
if not snake_multi:
    print("VERDICT: snake's body is RED here too. Then the shipped snake brick's greenness comes from\n"
          "         something OUTSIDE the body (its harness shape or its own gate's tolerance), and\n"
          "         the counterexample dissolves -- which would UNBLOCK the upstream report.")
elif gelu_multi:
    print("VERDICT: gelu is green multi-chunk in this harness -- contradicts probe_gelu_degree_chunks.\n"
          "         Re-check before trusting either.")
else:
    variants = {n: ok_at(n, 2) for n in ("gelu_noclamp", "gelu_absonly", "gelu_minonly")}
    green = sorted(n for n, o in variants.items() if o)
    print(f"VERDICT: snake GREEN multi-chunk, gelu RED -- the counterexample is real and reproduced\n"
          f"         side by side. Variants green at 2 chunks: {green or 'NONE'}.")
    if green:
        print("         -> removing that op FIXES it, so the trigger is that op combined with a\n"
              "            Horner chain in a multi-iteration loop. That is the upstream report.")
    else:
        print("         -> every gelu variant is red, so abs/min are NOT the trigger. The remaining\n"
              "            difference from snake is sin_v's argument fold (which snake has and gelu\n"
              "            lacks) or the store-from-add vs store-from-sub tail. Next rung.")
