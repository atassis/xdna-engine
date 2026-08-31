#!/usr/bin/env python3
"""degree x chunk-count, together: is gelu-erf's multi-chunk failure register pressure or structural?

Everything below is measured on device this session; none of it is assumption.

  probe_abs_min.py         abs / min(x,5) / min(|x|,5) / 0.5*(x+|x|)   rel-L2 EXACTLY 0 at 1 chunk
  probe_absmin_chunks.py   the same four, at 1/2/4/8 chunks            rel-L2 EXACTLY 0 at ALL
  probe_horner_degree.py   Horner degree 2..9, ONE chunk               all green, worst 1.273e-07
  probe_gelu_bisect.py     gelu body, compile-time bound, 1 chunk      PASS 1.497e-07
                           gelu body, compile-time bound, 2 chunks     FAIL 1.112e+01
                           gelu body, compile-time bound, 4 chunks     FAIL 1.302e+01
                           gelu body, compile-time bound, 8 chunks     FAIL 3.208e+01
                           gelu body, ANY runtime bound, 1 chunk       FAIL 7.097e-01

So: every ingredient is individually correct multi-chunk, and the whole body is correct at one
chunk, but the whole body multi-chunk is not. The one axis never swept TOGETHER is degree x chunks.

  * If low degrees survive 2 chunks and high ones do not, the trigger is CROSS-ITERATION register
    pressure -- the loop keeps the live set alive across iterations even with unrolling disabled,
    so per-iteration pressure (which probe_horner_degree.py measured at 1 chunk) understates it.
    That is actionable: it gives a degree budget, and it fits sin.cc's own recorded spill history.
  * If EVERY degree including 1 fails at 2 chunks, it is structural, pressure is not the axis, and
    the minimal repro is "this body, 2 chunks" with no polynomial involved at all.

Either answer sharpens the upstream report. NOTE the body here is the FULL gelu shape (abs, min,
add, mul, Horner, sub) -- not a bare Horner, which probe_horner_degree.py already cleared. The
difference between this and that probe is exactly the abs/min/clamp prologue and the final sub.
"""
import time
from pathlib import Path

import numpy as np

import bricklib

GEN = bricklib.GEN
ROWS = 32
GATE = 3e-2
DEGREES = [1, 2, 3, 5, 9]
CHUNK_COUNTS = [1, 2]

rng = np.random.default_rng(29)
POOL = np.concatenate([
    rng.uniform(-12.0, -4.0, 8192),
    rng.uniform(-3.0, 3.0, 8192),
    rng.uniform(4.0, 12.0, 8192),
]).astype(np.float32)
rng.shuffle(POOL)


def coeffs(deg):
    # Benign, well-scaled, nothing ill-conditioned: any explosion is codegen, not numerics.
    return [1.0 / (k + 1) for k in range(deg + 1)]


def body_for(cs):
    """The FULL gelu shape: clamp prologue -> Horner -> final sub. Not a bare Horner."""
    lines = [
        "::aie::vector<float,16> ax = ::aie::abs(v);",
        "::aie::vector<float,16> axc = ::aie::min(ax, ::aie::broadcast<float,16>(5.0f));",
        "::aie::vector<float,16> s = ::aie::add(v, ax);",
        "s = ::aie::mul(s, ::aie::broadcast<float,16>(0.5f));",
        f"::aie::vector<float,16> p = ::aie::broadcast<float,16>({cs[0]:.9e}f);",
    ]
    for c in cs[1:]:
        lines.append("p = ::aie::mul(p, axc);")
        lines.append(f"p = ::aie::add(p, ::aie::broadcast<float,16>({c:.9e}f));")
    lines.append("::aie::vector<float,16> r = ::aie::sub(s, p);")
    lines.append("::aie::store_v(out + i*16, r);")
    return "\n    ".join(lines)


def ref_for(x, cs):
    axc = np.minimum(np.abs(x.astype(np.float64)), 5.0)
    p = np.full_like(axc, cs[0])
    for c in cs[1:]:
        p = p * axc + c
    return 0.5 * (x.astype(np.float64) + np.abs(x.astype(np.float64))) - p


print(f"{'degree':>7} {'live':>5} " + " ".join(f"{'c=' + str(c):>14s}" for c in CHUNK_COUNTS))
table = {}
for deg in DEGREES:
    cs = coeffs(deg)
    row, cells = [], []
    for nch in CHUNK_COUNTS:
        cols = nch * 16
        x = POOL[:ROWS * cols].reshape(ROWS, cols)
        ref = ref_for(x, cs)
        cc = GEN / f"gdc_d{deg}_c{nch}.cc"
        cc.write_text(
            "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
            f"// cachebust {int(time.time()*1000) % 10**9}\n"
            f'extern "C" void gdc_d{deg}_c{nch}(float *inp, float *out) {{\n'
            "  event0();\n  #pragma clang loop unroll(disable)\n"
            f"  for (int i = 0; i < {nch}; i++) {{\n"
            "    ::aie::vector<float,16> v = ::aie::load_v<16>(inp + i*16);\n"
            f"    {body_for(cs)}\n"
            "  }\n  event1();\n}\n")
        res = bricklib.verify_rowwise(
            name=f"gdc_d{deg}_c{nch}", brick_cc=cc, shim_body="",
            symbol=f"gdc_d{deg}_c{nch}", m=ROWS, in_cols=cols, out_cols=cols,
            x=x, expected=ref, gate=GATE)
        row.append((nch, res["ok"], res["rel_l2"]))
        cells.append(f"{res['rel_l2']:>11.3e}{'   ' if res['ok'] else ' ! '}")
    table[deg] = row
    print(f"{deg:7d} {deg+1:5d} " + " ".join(cells))

print()
c1 = {d: ok for d, r in table.items() for n, ok, _ in r if n == 1}
c2 = {d: ok for d, r in table.items() for n, ok, _ in r if n == 2}
green2 = sorted(d for d, ok in c2.items() if ok)
red2 = sorted(d for d, ok in c2.items() if not ok)
if not all(c1.values()):
    print("VERDICT: the 1-chunk anchor itself is red for some degree -- that contradicts\n"
          "         probe_horner_degree.py and probe_gelu_bisect.py r6a. Re-check before trusting c=2.")
elif green2 and red2:
    print(f"VERDICT: CROSS-ITERATION REGISTER PRESSURE. degrees {green2} survive 2 chunks, {red2} do not,\n"
          f"         while ALL are green at 1 chunk. The loop keeps the live set alive across\n"
          f"         iterations, so per-iteration pressure understates it. Degree budget at 2 chunks\n"
          f"         is {max(green2)}. This also fits sin.cc's recorded spill history.")
elif not green2:
    print("VERDICT: STRUCTURAL, not pressure. EVERY degree including 1 fails at 2 chunks while all\n"
          "         are green at 1. The minimal repro is 'this body, 2 chunks' with the polynomial\n"
          "         irrelevant -- and since abs/min/mul/add are each bit-exact to 8 chunks\n"
          "         (probe_absmin_chunks.py), it is their COMBINATION in a multi-chunk loop.")
else:
    print("VERDICT: every degree survives 2 chunks -- this body is fine and gelu_erf.cc differs from\n"
          "         it in some further way. Diff them.")
