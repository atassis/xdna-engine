#!/usr/bin/env python3
"""Isolate aie::abs and aie::min on vector<float,16>, aie2p.

gelu-erf v3 uses ONLY abs/min/mul/add/sub, yet returns exactly 0.0 for large positive x
(x=+11.9882 -> 0.00000e+00, golden +11.98815) while x=+1 is correct to six digits. Its formula is
`y = 0.5*(x + abs(x)) - D(min(abs(x), 5))`, so a device 0 at large positive x requires
`x + abs(x) == 0`, i.e. abs(x) == -x. That is a claim about a primitive, and per AGENTS.md a defect
claim gets verified IN ISOLATION before it is allowed to justify anything -- the 2026-07-30
sliding_mul episode is the standing example of a primitive blamed from a full-pipeline measurement
and later refuted by a single-core repro.

Four one-line kernels on the same rail the brick uses, same widths, over the same value range:

  abs      : out = abs(x)
  min5     : out = min(x, 5)
  absmin   : out = min(abs(x), 5)        <- exactly what gelu_erf_v computes
  sxax     : out = 0.5*(x + abs(x))      <- the term that must be collapsing to 0

Each is gated against numpy directly. If all four are green, abs/min are exonerated and the defect
is in the Horner chain or its codegen; if `sxax` or `abs` is red at large x, the primitive is the
bug and gelu-erf is not fixable by any amount of polynomial work.
"""
import time
from pathlib import Path

import numpy as np

import bricklib

GEN = bricklib.GEN
ROWS, COLS = 32, 16
GATE = 3e-2

# Deliberately spans the region where gelu-erf breaks, plus the region where it is correct.
rng = np.random.default_rng(11)
x = np.concatenate([
    rng.uniform(-12.0, -4.0, ROWS * COLS // 4),
    rng.uniform(-3.0, 3.0, ROWS * COLS // 4),
    rng.uniform(4.0, 12.0, ROWS * COLS // 4),
    rng.uniform(-1.0, 1.0, ROWS * COLS - 3 * (ROWS * COLS // 4)),
]).astype(np.float32)
rng.shuffle(x)
x = x.reshape(ROWS, COLS)

CASES = {
    "abs":    ("::aie::store_v(out + i*16, ::aie::abs(v));",
               lambda a: np.abs(a)),
    "min5":   ("::aie::store_v(out + i*16, ::aie::min(v, ::aie::broadcast<float,16>(5.0f)));",
               lambda a: np.minimum(a, 5.0)),
    "absmin": ("::aie::vector<float,16> ax = ::aie::abs(v);\n"
               "    ::aie::store_v(out + i*16, ::aie::min(ax, ::aie::broadcast<float,16>(5.0f)));",
               lambda a: np.minimum(np.abs(a), 5.0)),
    "sxax":   ("::aie::vector<float,16> ax = ::aie::abs(v);\n"
               "    ::aie::vector<float,16> s = ::aie::add(v, ax);\n"
               "    s = ::aie::mul(s, ::aie::broadcast<float,16>(0.5f));\n"
               "    ::aie::store_v(out + i*16, s);",
               lambda a: 0.5 * (a + np.abs(a))),
}

print(f"{'case':10s} {'rel-L2':>12} {'max-abs':>12}  verdict")
bad = []
for name, (body, ref_fn) in CASES.items():
    cc = GEN / f"prim_{name}.cc"
    cc.write_text(
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f"// cachebust {int(time.time()*1000) % 10**9}\n"
        f'extern "C" void prim_{name}(float *inp, float *out) {{\n'
        "  event0();\n"
        f"  for (int i = 0; i < {COLS // 16}; i++) {{\n"
        "    ::aie::vector<float,16> v = ::aie::load_v<16>(inp + i*16);\n"
        f"    {body}\n"
        "  }\n  event1();\n}\n")
    ref = ref_fn(x.astype(np.float64))
    res = bricklib.verify_rowwise(
        name=f"prim_{name}", brick_cc=cc, shim_body="", symbol=f"prim_{name}",
        m=ROWS, in_cols=COLS, out_cols=COLS, x=x, expected=ref, gate=GATE)
    got = np.asarray(res["got"], np.float64)
    ma = float(np.max(np.abs(got - ref)))
    print(f"{name:10s} {res['rel_l2']:12.3e} {ma:12.3e}  {'PASS' if res['ok'] else 'FAIL'}")
    if not res["ok"]:
        bad.append(name)
        w = np.argsort(np.abs(got - ref).ravel())[::-1][:4]
        for fi in w:
            r, c = divmod(int(fi), COLS)
            print(f"    x={x[r,c]:+9.4f}  numpy={ref[r,c]:+11.5f}  device={got[r,c]:+13.5e}")

print()
print("VERDICT: " + ("abs/min/add/mul are all CORRECT in isolation -- the defect is in the "
                     "Horner chain or its codegen, not the primitives."
                     if not bad else
                     f"BROKEN IN ISOLATION: {', '.join(bad)}. gelu-erf cannot be fixed by "
                     "changing the polynomial."))
