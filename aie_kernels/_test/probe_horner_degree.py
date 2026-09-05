#!/usr/bin/env python3
"""At what Horner degree does the clamped value stop surviving the chain?

Established already this session:
  * abs / min / add / mul and `0.5*(x+abs(x))` are BIT-EXACT in isolation (probe_abs_min.py).
  * gelu-erf's blow-up scales with polynomial degree: degree 9 -> max-abs 1.85e+05,
    degree 5 -> 2.80e+02. A 660x reduction from shortening the chain alone.
  * The blow-up magnitude is what UNCLAMPED extrapolation of that same polynomial gives, i.e. the
    clamped `axc` is not what the chain actually evaluates at in the failing lanes -- even though
    the clamp is bit-exact on its own.

That is the register-pressure aliasing sin.cc's header records ("p3 returned r2's value ... zeros
interleaved"), and sin is red at 7.091e-01 for the same reason across sessions. So the question that
decides whether gelu-erf is shippable AT ALL on this toolchain is: is there a chain length short
enough that `axc` survives?

This isolates exactly that, with NO gelu semantics in it. The kernel is:

    ax  = abs(x)
    axc = min(ax, 5)          <- the value whose survival is in question
    p   = Horner_deg(axc)     <- chain of the given length, coefficients all 1/(k+1)
    out = p

and the golden is the same polynomial evaluated on numpy's min(abs(x),5). Inputs deliberately reach
|x| = 12, so if `axc` is ever lost the result explodes and cannot be mistaken for fit error.

A degree at which this is green, next to a degree at which it is red, is a MINIMAL REPRO of an
llvm-aie codegen defect -- worth far more upstream than a working gelu brick.
"""
import time
from pathlib import Path

import numpy as np

import bricklib

GEN = bricklib.GEN
ROWS, COLS = 32, 16
GATE = 3e-2
DEGREES = [2, 3, 4, 5, 7, 9]

rng = np.random.default_rng(3)
x = np.concatenate([
    rng.uniform(-12.0, -4.0, ROWS * COLS // 3),
    rng.uniform(-3.0, 3.0, ROWS * COLS // 3),
    rng.uniform(4.0, 12.0, ROWS * COLS - 2 * (ROWS * COLS // 3)),
]).astype(np.float32)
rng.shuffle(x)
x = x.reshape(ROWS, COLS)

# Benign, well-scaled coefficients: nothing here is ill-conditioned, so any explosion is codegen.
def coeffs(deg):
    return [1.0 / (k + 1) for k in range(deg + 1)]

print(f"{'degree':>7} {'live':>5} {'rel-L2':>12} {'max-abs':>12}  verdict")
rows = []
for deg in DEGREES:
    cs = coeffs(deg)
    body = [f"::aie::vector<float,16> p = ::aie::broadcast<float,16>({cs[0]:.9e}f);"]
    for c in cs[1:]:
        body.append("p = ::aie::mul(p, axc);")
        body.append(f"p = ::aie::add(p, ::aie::broadcast<float,16>({c:.9e}f));")
    cc = GEN / f"horner_d{deg}.cc"
    cc.write_text(
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f"// cachebust {int(time.time()*1000) % 10**9}\n"
        f'extern "C" void horner_d{deg}(float *inp, float *out) {{\n'
        "  event0();\n"
        "  #pragma clang loop unroll(disable)\n"
        "  for (int i = 0; i < 1; i++) {\n"
        "    ::aie::vector<float,16> v = ::aie::load_v<16>(inp + i*16);\n"
        "    ::aie::vector<float,16> ax = ::aie::abs(v);\n"
        "    ::aie::vector<float,16> axc = ::aie::min(ax, ::aie::broadcast<float,16>(5.0f));\n"
        "    " + "\n    ".join(body) + "\n"
        "    ::aie::store_v(out + i*16, p);\n"
        "  }\n  event1();\n}\n")

    axc = np.minimum(np.abs(x.astype(np.float64)), 5.0)
    ref = np.zeros_like(axc) + cs[0]
    for c in cs[1:]:
        ref = ref * axc + c

    res = bricklib.verify_rowwise(
        name=f"horner_d{deg}", brick_cc=cc, shim_body="", symbol=f"horner_d{deg}",
        m=ROWS, in_cols=COLS, out_cols=COLS, x=x, expected=ref, gate=GATE)
    got = np.asarray(res["got"], np.float64)
    ma = float(np.max(np.abs(got - ref)))
    rows.append((deg, res["ok"], res["rel_l2"], ma))
    print(f"{deg:7d} {deg+1:5d} {res['rel_l2']:12.3e} {ma:12.3e}  "
          f"{'PASS' if res['ok'] else 'FAIL'}")

print()
green = [d for d, ok, _, _ in rows if ok]
red = [d for d, ok, _, _ in rows if not ok]
if green and red:
    print(f"VERDICT: MINIMAL REPRO. degree {max(green)} green, degree {min(red)} red, same kernel "
          f"shape, benign coefficients.\n         The clamped value stops surviving the chain "
          f"between them. This is an llvm-aie codegen bug worth reporting upstream.")
elif not red:
    print("VERDICT: every degree green -- the clamp survives here, so gelu-erf's failure needs a\n"
          "         different explanation than chain length alone. Re-read.")
else:
    print("VERDICT: every degree red, including degree 2 -- the defect is not chain length; even a\n"
          "         minimal chain loses the clamped value. Suspect the abs/min feeding a loop body.")
