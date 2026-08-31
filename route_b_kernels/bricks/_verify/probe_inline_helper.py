#!/usr/bin/env python3
"""IDENTICAL math, two spellings: inline in the bound symbol vs behind a static inline helper.

This is the controlled A/B for the constraint the task file states but that nothing in the tree
demonstrates in isolation:

    "Macro-expand the body into the bound kernel symbol. Identical code behind a
     `static inline` function returned NaN."

What forced this probe: probe_horner_degree.py put a degree-9 Horner (10 live vectors) INLINE in the
bound extern "C" symbol and it was GREEN, refuting register pressure as gelu-erf's cause. Both
failing bricks -- gelu-erf (7.04e-01 .. 9.54e+00 across four rewrites) and sin (7.091e-01, open
across sessions) -- put their body behind a `static inline template<int N>` helper. Every green
brick's math either sits inline or is consumed differently.

So hold the MATH fixed and vary only the spelling:

  A `inline`  : the gelu degree-5 body written directly inside extern "C" gelu_A
  B `helper`  : the SAME body inside `template<int N> static inline gelu_v(...)`, called by gelu_B

Same coefficients, same clamp, same loop, same widths, same inputs, same gate. If A is green and B
is red, the constraint is real, isolated, and explains sin as well as gelu-erf -- and it is a
minimal repro of a Peano/llvm-aie inlining defect, which is worth more upstream than either brick.
If both are green, the spelling is exonerated and gelu_erf.cc differs from B in some third way.
"""
import time
from pathlib import Path

import numpy as np

import bricklib

GEN = bricklib.GEN
ROWS, COLS = 32, 16
GATE = 3e-2

# gelu-erf v4's own degree-5 D(ax) fit, read from the brick so the two stay in lockstep.
BRICK = (Path(__file__).parent.parent / "gelu-erf").resolve()
import importlib.util
spec = importlib.util.spec_from_file_location("gelu_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)
CS = [float(c) for c in golden._D_COEFFS]

rng = np.random.default_rng(5)
x = np.concatenate([
    rng.uniform(-12.0, -4.0, ROWS * COLS // 3),
    rng.uniform(-3.0, 3.0, ROWS * COLS // 3),
    rng.uniform(4.0, 12.0, ROWS * COLS - 2 * (ROWS * COLS // 3)),
]).astype(np.float32)
rng.shuffle(x)
x = x.reshape(ROWS, COLS)

# Golden: exactly what both spellings are supposed to compute.
axc = np.minimum(np.abs(x.astype(np.float64)), 5.0)
p = np.full_like(axc, CS[0])
for c in CS[1:]:
    p = p * axc + c
ref = 0.5 * (x.astype(np.float64) + np.abs(x.astype(np.float64))) - p

BODY = "\n".join(
    ["    ::aie::vector<float,16> ax = ::aie::abs(v);",
     "    ::aie::vector<float,16> axc = ::aie::min(ax, ::aie::broadcast<float,16>(5.0f));",
     "    ::aie::vector<float,16> s = ::aie::add(v, ax);",
     "    s = ::aie::mul(s, ::aie::broadcast<float,16>(0.5f));",
     f"    ::aie::vector<float,16> p = ::aie::broadcast<float,16>({CS[0]:.9e}f);"]
    + sum([["    p = ::aie::mul(p, axc);",
            f"    p = ::aie::add(p, ::aie::broadcast<float,16>({c:.9e}f));"] for c in CS[1:]], [])
    + ["    ::aie::vector<float,16> r = ::aie::sub(s, p);"])

CB = int(time.time() * 1000) % 10**9

SRC = {
    # A: body inline in the bound extern "C" symbol.
    "inline": (
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f"// cachebust {CB}\n"
        'extern "C" void gelu_A(float *inp, float *out) {\n'
        "  event0();\n  #pragma clang loop unroll(disable)\n"
        "  for (int i = 0; i < 1; i++) {\n"
        "    ::aie::vector<float,16> v = ::aie::load_v<16>(inp + i*16);\n"
        f"{BODY}\n"
        "    ::aie::store_v(out + i*16, r);\n"
        "  }\n  event1();\n}\n", "gelu_A"),
    # B: identical body behind a template static inline helper -- gelu_erf.cc / sin.cc's shape.
    "helper": (
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f"// cachebust {CB}\n"
        "template <int N>\n"
        "static inline ::aie::vector<float,N> gelu_v(::aie::vector<float,N> v) {\n"
        f"{BODY.replace(',16>', ',N>').replace('<16>', '<N>')}\n"
        "  return r;\n}\n"
        'extern "C" void gelu_B(float *inp, float *out) {\n'
        "  event0();\n  #pragma clang loop unroll(disable)\n"
        "  for (int i = 0; i < 1; i++) {\n"
        "    ::aie::store_v(out + i*16, gelu_v<16>(::aie::load_v<16>(inp + i*16)));\n"
        "  }\n  event1();\n}\n", "gelu_B"),
}

# C/D: the REAL question. gelu_erf_core/sin_core derive `chunks = n / N` from a RUNTIME int32
# parameter; every probe above used a compile-time bound. The parent task already recorded
# "sin_core's RUNTIME loop trip count corrupts after exactly 4 iterations" -- gelu_erf_core copied
# that shape. Hold the math AND the helper spelling fixed; vary only where the trip count comes from.
NCH = 4   # 4 chunks of 16 = 64 floats, exactly verify_gelu_erf.py's COLS and the recorded threshold
for tag, sig, loop in (
    ("rt_count", "float *inp, float *out, int32_t n", "const int chunks = n / 16;\n"
                                                       "  for (int i = 0; i < chunks; i++) {"),
    ("ct_count", "float *inp, float *out, int32_t n", f"(void)n;\n"
                                                       f"  for (int i = 0; i < {NCH}; i++) {{"),
):
    SRC[tag] = (
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f"// cachebust {CB}\n"
        "template <int N>\n"
        "static inline ::aie::vector<float,N> gelu_v_%s(::aie::vector<float,N> v) {\n" % tag
        + BODY.replace(',16>', ',N>').replace('<16>', '<N>') + "\n  return r;\n}\n"
        f'extern "C" void gelu_{tag}_k({sig}) {{\n'
        "  event0();\n  #pragma clang loop unroll(disable)\n  " + loop + "\n"
        "    ::aie::store_v(out + i*16, gelu_v_%s<16>(::aie::load_v<16>(inp + i*16)));\n" % tag
        + "  }\n  event1();\n}\n", f"gelu_{tag}_k")

print(f"{'spelling':10s} {'rel-L2':>12} {'max-abs':>12}  verdict")
out = {}
for tag, (src, sym) in SRC.items():
    cc = GEN / f"inl_{tag}.cc"
    cc.write_text(src)
    shim = "" if tag in ("inline", "helper") else (
        f'extern "C" void {sym}_w(float *x, float *o) {{ {sym}(x, o, 64); }}\n')
    res = bricklib.verify_rowwise(
        name=f"inl_{tag}", brick_cc=cc, shim_body=shim,
        symbol=(sym if tag in ("inline", "helper") else f"{sym}_w"),
        m=(ROWS if tag in ('inline','helper') else ROWS//4),
        in_cols=(COLS if tag in ('inline','helper') else 64),
        out_cols=(COLS if tag in ('inline','helper') else 64),
        x=(x if tag in ('inline','helper') else x.reshape(ROWS//4, 64)),
        expected=(ref if tag in ('inline','helper') else ref.reshape(ROWS//4, 64)),
        gate=GATE)
    got = np.asarray(res["got"], np.float64)
    _r = ref if tag in ('inline','helper') else ref.reshape(ROWS//4, 64)
    ma = float(np.max(np.abs(got - _r)))
    out[tag] = res["ok"]
    print(f"{tag:10s} {res['rel_l2']:12.3e} {ma:12.3e}  {'PASS' if res['ok'] else 'FAIL'}")

print()
if out["inline"] and not out["helper"]:
    print("VERDICT: CONFIRMED + MINIMAL REPRO. Identical math is correct inline and WRONG behind a\n"
          "         static inline template helper. That is the constraint the task file states, now\n"
          "         isolated, and it explains sin.cc (same shape, red across sessions) too.")
elif out["inline"] and out["helper"]:
    print("VERDICT: both green -- the spelling is EXONERATED. gelu_erf.cc differs from B in some\n"
          "         third way; diff them rather than rewriting the math again.")
else:
    print("VERDICT: the inline form is red too -- this math does not survive at all here; the\n"
          "         difference from probe_horner_degree.py's green chain is the abs/add/sub around it.")
