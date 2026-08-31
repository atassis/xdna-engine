#!/usr/bin/env python3
"""Do aie::abs / aie::min survive a MULTI-CHUNK loop?

probe_abs_min.py exonerated these primitives -- abs, min(x,5), min(abs(x),5) and 0.5*(x+abs(x)) each
came back rel-L2 EXACTLY 0. But it ran them at ONE chunk with a COMPILE-TIME bound (`for i < 1`),
which is precisely the one configuration probe_gelu_bisect.py has since shown to be the ONLY
known-good one for gelu-erf:

    r6a  compile-time bound, 1 chunk    PASS 1.497e-07
    r6b  compile-time bound, 2 chunks   FAIL 1.112e+01
    r6c  compile-time bound, 4 chunks   FAIL 1.302e+01
    r6d  compile-time bound, 8 chunks   FAIL 3.208e+01     <- error grows with chunk count
    r4b  runtime bound,      1 chunk    FAIL 7.097e-01
    r7   runtime bound, no division     FAIL 7.097e-01     <- not the division, ANY runtime bound

So the primitives were cleared under exactly the conditions that cannot fail. This closes that hole.

WHY THIS IS NOT JUST "gelu-erf IS BROKEN". `snake` runs FOUR chunks with a RUNTIME bound
(`snake_f32(x, out, 64, 1.75f)` -> `snake_core<16>` -> `chunks = t/N`) and is device-green at
5.143e-06, re-verified twice this session. So neither multi-chunk nor a runtime bound is broken in
general -- something in gelu_erf_v's body interacts with the loop. The body's distinguishing feature
against snake's is that it uses `aie::abs` and `aie::min`; snake uses only mul/add/sub plus sin_v.

The failure symptom fits a clamp that stops working: gelu computes
`0.5*(x+|x|) - D(min(|x|,5))`, and large positive x returning ~0 or a huge extrapolated value is
exactly what you get if `ax`/`axc` is wrong. probe_horner_degree.py already showed a degree-9 Horner
is fine at 1 chunk, and D() alone is just mul/add -- the same ops snake uses safely.

So: run probe_abs_min.py's four cases at 1, 2, 4 and 8 chunks each, compile-time bound throughout.

  abs      : out = abs(x)
  min5     : out = min(x, 5)
  absmin   : out = min(abs(x), 5)
  sxax     : out = 0.5*(x + abs(x))
  mulonly  : out = x*0.5 + 1.0     <- CONTROL. no abs, no min, snake-like ops only. If this is green
                                      at every chunk count while absmin is not, the fault is isolated
                                      to abs/min rather than to multi-chunk looping as such.

A case that is green at 1 chunk and red at 2+ is a MINIMAL REPRO of an llvm-aie/Peano codegen defect
on a basic primitive, which is worth considerably more upstream than either brick it blocks.
"""
import time
from pathlib import Path

import numpy as np

import bricklib

GEN = bricklib.GEN
ROWS = 32
GATE = 3e-2
CHUNKS = [1, 2, 4, 8]

rng = np.random.default_rng(17)
# Reaches |x| = 12 so a lost clamp is unmistakable, and straddles the min() threshold at 5.
POOL = np.concatenate([
    rng.uniform(-12.0, -4.0, 4096),
    rng.uniform(-3.0, 3.0, 4096),
    rng.uniform(4.0, 12.0, 4096),
]).astype(np.float32)
rng.shuffle(POOL)

CASES = {
    "abs":     ("::aie::store_v(out + i*16, ::aie::abs(v));",
                lambda a: np.abs(a)),
    "min5":    ("::aie::store_v(out + i*16, ::aie::min(v, ::aie::broadcast<float,16>(5.0f)));",
                lambda a: np.minimum(a, 5.0)),
    "absmin":  ("::aie::vector<float,16> ax = ::aie::abs(v);\n"
                "    ::aie::store_v(out + i*16, ::aie::min(ax, ::aie::broadcast<float,16>(5.0f)));",
                lambda a: np.minimum(np.abs(a), 5.0)),
    "sxax":    ("::aie::vector<float,16> ax = ::aie::abs(v);\n"
                "    ::aie::vector<float,16> s = ::aie::add(v, ax);\n"
                "    s = ::aie::mul(s, ::aie::broadcast<float,16>(0.5f));\n"
                "    ::aie::store_v(out + i*16, s);",
                lambda a: 0.5 * (a + np.abs(a))),
    "mulonly": ("::aie::vector<float,16> s = ::aie::mul(v, ::aie::broadcast<float,16>(0.5f));\n"
                "    s = ::aie::add(s, ::aie::broadcast<float,16>(1.0f));\n"
                "    ::aie::store_v(out + i*16, s);",
                lambda a: a * 0.5 + 1.0),
}

print(f"{'case':10s} " + " ".join(f"{'c=' + str(c):>13s}" for c in CHUNKS))
results = {}
for name, (body, ref_fn) in CASES.items():
    row = []
    for nch in CHUNKS:
        cols = nch * 16
        x = POOL[:ROWS * cols].reshape(ROWS, cols)
        ref = ref_fn(x.astype(np.float64))
        cc = GEN / f"ac_{name}_{nch}.cc"
        cc.write_text(
            "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
            f"// cachebust {int(time.time()*1000) % 10**9}\n"
            f'extern "C" void ac_{name}_{nch}(float *inp, float *out) {{\n'
            "  event0();\n  #pragma clang loop unroll(disable)\n"
            f"  for (int i = 0; i < {nch}; i++) {{\n"
            "    ::aie::vector<float,16> v = ::aie::load_v<16>(inp + i*16);\n"
            f"    {body}\n"
            "  }\n  event1();\n}\n")
        res = bricklib.verify_rowwise(
            name=f"ac_{name}_{nch}", brick_cc=cc, shim_body="", symbol=f"ac_{name}_{nch}",
            m=ROWS, in_cols=cols, out_cols=cols, x=x, expected=ref, gate=GATE)
        row.append((nch, res["ok"], res["rel_l2"]))
    results[name] = row
    print(f"{name:10s} " + " ".join(
        f"{rl:>10.3e}{'  ' if ok else ' !'}" for _, ok, rl in row))

print()
broke = {n: [c for c, ok, _ in r if not ok] for n, r in results.items()}
broke = {n: c for n, c in broke.items() if c}
if not broke:
    print("VERDICT: every primitive survives every chunk count. abs/min are fully exonerated and\n"
          "         gelu-erf's multi-chunk failure is in its own body, not in these ops.")
elif "mulonly" in broke:
    print("VERDICT: even the mul/add CONTROL fails multi-chunk on this rail -- the defect is the\n"
          "         multi-chunk loop itself, NOT abs/min. But note snake runs 4 chunks green, so\n"
          "         reconcile against snake before reporting anything upstream.")
else:
    print(f"VERDICT: MINIMAL REPRO. {sorted(broke)} fail at chunk counts {broke} while the mul/add\n"
          "         control stays green. A basic aie_api primitive miscompiles in a multi-chunk\n"
          "         loop. This is an llvm-aie/Peano codegen defect worth reporting upstream, and it\n"
          "         explains gelu-erf without any reference to its polynomial.")
