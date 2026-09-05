#!/usr/bin/env python3
"""Does the every-other-tile fault track CODE SIZE or WORK DONE?

A straight-line Horner chain confounds them: D multiply-adds is both D units of arithmetic and
~173 B of .text per unit. D=8 fills 1.0000, D=32 fills 0.5000.

This separates them. Same 32 multiply-adds of arithmetic, two spellings:

  flat : straight-line, 32 emitted mul/add pairs   (~5632 B .text, known to FAIL)
  loop : for(i<32) over a 32-float coefficient array, unroll disabled -- SAME arithmetic, ~10x less code

  loop fills 1.0000 => CODE SIZE is the variable (a compilation/instruction-fetch story)
  loop fills 0.5000 => WORK/TIME is the variable (a per-tile duration story)
"""
import time
import numpy as np
import bricklib

GEN = bricklib.GEN
ROWS, COLS, GATE, D = 32, 16, 3e-2, 32
CB = int(time.time() * 1000) % 10**9
rng = np.random.default_rng(7)
x = rng.uniform(-1.0, 1.0, (ROWS, COLS)).astype(np.float32)
CS = [0.11, -0.23, 0.37, -0.41, 0.53, -0.61, 0.71, -0.79]
co = [CS[i % len(CS)] for i in range(D)]
ref = np.full_like(x.astype(np.float64), co[0])
for c in co[1:]:
    ref = ref * x.astype(np.float64) + c

flat_lines = [f"  ::aie::vector<float,16> p = ::aie::broadcast<float,16>({co[0]:.9e}f);"]
for c in co[1:]:
    flat_lines += ["  p = ::aie::mul(p, v);",
                   f"  p = ::aie::add(p, ::aie::broadcast<float,16>({c:.9e}f));"]

arr = ", ".join(f"{c:.9e}f" for c in co)
ARMS = {
    "flat": "\n".join(flat_lines),
    "loop": (f"  static const float K[{D}] = {{{arr}}};\n"
             "  ::aie::vector<float,16> p = ::aie::broadcast<float,16>(K[0]);\n"
             "  #pragma clang loop unroll(disable)\n"
             f"  for (int i = 1; i < {D}; i++) {{\n"
             "    p = ::aie::mul(p, v);\n"
             "    p = ::aie::add(p, ::aie::broadcast<float,16>(K[i]));\n  }"),
}

print(f"{'arm':6s} {'rel-L2':>12} {'fill':>8}  verdict")
for tag, body in ARMS.items():
    src = ("#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
           f"// cachebust {CB}\n"
           f'extern "C" void cvt_{tag}(float *inp, float *out) {{\n  event0();\n'
           "  ::aie::vector<float,16> v = ::aie::load_v<16>(inp);\n"
           + body + "\n  ::aie::store_v(out, p);\n  event1();\n}\n")
    cc = GEN / f"cvt_{tag}.cc"; cc.write_text(src)
    try:
        res = bricklib.verify_rowwise(name=f"cvt_{tag}", brick_cc=cc, shim_body="",
                                      symbol=f"cvt_{tag}", m=ROWS, in_cols=COLS, out_cols=COLS,
                                      x=x, expected=ref, gate=GATE)
        got = np.asarray(res["got"], np.float64)
        print(f"{tag:6s} {res['rel_l2']:12.3e} {float((got != 0).mean()):8.4f}  "
              f"{'PASS' if res['ok'] else 'FAIL'}")
    except Exception as ex:
        print(f"{tag:6s} ERROR {type(ex).__name__}: {str(ex)[:200]}")
