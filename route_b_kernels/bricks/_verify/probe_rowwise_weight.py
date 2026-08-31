#!/usr/bin/env python3
"""Is the every-other-tile fault about softmax_core, or about per-tile WEIGHT?

Established: through an identical verify_rowwise call, a trivial copy fills 1.0000 while a kernel
containing softmax_core fills 0.5000 (deterministic, completed tiles bit-exact) and the smallest
softmax variant times out. The arms so far are ordered by per-tile work and so is the outcome, which
is equally consistent with "softmax_core is special" and with "a tile that takes long enough does not
complete".

This separates them with HEAVY BUT SIMPLE bodies -- a Horner chain of D multiply-adds, no helper
template, no include, no exp/tanh -- swept over D. Golden is the same polynomial in float64.

  fill stays 1.0 at every D  => weight is not the variable; softmax_core is implicated.
  fill drops as D grows      => per-tile weight is the variable and softmax was never special.
"""
import time
import numpy as np
import bricklib

GEN = bricklib.GEN
ROWS, COLS, GATE = 32, 16, 3e-2
CB = int(time.time() * 1000) % 10**9
rng = np.random.default_rng(7)
x = rng.uniform(-1.0, 1.0, (ROWS, COLS)).astype(np.float32)
CS = [0.11, -0.23, 0.37, -0.41, 0.53, -0.61, 0.71, -0.79]  # cycled to length D

print(f"{'D (mul-adds)':>13} {'rel-L2':>12} {'fill':>8}  verdict")
for D in (1, 8, 32, 128, 512):
    coeffs = [CS[i % len(CS)] for i in range(D)]
    ref = np.full_like(x.astype(np.float64), coeffs[0])
    for c in coeffs[1:]:
        ref = ref * x.astype(np.float64) + c
    lines = [f"  ::aie::vector<float,16> p = ::aie::broadcast<float,16>({coeffs[0]:.9e}f);"]
    for c in coeffs[1:]:
        lines.append("  p = ::aie::mul(p, v);")
        lines.append(f"  p = ::aie::add(p, ::aie::broadcast<float,16>({c:.9e}f));")
    src = ("#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
           f"// cachebust {CB}\n"
           f'extern "C" void rw_{D}(float *inp, float *out) {{\n  event0();\n'
           "  ::aie::vector<float,16> v = ::aie::load_v<16>(inp);\n"
           + "\n".join(lines) + "\n"
           "  ::aie::store_v(out, p);\n  event1();\n}\n")
    cc = GEN / f"rw_{D}.cc"; cc.write_text(src)
    try:
        res = bricklib.verify_rowwise(name=f"rw_{D}", brick_cc=cc, shim_body="", symbol=f"rw_{D}",
                                      m=ROWS, in_cols=COLS, out_cols=COLS, x=x, expected=ref,
                                      gate=GATE)
        got = np.asarray(res["got"], np.float64)
        print(f"{D:>13} {res['rel_l2']:12.3e} {float((got != 0).mean()):8.4f}  "
              f"{'PASS' if res['ok'] else 'FAIL'}")
    except Exception as ex:
        print(f"{D:>13} ERROR {type(ex).__name__}: {str(ex)[:70]}")
