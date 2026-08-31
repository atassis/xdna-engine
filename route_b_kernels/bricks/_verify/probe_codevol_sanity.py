#!/usr/bin/env python3
"""Is the device output actually WRONG, or is my golden misaligned?

Every arm of probe_codevol_repro.py / _control.py returns exactly 7.134e-01 regardless of fill
spelling, output target or MAC count. A constant error across bodies that all compute the same
softmax is the signature of a bad EXPECTED, not a bad kernel. This dumps the values.
"""
import time
import numpy as np
import bricklib

GEN = bricklib.GEN
ROWS, COLS, GATE = 32, 16, 3e-2
CB = int(time.time() * 1000) % 10**9
rng = np.random.default_rng(7)
x = rng.uniform(-8.0, 8.0, (ROWS, COLS)).astype(np.float32)
e = np.exp(x.astype(np.float64) - x.astype(np.float64).max(axis=1, keepdims=True))
ref = e / e.sum(axis=1, keepdims=True)

src = ("#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
       f"// cachebust {CB}\n"
       '#include "../../softmax/softmax.cc"\n'
       'extern "C" void cv_san(float *inp, float *out) {\n'
       "  event0();\n  alignas(64) float s[16];\n  alignas(64) float p[16];\n"
       "  for (int j = 0; j < 16; j++) s[j] = inp[j];\n"
       "  route_b_bricks::softmax_core<16>(s, p, 16);\n"
       "  for (int j = 0; j < 16; j++) out[j] = p[j];\n  event1();\n}\n")
cc = GEN / "cvc_san.cc"; cc.write_text(src)
res = bricklib.verify_rowwise(name="cvc_san", brick_cc=cc, shim_body="", symbol="cv_san",
                              m=ROWS, in_cols=COLS, out_cols=COLS, x=x, expected=ref, gate=GATE)
got = np.asarray(res["got"], np.float64)
print(f"\nrel_l2 {res['rel_l2']:.3e}")
print("row0 in      :", np.round(x[0], 3))
print("row0 device  :", np.round(got[0], 4))
print("row0 expected:", np.round(ref[0], 4))
print(f"\ndevice row sums (first 4): {np.round(got[:4].sum(axis=1), 4)}   (softmax => 1.0)")
print(f"device argmax vs input argmax (first 8): {got[:8].argmax(1)} vs {x[:8].argmax(1)}")
# is the device output a softmax of SOME row of x?
d0 = got[0]
match = [r for r in range(ROWS) if np.allclose(d0, ref[r], atol=2e-2)]
print(f"device row0 matches expected row(s): {match}  (0 => genuinely different, not a shift)")
