#!/usr/bin/env python3
"""The control probe_codevol_repro.py never ran: its N=0 arm used a zero-trip loop and TIMED OUT.

Without it the flat 7.134e-01 across N=1..16 is uninterpretable -- it cannot distinguish "the MAC
loop causes it" from "this kernel/harness is wrong with no loop at all". This runs the no-loop case
as a SEPARATE SOURCE (no zero-trip for), plus two arms that step toward the shape of the isolation
probe, which passes at 2.825e-04.

  bare      : softmax only, vector store_v fill, output straight to the streamed buffer
  stackout  : softmax only, but output to a stack local then copied out (isolation's shape)
  scalarfill: softmax only, input filled by a scalar loop (isolation's shape) + stack output

If `bare` fails, the MAC loop was never the variable and the flat sweep is explained.
Whichever arm first goes green names the difference from the passing isolation call site.
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

HEAD = ("#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f"// cachebust {CB}\n"
        '#include "../../softmax/softmax.cc"\n')

ARMS = {
    # output straight to the streamed buffer, vector fill
    "bare": HEAD + 'extern "C" void cv_bare(float *inp, float *out) {\n'
                   "  event0();\n  alignas(64) float s[16];\n"
                   "  ::aie::store_v(s, ::aie::load_v<16>(inp));\n"
                   "  route_b_bricks::softmax_core<16>(s, out, 16);\n  event1();\n}\n",
    # isolation's output shape: softmax -> stack local, then scalar copy out
    "stackout": HEAD + 'extern "C" void cv_stackout(float *inp, float *out) {\n'
                       "  event0();\n  alignas(64) float s[16];\n  alignas(64) float p[16];\n"
                       "  ::aie::store_v(s, ::aie::load_v<16>(inp));\n"
                       "  route_b_bricks::softmax_core<16>(s, p, 16);\n"
                       "  for (int j = 0; j < 16; j++) out[j] = p[j];\n  event1();\n}\n",
    # isolation's input shape too: scalar fill loop rather than one store_v
    "scalarfill": HEAD + 'extern "C" void cv_scalarfill(float *inp, float *out) {\n'
                         "  event0();\n  alignas(64) float s[16];\n  alignas(64) float p[16];\n"
                         "  for (int j = 0; j < 16; j++) s[j] = inp[j];\n"
                         "  route_b_bricks::softmax_core<16>(s, p, 16);\n"
                         "  for (int j = 0; j < 16; j++) out[j] = p[j];\n  event1();\n}\n",
}

print(f"{'arm':12s} {'rel-L2':>12} {'non-finite':>12}  verdict")
for tag, src in ARMS.items():
    cc = GEN / f"cvc_{tag}.cc"
    cc.write_text(src)
    try:
        res = bricklib.verify_rowwise(name=f"cvc_{tag}", brick_cc=cc, shim_body="",
                                      symbol=f"cv_{tag}", m=ROWS, in_cols=COLS, out_cols=COLS,
                                      x=x, expected=ref, gate=GATE)
        got = np.asarray(res["got"], np.float64)
        nf = int((~np.isfinite(got)).sum())
        print(f"{tag:12s} {res['rel_l2']:12.3e} {nf:>7}/{got.size}  {'PASS' if res['ok'] else 'FAIL'}")
    except Exception as ex:
        print(f"{tag:12s} ERROR {type(ex).__name__}: {str(ex)[:80]}")
