#!/usr/bin/env python3
"""MINIMAL REPRO attempt: does softmax_core's result degrade as the ENCLOSING FUNCTION grows?

prefill_attn's defect is established as a codegen fault localised to the softmax call
boundary: with the emitted data and golden held
identical, adding vector code BEFORE the softmax call makes the result wrong-but-finite, and adding
code AFTER it -- code that never writes the buffer -- makes it NaN. Severity tracked total in-function
vector code volume and nothing else.

prefill_attn.cc is far too large to file as a reproducer. This is the smallest thing that should show
the same scaling: ONE softmax_core call on a fixed input, preceded by N iterations of throwaway
vector MAC work whose result cannot reach the output. The golden is softmax(x) for EVERY arm, so any
movement across N is the compiler, not the math.

  N = 0   -- softmax alone; must be green, or this probe is measuring something else
  N > 0   -- identical answer required; if rel-L2 grows with N, the scaling is reproduced minimally

Run:  ./run.sh probe_codevol_repro.py
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

print(f"{'N':>4} {'rel-L2':>12} {'non-finite':>11}  verdict")
results = {}
for N in (0, 1, 2, 4, 8, 16):
    src = (
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f"// cachebust {CB}\n"
        '#include "../../softmax/softmax.cc"\n'
        f'extern "C" void cv_{N}(float *inp, float *out) {{\n'
        "  event0();\n"
        "  alignas(64) float scores[16];\n"
        "  ::aie::store_v(scores, ::aie::load_v<16>(inp));\n"
        # N iterations of throwaway vector MAC -- same op sequence as the dot-product loop.
        "  ::aie::accum<accfloat,16> acc;\n"
        "  acc.from_vector(::aie::zeros<float,16>());\n"
        "  #pragma clang loop unroll(disable)\n"
        f"  for (int i = 0; i < {N}; i++) {{\n"
        "    acc = ::aie::mac(acc, ::aie::load_v<16>(inp), ::aie::load_v<16>(inp));\n"
        "  }\n"
        "  alignas(64) float sink[16];\n"
        "  ::aie::store_v(sink, acc.to_vector<float>());\n"
        f"  route_b_bricks::softmax_core<16>(scores, out, 16);\n"
        # keep the throwaway work live without letting it reach the answer
        "  if (sink[0] == 1.0e30f) out[0] = 0.0f;\n"
        "  event1();\n}\n")
    cc = GEN / f"codevol_{N}.cc"
    cc.write_text(src)
    try:
        res = bricklib.verify_rowwise(name=f"codevol_{N}", brick_cc=cc, shim_body="",
                                      symbol=f"cv_{N}", m=ROWS, in_cols=COLS, out_cols=COLS,
                                      x=x, expected=ref, gate=GATE)
        got = np.asarray(res["got"], np.float64)
        nf = int((~np.isfinite(got)).sum())
        results[N] = res["rel_l2"]
        print(f"{N:>4} {res['rel_l2']:12.3e} {nf:>7}/{got.size}  {'PASS' if res['ok'] else 'FAIL'}")
    except Exception as ex:
        print(f"{N:>4} ERROR {type(ex).__name__}: {str(ex)[:90]}")

print()
ok = [n for n, r in results.items() if isinstance(r, float) and r < GATE]
if 0 in ok and len(ok) < len(results):
    print("VERDICT: REPRODUCED MINIMALLY -- softmax alone is green and degrades as the enclosing\n"
          "         function's throwaway vector work grows. This is filable as-is.")
elif len(ok) == len(results):
    print("VERDICT: all green -- throwaway MAC work does NOT reproduce it. The scaling needs\n"
          "         something prefill_attn has that this does not (bf16 operands, more live\n"
          "         vectors, or the resident-operand rail). Narrow from prefill_attn instead.")
else:
    print("VERDICT: N=0 is not green -- this probe is measuring something else; fix it before\n"
          "         reading anything into the sweep.")
