#!/usr/bin/env python3
"""Is the half-filled output my verify_rowwise CALL, or something about the softmax kernel?

probe_codevol_sanity.py showed every other output row zero with the written rows bit-exact. That is
a coverage fault. This runs the SAME call shape with a trivial copy kernel -- no softmax, no include
-- so the only thing left is the harness invocation itself.

  copy half-fills too => my m/in_cols/out_cols usage is wrong; softmax was never involved.
  copy fills          => the coverage fault needs the softmax kernel present; that is a real finding.
"""
import time
import numpy as np
import bricklib

GEN = bricklib.GEN
ROWS, COLS, GATE = 32, 16, 3e-2
CB = int(time.time() * 1000) % 10**9
rng = np.random.default_rng(7)
x = rng.uniform(-8.0, 8.0, (ROWS, COLS)).astype(np.float32)

for tag, body in (
    ("copy_scalar", "  for (int j = 0; j < 16; j++) out[j] = inp[j];\n"),
    ("copy_vector", "  ::aie::store_v(out, ::aie::load_v<16>(inp));\n"),
):
    src = ("#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
           f"// cachebust {CB}\n"
           f'extern "C" void rc_{tag}(float *inp, float *out) {{\n  event0();\n'
           + body + "  event1();\n}\n")
    cc = GEN / f"rc_{tag}.cc"; cc.write_text(src)
    try:
        res = bricklib.verify_rowwise(name=f"rc_{tag}", brick_cc=cc, shim_body="",
                                      symbol=f"rc_{tag}", m=ROWS, in_cols=COLS, out_cols=COLS,
                                      x=x, expected=x, gate=GATE)
        got = np.asarray(res["got"], np.float64)
        fill = float((got != 0).mean())
        sums = np.round(np.abs(got[:4]).sum(axis=1), 3)
        print(f"{tag:12s} rel_l2={res['rel_l2']:.3e} fill={fill:.4f} "
              f"row-abs-sums(first4)={sums}  {'PASS' if res['ok'] else 'FAIL'}")
    except Exception as ex:
        print(f"{tag:12s} ERROR {type(ex).__name__}: {str(ex)[:80]}")
