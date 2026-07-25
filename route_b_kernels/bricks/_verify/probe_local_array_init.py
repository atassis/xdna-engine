#!/usr/bin/env python3
"""Is a scalar-filled LOCAL float[64] fully written on an AIE2P core?

WHY THIS EXISTS. `probe_dequant_bisect2.py` showed the int4 dequant epilogue diverging at
`aie::mul(pf, sv)` -- but ONLY for column tiles ni>=2, on both mi rows:

    (0,0) ok  (0,1) ok  (0,2) BAD  (0,3) BAD
    (1,0) ok  (1,1) ok  (1,2) BAD  (1,3) BAD

That shape cannot be a value-path miscompile. In that shim the scale is a local
`float pScale[64]` filled with 1.0f and read as `sg = pScale + ni*16`, so the multiplier
vector is IDENTICAL for every ni. Identical inputs cannot give correct results for ni<2
and 0/NaN for ni>=2. The only thing tracking ni is WHICH 16-float slice of pScale is read
-- and ni>=2 is exactly its second half. Hypothesis:

    a scalar-filled local float[64] is only HALF written (first 32 floats).

This matters beyond the probe: the standing diagnosis in the int4-dequant-brick worklog is
"a Peano f32-epilogue codegen bug at a loop boundary", filed as an upstream llvm-aie
candidate. That was reached using the SAME `scale=1.0 baked into a local array` trick. If
local arrays are what's broken, the "epilogue miscompile" is a probe artifact and the
brick's real path (scale read from the L1 input buffer) was never at fault.

TWO EARLIER VERSIONS HUNG THE CORE (ERT_CMD_STATE_TIMEOUT) -- first a 32/64/128/256 sweep,
then a single `a[i] = (float)(i+1)` ramp. A known-good brick passed bit-exact immediately
after, so the device was fine; the hang was the probe. The ramp used a SCALAR int->float
conversion, which on aie2p is a known-bad family (the active Peano pin carries b892afe21
for an int->float -O2 delay-slot miscompile). So this version uses NO int->float at all:
it fills with a literal 1.0f, exactly as bisect2 does.

Two cases in one kernel, to separate WHICH loop truncates:
  A: local float[64] filled 1.0f, then copied out   -- array + copy
  B: out[i] = 1.0f written directly, no local array -- copy only (control)
Output starts as host zeros, so "0.0" means never written.

Run:  ./run.sh probe_local_array_init.py
"""
from pathlib import Path
import numpy as np

import aie.iron as iron
import bricklib

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

L = 64
OUT_N = 2 * L

sym = "local_array_init_probe"
shim = f'''#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void {sym}(const int8_t* __restrict unused, float* __restrict out) {{
  (void)unused;
  // A: via a local array (the bisect2 / brick-probe shape)
  float a[{L}];
  for (int i = 0; i < {L}; ++i) a[i] = 1.0f;
  for (int i = 0; i < {L}; ++i) out[i] = a[i];
  // B: control -- same store loop, no local array
  for (int i = 0; i < {L}; ++i) out[{L} + i] = 1.0f;
}}'''
(GEN / f"{sym}_shim.cc").write_text(shim)

design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc",
                                 [64], OUT_N, [np.int8], np.float32, [])
u_t = iron.tensor(np.zeros(64, np.int8), dtype=np.int8, device="npu")
o_t = iron.zeros((OUT_N,), dtype=np.float32, device="npu")
design(u_t, o_t)
got = np.array(o_t.numpy().copy(), copy=True)

a_got, b_got = got[:L], got[L:]
a_ok, b_ok = int((a_got == 1.0).sum()), int((b_got == 1.0).sum())

print(f"[local_array_init] local float[{L}] filled with literal 1.0f\n")
print(f"  A via local array : {a_ok}/{L} correct")
print(f"     A[28:36] = {a_got[28:36]}")
print(f"     A[60:64] = {a_got[60:64]}")
print(f"  B direct store    : {b_ok}/{L} correct   (control)")
print(f"     B[28:36] = {b_got[28:36]}")

if a_ok == L and b_ok == L:
    verdict = ("both fully written -- hypothesis REFUTED; the bisect2 ni>=2 divergence "
               "needs another explanation")
elif a_ok < L and b_ok == L:
    verdict = (f"LOCAL ARRAY truncated at index {int(np.argmax(a_got != 1.0))} while the "
               "direct-store control is complete -- the local array IS the defect, and the "
               "'f32-epilogue miscompile' is a local-array artifact")
else:
    verdict = (f"the STORE LOOP itself is truncated (A={a_ok}/{L}, B={b_ok}/{L}) -- not "
               "specific to the local array")
print(f"\n  VERDICT: {verdict}")
