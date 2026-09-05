#!/usr/bin/env python3
"""Does a SCALAR f32 multiply give the wrong answer on aie2p, in isolation?

WHY THIS EXISTS. gatedeltanet's device output was NaN until one line changed: the state-write row
loop did `float bk = beta * k[i]` (a scalar float multiply on a scalar-loaded gate and a vector
element extract), and hoisting that product into the vector domain took the brick from rel_l2=nan
to 0.000e+00 BIT-EXACT. The two forms are the same f32 product, so the difference is codegen.

That is NOT enough to call it a miscompile, and this probe exists to stop exactly that inference.
The delay-slot fix `b892afe21` that was named as the root cause of the earlier rope-lut scalar-f32
collapse is ALREADY an ancestor of our Peano pin base (`git merge-base --is-ancestor b892afe21
706c8d9ea6b1` succeeds), so whatever bites gatedeltanet survives that fix and is not the same bug.
Asserting the mechanism from a full-pipeline symptom is how the `sliding_mul` bf16 "defect" got
recorded and then refuted by an isolated repro -- see sliding-mul-bf16-isolated-repro-refutes-defect.

So: two arms in ONE kernel, same inputs, same output buffer, differing only in where the multiply
happens. Both must equal the host f32 product.

  A: scalar  -- `float bk = beta * kf[i]` per lane, exactly gatedeltanet's old shape
  B: vector  -- `aie::mul(kf, broadcast(beta))`, exactly the fix

A wrong and B right => the scalar f32 path miscompiles here, and it is an upstream llvm-aie defect
worth a minimal repro plus an issue. Both right => the scalar form is fine in isolation, the
gatedeltanet NaN needed the surrounding loop/recurrence context, and the "scalar f32 is unsafe"
framing must NOT be promoted to a rule on this evidence.

k is read from a 64-BYTE-ALIGNED offset (16 floats in), not from `in + 1`: an unaligned float
load_v snaps to the aligned base on this backend and fabricates a convincing lane shift -- a probe
artifact this tree has already been burned by once.

Run:  ./run.sh probe_scalar_f32_mul.py
"""
from pathlib import Path

import numpy as np

import aie.iron as iron
import bricklib

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

DK = 32
PAD = 16          # 16 floats = 64 bytes, so `in + PAD` is vector-aligned
IN_N = PAD + DK
OUT_N = 2 * DK

sym = "scalar_f32_mul_probe"
shim = f'''#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void {sym}(const float* __restrict in, float* __restrict out) {{
  const float beta = in[0];
  ::aie::vector<float, {DK}> kf = ::aie::load_v<{DK}>(in + {PAD});

  // A: scalar multiply per lane, then broadcast -- gatedeltanet's old state-write shape
  for (unsigned i = 0; i < {DK}; ++i) {{
    float bk = beta * kf[i];
    out[i] = bk;
  }}

  // B: one vector multiply -- the form that made the brick bit-exact
  ::aie::vector<float, {DK}> bkv =
      ::aie::mul(kf, ::aie::broadcast<float, {DK}>(beta)).to_vector<float>();
  ::aie::store_v(out + {DK}, bkv);
}}'''
(GEN / f"{sym}_shim.cc").write_text(shim)

rng = np.random.default_rng(0)
x = np.zeros(IN_N, np.float32)
beta = np.float32(0.37109375)          # exact in bf16 and f32, so the golden is unambiguous
x[0] = beta
k = rng.standard_normal(DK).astype(np.float32)
x[PAD:] = k
ref = (beta * k).astype(np.float32)

design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc",
                                 [IN_N], OUT_N, [np.float32], np.float32, [])
i_t = iron.tensor(x, dtype=np.float32, device="npu")
o_t = iron.zeros((OUT_N,), dtype=np.float32, device="npu")
design(i_t, o_t)
got = np.array(o_t.numpy().copy(), copy=True)

a, b = got[:DK], got[DK:]
a_exact = int(np.sum(a == ref))
b_exact = int(np.sum(b == ref))
print(f"  beta={beta}")
print(f"  A scalar  bit-exact {a_exact}/{DK}   max|err|={np.max(np.abs(a - ref)):.3e}")
print(f"  B vector  bit-exact {b_exact}/{DK}   max|err|={np.max(np.abs(b - ref)):.3e}")
if a_exact < DK:
    print(f"  A first 8 got: {a[:8]}")
    print(f"  A first 8 ref: {ref[:8]}")

if a_exact == DK and b_exact == DK:
    print("VERDICT: both arms exact -- scalar f32 mul is NOT broken in isolation")
elif b_exact == DK and a_exact < DK:
    print("VERDICT: scalar arm WRONG, vector arm exact -- scalar f32 mul miscompiles")
else:
    print("VERDICT: inconclusive -- the vector arm is not exact either, fix the probe first")
