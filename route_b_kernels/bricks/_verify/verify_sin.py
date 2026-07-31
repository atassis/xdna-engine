#!/usr/bin/env python3
"""Device rel-L2 verify for the sin brick. Gate 3e-2. Run under the device lock.

GREEN as of 2026-07-31: **rel-L2 1.275e-04**, three consecutive fresh builds, run2run 0, at 256x16
over x in [-64, 64] (~10 periods, so the argument fold is genuinely exercised).

THE FIX WAS THE LOOP, NOT THE MATHS. `sin_core` used to derive `chunks = n / N` from a runtime
`int32_t n`; it now takes ONE 16-wide vector per call, with no element count and no loop, and volume
comes from the harness's tile loop. Same change that took `gelu-erf` from 9.538e+00 to 1.138e-03.
See kb/kernel-internal-loops-miscompile-put-volume-in-the-worker.

TWO CLAIMS THIS DOCSTRING USED TO MAKE, BOTH NOW FALSIFIED -- kept so they are not re-derived:

  * "The 1.275e-04 was a STALE-CACHE ARTIFACT, not a measurement." **Wrong.** The brick now measures
    exactly 1.275e-04 under the content-hash-keyed harness across three fresh builds. That number is
    simply what a correct sin brick produces here; host simulation of this kernel with a correct fold
    independently predicts 1.2747e-04. Whatever the old cache did, the value was not fabricated.
  * "The defect needs the STREAMED rail -- the first invocation is right and later ones are not."
    **Wrong.** Measured: it fails on the ONESHOT rail too, and it fails at ONE chunk per call
    (7.114e-01 at n=16). Also ruled out by measurement: input range (|x|<=12 gives 7.035e-01, same as
    |x|<=64) and the fold constant (2^23 vs 1.5*2^23 are BIT-IDENTICAL at one chunk).

snake stays green at 5.143e-06 across this change -- it composes `sin_v` directly and never used
`sin_core`, so its own loop is untouched. conv_transpose_1d 3.894e-08.

`brick_cc` must be an ABSOLUTE path -- the shim compiles from the cache dir, so a relative include
resolves to nothing.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = (HERE.parent / "sin").resolve()

M, COLS = 256, 16          # 4096 elements. 64 per CALL is required by this kernel: a copy kernel
                          # delivers bit-exact tiles at 1024 (probe_tile_limit.py), so this is NOT a
                          # delivery limit -- sin specifically goes wrong above 64/call with ALIASED,
                          # unbounded output. Disabling loop unrolling did not change it, so the
                          # mechanism is still unidentified. Volume comes from the row count.
GATE = 3e-2

spec = importlib.util.spec_from_file_location("sin_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(0)
# +/-64 spans about 10 periods so the argument fold is exercised, not just the polynomial.
x = (rng.random((M, COLS), dtype=np.float32) * 128.0 - 64.0).astype(np.float32)
ref = golden.sin_ref(x)

_cb = int(time.time() * 1000) % 10**9

res = bricklib.verify_rowwise(
    name="sin",
    brick_cc=BRICK / "sin.cc",
    shim_body=(
        f"// cachebust {_cb}\n"
        f'extern "C" void sin_verify(float *x, float *out) {{ sin_f32(x, out); }}\n'
    ),
    symbol="sin_verify",
    m=M, in_cols=COLS, out_cols=COLS,
    x=x, expected=ref, gate=GATE,
)

got = np.asarray(res["got"], np.float32)
print(f"  device vs np.sin     rel-L2: {res['rel_l2']:.3e}")
print(f"  device vs emulation  rel-L2: {golden.rel_l2(got, golden.sin_poly_emulated(x)):.3e}")
print(f"  max abs err vs np.sin      : {float(np.max(np.abs(got - ref))):.3e}")
print(f"  out range                  : [{float(got.min()):+.4f}, {float(got.max()):+.4f}] (must be in [-1,1])")
assert res["ok"], f"sin device gate failed: {res['status']} rel_l2={res['rel_l2']}"
print("PASS")
