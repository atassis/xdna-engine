#!/usr/bin/env python3
"""Device rel-L2 verify for the sin brick. Gate 3e-2. Run under the device lock.

THIS GATE CURRENTLY FAILS, AND THAT IS THE CORRECT RESULT. rel-L2 0.709 at 64x64, run2run 0.

The 1.275e-04 this brick was recorded green at was a STALE-CACHE ARTIFACT, not a measurement. The
harness keyed its artifact cache on the design NAME, and this file reuses the symbol `sin_verify`
on every run, so once any build of that name succeeded every later run re-reported it no matter
what sin.cc said. bricklib._design_name now keys on the actual build inputs; re-run under the fixed
harness, the committed brick fails at its own recorded configuration. The `// cachebust` marker
below never helped -- it only ever changed the shim TEXT, which was not the key.

What is actually wrong is a codegen instability, not a size limit and not the loop bound:
  * A copy kernel is bit-exact on this same rail at 32x512 and 8x512 (probe_rail_512.py,
    rel-L2 exactly 0), so nothing is truncated in delivery.
  * On the ONESHOT rail the same kernel passes to 1024. The defect needs the streamed rail, where
    the kernel is invoked once per tile -- the first invocation is right and later ones are not.
  * Rewriting the trip count as a compile-time template parameter fixes it in one arrangement and
    breaks it in the next; adding or REMOVING unrelated functions from the translation unit flips
    the verdict, and always_inline made a green arrangement fail. That is the register-pressure
    aliasing sin.cc's own header warns about (llvm-aie #1155 territory), reached by this kernel's
    live set, not something to be fixed by shuffling source.

snake, which inlines sin_v into its own loop rather than calling this path, is genuinely green
(5.143e-06, reproduces exactly under the fixed harness), as is conv_transpose_1d (3.894e-08). The
codec upsample stage needs those two, so it is not blocked on this.

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

M, COLS = 64, 64          # 4096 elements. 64 per CALL is required by this kernel: a copy kernel
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
        f'extern "C" void sin_verify(float *x, float *out) {{ sin_f32(x, out, {COLS}); }}\n'
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
