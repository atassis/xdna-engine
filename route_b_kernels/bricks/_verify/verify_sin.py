#!/usr/bin/env python3
"""Device rel-L2 verify for the sin brick. Gate 3e-2. Run under the device lock.

Uses the ROWWISE rail, like verify_rmsnorm, not verify_oneshot. Measured reason: with oneshot the
kernel only ever receives the first 64 floats (256 B) of the buffer -- at N=256/512/1024 the first
wrong element is index 64 every time, and the values past it are UNBOUNDED (sin cannot exceed 1), so
the kernel is computing sin of whatever follows the delivered tile. The same kernel passes oneshot at
N=64 exactly (rel-L2 1.2e-4). Rowwise streams m rows of in_cols and is the rail the shipped norm
bricks verify on.

Two cache notes, both of which invalidated earlier runs of this file:
  * The JIT cache key hashes the SHIM text only, not the brick .cc it #includes, so editing sin.cc
    alone silently reuses a stale xclbin. Proven: a kernel edited to store the constant 42 still
    returned sin values. The `_cb` marker keeps the shim text unique per run.
  * `brick_cc` must be an ABSOLUTE path -- the shim compiles from the cache dir, so a relative
    include resolves to nothing.
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
