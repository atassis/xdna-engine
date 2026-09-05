#!/usr/bin/env python3
"""Device rel-L2 verify for the snake brick. Gate 3e-2. Run under the device lock.

Rowwise, 64 floats per call: the sin brick it composes is exact at 64 and reads past the tile for
larger single calls. One alpha per row would need a per-row scalar, which the pure-buffer shim ABI
does not carry, so this gates ONE alpha across all rows and the per-channel sweep is covered by the
host golden (snake_emulated tracks snake_ref to 5.4e-06 across 8 distinct alphas).
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = (HERE.parent / "snake").resolve()

M, COLS = 64, 64
ALPHA = 1.75
GATE = 3e-2

spec = importlib.util.spec_from_file_location("snake_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(0)
x = (rng.standard_normal((M, COLS)).astype(np.float32) * 3.0)
alpha = np.full(M, ALPHA, dtype=np.float32)
ref = golden.snake_ref(x, alpha)

_cb = int(time.time() * 1000) % 10**9
res = bricklib.verify_rowwise(
    name="snake",
    brick_cc=BRICK / "snake.cc",
    shim_body=(
        f"// cachebust {_cb}\n"
        f'extern "C" void snake_verify(float *x, float *out) {{ snake_f32(x, out, {COLS}, {ALPHA}f); }}\n'
    ),
    symbol="snake_verify",
    m=M, in_cols=COLS, out_cols=COLS,
    x=x, expected=ref, gate=GATE,
)

got = np.asarray(res["got"], np.float32)
print(f"  device vs ref        rel-L2: {res['rel_l2']:.3e}")
print(f"  device vs emulation  rel-L2: {golden.rel_l2(got, golden.snake_emulated(x, alpha)):.3e}")
print(f"  invariant y >= x           : {bool(np.all(got >= x - 1e-3))}")
assert res["ok"], f"snake device gate failed: {res['status']} rel_l2={res['rel_l2']}"
print("PASS")
