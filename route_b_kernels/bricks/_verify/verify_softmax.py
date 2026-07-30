#!/usr/bin/env python3
"""Device rel-L2 verify for the softmax brick. Gate 3e-2. Run under the device lock.

Rowwise, 64 floats per call: matches sin.cc's OWN measured ceiling ("exact at 64 floats per call
and wrong above that due to spill"), and softmax's exp2_v poly is heavier per chunk than sin_v
(round-trick + int32 detour + 7-term Horner vs sin_v's 6-term odd-only Horner) -- so 64 is not a
number to exceed here without new device evidence. See softmax.cc's header for the full reasoning,
including the c=1024 vs T<=65 discrepancy this brick's own oracle module docstring documents.

Rows deliberately cover the four cases that break a naive (no max-subtract, no underflow-guard)
implementation -- a gate that only sees plain random rows would pass a kernel broken on the real
input:
  (i)   plain random rows
  (ii)  rows with -1e9 masked entries (the real _causal_window_mask/rvq_transformer input)
  (iii) rows of large positive values (naive exp(x) overflows without the max-subtract)
  (iv)  a row of identical values (output must be exactly uniform 1/cols)
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = (HERE.parent / "softmax").resolve()

ROWS_PER_CASE = 8
COLS = 64
GATE = 3e-2

spec = importlib.util.spec_from_file_location("softmax_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(0)

# (i) plain random rows
x_random = (rng.standard_normal((ROWS_PER_CASE, COLS)).astype(np.float32) * 4.0)

# (ii) rows with -1e9 masked entries (>=1 unmasked entry per row, the real causal-mask invariant)
x_masked = (rng.standard_normal((ROWS_PER_CASE, COLS)).astype(np.float32) * 4.0)
mask = rng.random((ROWS_PER_CASE, COLS)) < 0.6
mask[:, 0] = False
x_masked[mask] = -1e9

# (iii) rows of large positive values -- naive exp(x) (no max-subtract) overflows in f32 here
x_large_pos = (rng.standard_normal((ROWS_PER_CASE, COLS)).astype(np.float32) * 5.0 + 80.0)

# (iv) rows of identical values -- output must be exactly uniform 1/COLS
x_uniform = np.full((ROWS_PER_CASE, COLS), 3.0, dtype=np.float32)

x = np.concatenate([x_random, x_masked, x_large_pos, x_uniform], axis=0)
M = x.shape[0]
ref = golden.softmax_ref(x)

_cb = int(time.time() * 1000) % 10**9
res = bricklib.verify_rowwise(
    name="softmax",
    brick_cc=BRICK / "softmax.cc",
    shim_body=(
        f"// cachebust {_cb}\n"
        f'extern "C" void softmax_verify(float *x, float *out) {{ softmax_f32(x, out, {COLS}); }}\n'
    ),
    symbol="softmax_verify",
    m=M, in_cols=COLS, out_cols=COLS,
    x=x, expected=ref, gate=GATE,
)

got = np.asarray(res["got"], np.float64)
row_sums = got.sum(axis=-1)
print(f"  device vs ref              rel-L2: {res['rel_l2']:.3e}")
print(f"  sum(out) per row (expect 1): min={row_sums.min():.6f} max={row_sums.max():.6f}")
print(f"  invariant sum(out)==1      : {bool(np.allclose(row_sums, 1.0, atol=3e-2))}")

# per-case breakdown so a failure on one case doesn't hide inside the aggregate rel-L2
bounds = [0, ROWS_PER_CASE, 2 * ROWS_PER_CASE, 3 * ROWS_PER_CASE, 4 * ROWS_PER_CASE]
names = ["random", "masked(-1e9)", "large_positive", "uniform"]
for i, name in enumerate(names):
    lo, hi = bounds[i], bounds[i + 1]
    case_rl2 = golden.rel_l2(got[lo:hi], ref[lo:hi])
    print(f"  case {name:16s} rel-L2: {case_rl2:.3e}")

# case (iv) uniform rows: check the on-device value is close to the exact 1/COLS constant
uniform_got = got[bounds[3]:bounds[4]]
print(f"  uniform row value (expect {1.0/COLS:.6f}): got[0,0]={uniform_got[0, 0]:.6f}")

assert res["ok"], f"softmax device gate failed: {res['status']} rel_l2={res['rel_l2']}"
assert np.allclose(row_sums, 1.0, atol=3e-2), f"sum(out) invariant failed: {row_sums}"
print("PASS")
