#!/usr/bin/env python3
"""Is gelu-erf's device failure NUMERICS, or the streamed rail?

verify_gelu_erf.py gates 64 rows x 64 cols through verify_rowwise, i.e. 64 SEPARATE kernel calls
on the streamed rail, and fails at rel-L2 7.04e-01. But its row-0 landmarks (x = -10 .. +10) all
come back near-correct, and one bad element cannot produce that norm: max-abs is 12.0 against an
expected-vector norm of ~1e2, so an error norm of ~90 needs MANY wrong elements. Row 0 right and
the rest wrong is exactly the signature verify_sin.py records for sin:

    "On the ONESHOT rail the same kernel passes to 1024. The defect needs the streamed rail,
     where the kernel is invoked once per tile -- the first invocation is right and later ones
     are not."

So run the SAME kernel, SAME inputs, on both rails and compare:

  * oneshot  -- one call, whole 4096-element buffer, no per-tile re-invocation
  * rowwise  -- 64 calls of 64 elements, the streamed rail verify_gelu_erf.py uses

If oneshot is green and rowwise is red, the kernel's arithmetic is fine and gelu-erf is blocked on
the same unresolved rail defect as sin -- which is a very different (and much more valuable)
finding than "the erf approximation is inaccurate", and it means no amount of polynomial work
fixes it. If BOTH are red, the arithmetic really is wrong and the tail hunt continues.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = (HERE.parent / "gelu-erf").resolve()
# 1024 elements, NOT 4096: the oneshot rail stages the whole buffer into L1, and
# 4096 f32 in + 4096 f32 out at objectFIFO depth 2 is 64 KB, which aiecc refuses
# ('Basic sequential allocation also failed'). Both rails cover the SAME 1024
# elements so the only difference is 1 call vs 16 calls.
M, COLS = 16, 64
N = M * COLS
GATE = 3e-2

spec = importlib.util.spec_from_file_location("gelu_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

# Same input construction as verify_gelu_erf.py, same seed, so the two are comparable.
rng = np.random.default_rng(0)
tail_neg = np.linspace(-12.0, -4.0, 512, dtype=np.float32)
tail_pos = np.linspace(4.0, 12.0, 512, dtype=np.float32)
bend = np.linspace(-3.0, 3.0, 2048, dtype=np.float32)
x_flat = np.concatenate([tail_neg, tail_pos, bend]).astype(np.float32)
rng.shuffle(x_flat)
x_flat = x_flat[:N]   # keeps tails, bend and the x=12.0 point in play
x = x_flat.reshape(M, COLS)
ref = golden.gelu_erf_ref(x)

_cb = int(time.time() * 1000) % 10**9

# --- rail A: ONESHOT. one call, all 4096 elements ---
one = bricklib.verify_oneshot(
    name="gelu_erf_oneshot",
    brick_cc=BRICK / "gelu_erf.cc",
    shim_body=(f"// cachebust {_cb}\n"
               f'extern "C" void gelu_one(float *x, float *out) {{ gelu_erf_f32(x, out, {N}); }}\n'),
    symbol="gelu_one",
    inputs=[(x.reshape(-1), np.float32)],
    out_numel=N, out_shape=(M, COLS),
    unpack=lambda f: np.asarray(f, np.float32).reshape(M, COLS),
    golden=ref, gate=GATE, out_dt=np.float32,
)

# --- rail B: ROWWISE. 64 calls of 64 elements, what verify_gelu_erf.py does ---
row = bricklib.verify_rowwise(
    name="gelu_erf_rowwise",
    brick_cc=BRICK / "gelu_erf.cc",
    shim_body=(f"// cachebust {_cb + 1}\n"
               f'extern "C" void gelu_row(float *x, float *out) {{ gelu_erf_f32(x, out, {COLS}); }}\n'),
    symbol="gelu_row",
    m=M, in_cols=COLS, out_cols=COLS,
    x=x, expected=ref, gate=GATE,
)

print()
print(f"  oneshot rel-L2 {one['rel_l2']:.3e}   rowwise rel-L2 {row['rel_l2']:.3e}")

# Per-row error tells us WHICH invocations are wrong, which is the whole question.
for tag, res in (("oneshot", one), ("rowwise", row)):
    got = np.asarray(res["got"], np.float64).reshape(M, COLS)
    per_row = np.linalg.norm(got - ref.astype(np.float64), axis=1)
    bad = int((per_row > 1e-2).sum())
    print(f"  {tag:8s} rows with err>1e-2: {bad:3d}/{M}   row0={per_row[0]:.3e} "
          f"row1={per_row[1]:.3e} row2={per_row[2]:.3e} rowlast={per_row[-1]:.3e}")

print()
if one["ok"] and not row["ok"]:
    print("VERDICT: arithmetic is FINE. gelu-erf is blocked on the STREAMED RAIL, same as sin.")
elif not one["ok"] and not row["ok"]:
    print("VERDICT: both rails red -- the arithmetic really is wrong, keep hunting the tail.")
else:
    print("VERDICT: rowwise green -- re-read, this contradicts verify_gelu_erf.py.")
