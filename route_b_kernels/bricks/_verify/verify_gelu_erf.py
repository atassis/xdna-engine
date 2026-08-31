#!/usr/bin/env python3
"""Device rel-L2 verify for the gelu-erf brick. Gate 3e-2. Run under the device lock.

Elementwise unary op, no per-row scalar (unlike snake's alpha), so the input is free to be
constructed for coverage. Deliberately NOT plain N(0,1) noise: GELU saturates to 0 for x << 0 and
to x for x >> 0, and a kernel that is only right near 0 would still pass a noise-only gate. So the
input mixes wide tails (|x| up to 12), the asymmetric bending region [-3, 3] densely, and noise,
plus a fixed row of scalar landmarks spliced in so their device value is directly inspectable.

Per-call width 64: gelu_erf.cc's own conservative choice (mirrors sin.cc/snake.cc's proven-safe
width; see gelu_erf.cc header -- there is no device signal for this kernel yet).
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = (HERE.parent / "gelu-erf").resolve()

# COLS=16 is REQUIRED, not a tuning choice: the kernel now takes exactly one 16-wide
# vector per call (no element count, no internal loop), because a loop of ANY trip count
# miscompiles for this body. M carries the volume instead. See gelu_erf_core's header.
M, COLS = 256, 16
GATE = 3e-2

spec = importlib.util.spec_from_file_location("gelu_erf_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(0)
tail_neg = np.linspace(-12.0, -4.0, 512, dtype=np.float32)
tail_pos = np.linspace(4.0, 12.0, 512, dtype=np.float32)
bend = np.linspace(-3.0, 3.0, 2048, dtype=np.float32)
noise = (rng.standard_normal(M * COLS - tail_neg.size - tail_pos.size - bend.size).astype(np.float32) * 3.0)
x_flat = np.concatenate([tail_neg, tail_pos, bend, noise]).astype(np.float32)
rng.shuffle(x_flat)
assert x_flat.size == M * COLS
x = x_flat.reshape(M, COLS)

# Fixed landmark scalars in row 0 so their device value is directly inspectable below.
landmarks = np.array([-10.0, -6.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 6.0, 10.0], dtype=np.float32)
x[0, :landmarks.size] = landmarks

ref = golden.gelu_erf_ref(x)

_cb = int(time.time() * 1000) % 10**9
res = bricklib.verify_rowwise(
    name="gelu_erf",
    brick_cc=BRICK / "gelu_erf.cc",
    shim_body=(
        f"// cachebust {_cb}\n"
        f'extern "C" void gelu_erf_verify(float *x, float *out) '
        f'{{ gelu_erf_f32(x, out); }}\n'
    ),
    symbol="gelu_erf_verify",
    m=M, in_cols=COLS, out_cols=COLS,
    x=x, expected=ref, gate=GATE,
)

got = np.asarray(res["got"], np.float64)
exp64 = ref.astype(np.float64)
max_abs = float(np.max(np.abs(got - exp64)))
print(f"  device vs ref (A&S oracle) rel-L2: {res['rel_l2']:.3e}  max-abs: {max_abs:.3e}")

hip = golden.gelu_erf_hiprec(x)
hip_rl2 = golden.rel_l2(got, hip)
hip_max_abs = float(np.max(np.abs(got - hip)))
print(f"  device vs true erf (hiprec) rel-L2: {hip_rl2:.3e}  max-abs: {hip_max_abs:.3e}")

emu = golden.gelu_erf_emulated(x)
print(f"  device vs host emulation    rel-L2: {golden.rel_l2(got, emu):.3e}")

print("  landmark x -> golden / device:")
for i, xv in enumerate(landmarks):
    print(f"    x={xv:+7.2f}  golden={ref[0, i]:+10.6f}  device={got[0, i]:+10.6f}")

# Invariants: GELU(0) == 0, and GELU never exceeds its input by more than a small margin
# (0.5*x is the max positive overshoot as x -> -inf's opposite tail is bounded; the real
# invariant that is cheap to check everywhere is monotonic tail sign: very negative x should be
# near 0, not negative-blown-up, and very positive x should track x closely).
neg_tail_mask = x <= -8.0
pos_tail_mask = x >= 8.0
if np.any(neg_tail_mask):
    neg_err = np.abs(got[neg_tail_mask])
    j = int(np.argmax(neg_err))
    print(f"  neg tail (x<=-8) max |device|      : {float(neg_err[j]):.3e}  "
          f"at x={float(x[neg_tail_mask][j]):+.6f}  device={float(got[neg_tail_mask][j]):+.6e}")
if np.any(pos_tail_mask):
    # DIAGNOSTIC (per coordinator, kept across v1/v2/v3/v4 -- "it is doing its job"). argmax
    # |device - x| over x>=8, and the x value there. History: v1/v2 (A&S erf via exp2+reciprocal)
    # this named x=12 reading exactly 0.0 -- v2's own final invariant clamp had made that failure
    # undiagnosable. v3 (exp2/aie::inv removed, no final clamp) turned out to have the RIGHT
    # formula but a Horner chain long enough to spill on device (found via the coordinator's own
    # oneshot/rowwise probe, not this script -- this test distribution's aggregate rel-L2 alone did
    # not localize it). v4 shortens that chain (degree 9->5, fewer live vectors across the loop;
    # see gelu_erf.cc header). No final clamp here either, so this print still shows the RAW device
    # value with nothing hiding a wrong answer behind a plausible one.
    pos_err = np.abs(got[pos_tail_mask] - x[pos_tail_mask].astype(np.float64))
    j = int(np.argmax(pos_err))
    xj = float(x[pos_tail_mask][j])
    print(f"  pos tail (x>=8) max |device - x|    : {float(pos_err[j]):.3e}  "
          f"at x={xj:+.6f}  device={float(got[pos_tail_mask][j]):+.6e}  "
          f"golden={float(exp64[pos_tail_mask][j]):+.6e}")

assert res["ok"], f"gelu_erf device gate failed: {res['status']} rel_l2={res['rel_l2']}"
print("PASS")
