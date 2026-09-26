#!/usr/bin/env python3
"""conv2d-3x3-u8 kernel rate on ONE core, as a fitted slope.

Run with BRICK_JIT_CACHE=1, or every timed call recompiles through aiecc and the timer measures
the compiler (bricklib warns). The shim runs the kernel REPS times per streamed row, and wall time is fitted against REPS at a
fixed row count, so dispatch, DMA and host sync cancel and the slope is kernel time only.
Reports cycles per output pixel and MAC/cycle at an assumed clock, which is printed, because the
clock is a DPM variable. Prints numbers and exits 0; this is a probe, not a gate.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import aie.iron as iron
import bricklib

BRICK = Path(__file__).parent.parent / "conv2d-3x3-u8"
_spec = importlib.util.spec_from_file_location("conv3x3_golden", BRICK / "golden.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

CIN, COUT, W, ROWS, SHIFT = 64, 16, 64, 64, 11
CLOCK_HZ = 1.8e9  # assumption for the MAC/cycle column only; cycles/px is also clock-bound
REPS = [1, 9, 17, 25, 33]
TRIALS = 7


def build(reps):
    shim = bricklib.GEN / f"c3rate_r{reps}_shim.cc"
    sym = f"conv3x3_rate_r{reps}"
    shim.write_text(
        f'#include <stdint.h>\n#include "{BRICK / "conv3x3_u8.cc"}"\n'
        f'extern "C" void {sym}(uint8_t *t, int8_t *p, uint8_t *o) {{\n'
        f'  for (int r = 0; r < {reps}; r++)\n'
        f'    conv3x3_u8(t, t + {(W + 16) * CIN}, t + {2 * (W + 16) * CIN}, p, o, {W}, 1, {SHIFT}, 0, 0, {W});\n'
        f'}}\n')
    return bricklib._build_streamed(sym, shim, ROWS, 3 * (W + 16) * CIN, (W + 16) * COUT, len(params),
                                    [f"-DCONV3X3_CIN={CIN}", f"-DCONV3X3_COUT={COUT}"],
                                    np.uint8, np.uint8, np.int8,
                                    stack_size=2048)  # >= aiecc's measured need for every REPS variant (1088-1152 seen); aiecc re-checks


rng = np.random.default_rng(0)
tiles = rng.integers(0, 256, size=(ROWS, 3 * (W + 16) * CIN), dtype=np.int64).astype(np.uint8)
w = rng.integers(-127, 128, size=(COUT, CIN, 3, 3), dtype=np.int64).astype(np.int8)
b = rng.integers(-(1 << 15), 1 << 15, size=(COUT,), dtype=np.int64).astype(np.int32)
params = g.pack_params(w, b)

xs, ys = [], []
for reps in REPS:
    design = build(reps)
    in_t = iron.tensor(tiles.reshape(-1), dtype=np.uint8, device="npu")
    p_t = iron.tensor(params, dtype=np.int8, device="npu")
    out_t = iron.zeros((ROWS * (W + 16) * COUT,), dtype=np.uint8, device="npu")
    design(in_t, p_t, out_t)  # warm: load + first run
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        design(in_t, p_t, out_t)
        ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    print(f"reps={reps:3d}  median {med * 1e3:8.3f} ms  min {min(ts) * 1e3:8.3f} ms")
    xs.append(reps)
    ys.append(med)

slope, icpt = np.polyfit(xs, ys, 1)
resid = np.array(ys) - (slope * np.array(xs) + icpt)
px = ROWS * W
macs_px = 9 * CIN * COUT
cyc_px = slope / px * CLOCK_HZ
print(f"\nslope {slope * 1e6:.1f} us per rep over {px} px ({ROWS} rows x {W}); "
      f"intercept {icpt * 1e3:.3f} ms; max residual {np.abs(resid).max() * 1e6:.1f} us")
print(f"kernel: {slope / px * 1e9:.2f} ns/px = {cyc_px:.1f} cyc/px at {CLOCK_HZ / 1e9:.2f} GHz, "
      f"{macs_px} MAC/px -> {macs_px / cyc_px:.1f} MAC/cyc ({100 * macs_px / cyc_px / 512:.1f}% of 512)")
