#!/usr/bin/env python3
"""How much of a dispatch is per-objectFIFO-TRANSACTION overhead, and does bigger amortise it?

Measured: one vectorised conv dispatch streams 384 tiles of 3588 B and takes 65.6 ms, against
0.76 ms if compute-bound and 0.026 ms if DRAM-bound. That points at ~0.17 ms of fixed cost per
tile. If that reading is right, holding the TOTAL bytes fixed while making tiles bigger (and fewer)
must cut the time roughly in proportion to the tile COUNT, and the ms/tile should stay flat.

That is the whole experiment: sweep tile size at constant total bytes. A flat ms-per-tile confirms
per-transaction overhead and sizes the fix; a flat ms-per-BYTE would refute it and mean the cost is
per byte after all.

The kernel is unchanged -- only how many f32 ride in one objectFIFO tile changes. The shim treats
each tile as `reps` consecutive weight rows and loops over them, a COMPILE-TIME bound (the shipped
`snake` and `softmax` both carry compile-time internal loops and are green; a RUNTIME bound is the
recorded hazard).
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aie.utils.hostruntime.xrtruntime import hostruntime as hr

_dev = {"ms": 0.0}
_orig = hr.CachedXRTRuntime.run
def _timed(self, *a, **k):
    t0 = time.perf_counter()
    try:
        return _orig(self, *a, **k)
    finally:
        _dev["ms"] += (time.perf_counter() - t0) * 1e3
hr.CachedXRTRuntime.run = _timed

import window_driver as wd
import bricklib

C_IN, K, T = 128, 7, wd.T
ROW = C_IN * K + 1                 # 897 f32 = 3588 B, one output channel's weights + bias
C_OUT_TOTAL = 384                  # held FIXED: same total bytes every trial
rng = np.random.default_rng(31)

print(f"c_in={C_IN} k={K} T={T}  total rows={C_OUT_TOTAL} (constant {C_OUT_TOTAL*ROW*4} B streamed)",
      flush=True)
print(f"{'reps':>5} {'tiles':>6} {'tile_B':>8} {'device_ms':>10} {'ms/tile':>9} {'ms/MB':>8}", flush=True)

for reps in (1, 2, 4, 8, 16):
    n_tiles = C_OUT_TOTAL // reps
    tile_floats = ROW * reps
    sym = f"tg_r{reps}"
    shim = bricklib.GEN / f"tg_r{reps}_shim.cc"
    # one call handles `reps` output channels; the loop bound is a literal, not a runtime value
    shim.write_text(
        f"#include <stdint.h>\n"
        f'#include "{wd.CONV_CC}"\n'
        f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
        f"  for (int r = 0; r < {reps}; ++r)\n"
        f"    route_b_bricks::conv_1d_causal_core_vec<16>(resident, tile + r*{ROW},\n"
        f"        tile[r*{ROW} + {C_IN*K}], out + r*{T}, {C_IN}, {K}, {T}, 1);\n}}\n")

    tiles = (rng.standard_normal((n_tiles, tile_floats)) * 0.02).astype(np.float32)
    win = rng.standard_normal((C_IN, T)).astype(np.float32)
    _dev["ms"] = 0.0
    try:
        design = bricklib._build_streamed(sym, shim, n_tiles, tile_floats, T * reps, C_IN * T,
                                          None, np.float32, np.float32, np.float32, 1)
        in_t = bricklib.iron.tensor(tiles.reshape(-1), dtype=np.float32, device="npu")
        r_t = bricklib.iron.tensor(win.reshape(-1), dtype=np.float32, device="npu")
        out_t = bricklib.iron.zeros((n_tiles * T * reps,), dtype=np.float32, device="npu")
        design(in_t, r_t, out_t)                       # warm the build
        _dev["ms"] = 0.0
        design(in_t, r_t, out_t)                       # the measured dispatch
        got = out_t.numpy().copy()
        ms = _dev["ms"]
        mb = C_OUT_TOTAL * ROW * 4 / 1e6
        print(f"{reps:5d} {n_tiles:6d} {tile_floats*4:8d} {ms:10.1f} {ms/n_tiles:9.3f} {ms/mb:8.1f}"
              f"   nz={np.abs(got).sum():.3e}", flush=True)
    except Exception as e:
        print(f"{reps:5d} {n_tiles:6d} {tile_floats*4:8d}  FAILED: {type(e).__name__}: {str(e)[:70]}",
              flush=True)
