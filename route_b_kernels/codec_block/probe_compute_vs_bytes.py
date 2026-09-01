#!/usr/bin/env python3
"""Compute-bound or byte-bound? Vary the WORK at constant streamed BYTES.

probe_tile_granularity.py refuted per-transaction overhead: 4x fewer, 4x bigger objectFIFO tiles
carrying the same total bytes took the same 65.8 ms. But it could not separate the two remaining
explanations, and I initially misread it as if it could. At fixed window length t, the MAC count is
proportional to the streamed weight bytes (every weight element feeds t MACs), so "time scales with
bytes" is exactly what a COMPUTE-bound kernel looks like too.

This varies t instead. Weight bytes are held constant; MACs scale linearly with t.

    time proportional to t  -> COMPUTE-bound: the core is the limit, more/faster cores is the lever
    time flat in t          -> BYTE-bound: delivery is the limit, formats and residency are the lever

t only goes DOWN from 64: the resident activation is [c_in, t] f32 and 128x64x4 = 32 KB already,
so t=128 would not fit L1 at this c_in.
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

C_IN, K, C_OUT = 128, 7, 384
ROW = C_IN * K + 1
rng = np.random.default_rng(41)
print(f"c_in={C_IN} k={K} c_out={C_OUT}  weight bytes CONSTANT at {C_OUT*ROW*4} B", flush=True)
print(f"{'t':>4} {'MACs(M)':>9} {'device_ms':>10} {'ms/MAC(n)':>10} {'MAC/cyc':>8}", flush=True)

for t in (8, 16, 32, 64):
    sym = f"cvb_t{t}"
    shim = bricklib.GEN / f"cvb_t{t}_shim.cc"
    shim.write_text(
        f"#include <stdint.h>\n"
        f'#include "{wd.CONV_CC}"\n'
        f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
        f"  route_b_bricks::conv_1d_causal_core_vec<16>(resident, tile, tile[{C_IN*K}],\n"
        f"      out, {C_IN}, {K}, {t}, 1);\n}}\n")
    tiles = (rng.standard_normal((C_OUT, ROW)) * 0.02).astype(np.float32)
    win = rng.standard_normal((C_IN, t)).astype(np.float32)
    _dev["ms"] = 0.0
    try:
        design = bricklib._build_streamed(sym, shim, C_OUT, ROW, t, C_IN * t, None,
                                          np.float32, np.float32, np.float32, 1)
        in_t = bricklib.iron.tensor(tiles.reshape(-1), dtype=np.float32, device="npu")
        r_t = bricklib.iron.tensor(win.reshape(-1), dtype=np.float32, device="npu")
        out_t = bricklib.iron.zeros((C_OUT * t,), dtype=np.float32, device="npu")
        design(in_t, r_t, out_t); _dev["ms"] = 0.0
        design(in_t, r_t, out_t)
        got = out_t.numpy().copy()
        macs = C_OUT * C_IN * K * t
        ms = _dev["ms"]
        print(f"{t:4d} {macs/1e6:9.2f} {ms:10.1f} {ms*1e6/macs:10.3f} {macs/(ms/1e3)/1.8e9:8.4f}"
              f"   nz={np.abs(got).sum():.3e}", flush=True)
    except Exception as e:
        print(f"{t:4d}  FAILED: {type(e).__name__}: {str(e)[:70]}", flush=True)
