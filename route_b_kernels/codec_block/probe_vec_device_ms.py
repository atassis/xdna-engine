#!/usr/bin/env python3
"""THE attribution experiment: is conv-1d's per-byte cost core COMPUTE or per-tile OVERHEAD?

probe_device_ms.py measured the scalar core at 0.97 ms/KiB = 1.03 MB/s = 0.00200 MAC/cycle, and the
TDR watchdog is reached at ~2 MB of streamed operand purely because of that rate. The obvious
reading is "the kernel is scalar, vectorise it". But a competently scheduled scalar FMA loop should
cost ~5-10 cycles/element, and the measured 3488 us per tile is ~30-60x even THAT -- which says the
time may be dominated by something outside the core (objectFIFO handshake, DMA, lock waits), in
which case cutting the core's instruction count buys little.

The vector core settles it, and nothing cheaper does. Identical tile geometry, identical rail, same
CachedXRTRuntime.run hook, only the core swapped:

  compute-bound  -> ms/KiB drops toward the instruction-count ratio (~14-16x fewer MAC-class ops).
  overhead-bound -> ms/KiB barely moves, and vectorising is the WRONG lever for the dispatch count.

Correctness is already settled separately (verify_conv_1d.py: vec matches scalar at 4.483e-07 /
4.253e-07 / 2.860e-07 for dilation 1/3/9, and 1.867e-07 at k=1). This probe only times.
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aie.utils.hostruntime.xrtruntime import hostruntime as hr

_last = {}
_orig = hr.CachedXRTRuntime.run
def _timed(self, *a, **k):
    t0 = time.perf_counter()
    try:
        return _orig(self, *a, **k)
    finally:
        _last["ms"] = (time.perf_counter() - t0) * 1e3
hr.CachedXRTRuntime.run = _timed

import window_driver as wd
import bricklib

CI, K, T, DIL = 128, 7, wd.T, 1
CONV_CC = wd.CONV_CC
rng = np.random.default_rng(11)


def run_core(core, c_out, tag):
    """One dispatch streaming c_out weight rows past a resident [CI, T] activation."""
    wide = CI * K + 1
    sym = f"probe_{tag}"
    shim = bricklib.GEN / f"probe_{tag}_shim.cc"
    shim.write_text(
        f"#include <stdint.h>\n"
        f'#include "{CONV_CC}"\n'
        f'extern "C" void {sym}(float *tile, float *resident, float *out) {{\n'
        f"  route_b_bricks::{core}(resident, tile, tile[{CI * K}], out,\n"
        f"                         {CI}, {K}, {T}, {DIL});\n}}\n")
    tiles = (rng.standard_normal((c_out, wide)) * 0.02).astype(np.float32)
    win = rng.standard_normal((CI, T)).astype(np.float32)
    _last.clear()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        bricklib.verify_streamed(
            name=tag, shim=shim, symbol=sym, in_tiles=tiles, out_tile_numel=T,
            resident=win.reshape(-1).astype(np.float32),
            unpack=lambda d: np.asarray(d), golden=np.zeros((c_out, T)), gate=np.inf,
            in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32, resident_depth=1)
    return _last.get("ms", float("nan")), c_out * wide * 4


print(f"CI={CI} K={K} T={T} dilation={DIL}   one dispatch per row, c_out tiles streamed", flush=True)
print(f"{'core':>10} {'c_out':>6} {'streamed_B':>11} {'device_ms':>10} {'ms/KiB':>8}", flush=True)
for core, short in (("conv_1d_causal_core", "scalar"), ("conv_1d_causal_core_vec<16>", "vec")):
    for c_out in (128, 384):
        tag = f"{short}_{c_out}"
        ms, sb = run_core(core, c_out, tag)
        print(f"{short:>10} {c_out:6d} {sb:11d} {ms:10.1f} {ms/(sb/1024):8.3f}", flush=True)
