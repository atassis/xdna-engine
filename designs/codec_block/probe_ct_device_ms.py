#!/usr/bin/env python3
"""Measure conv_transpose's own ms-per-streamed-KiB, rather than inheriting conv-1d's.

stage_shapes.STREAM_MS_PER_KIB = 0.97 was measured on conv-1d and is deliberately NOT applied to
the upsample plans: conv_transpose is a different kernel (wider output tile, t*stride instead of t)
and its rate is its own. Borrowing conv-1d's number would set the TDR chunk cap from the wrong
slope -- too small and the dispatch count explodes ~16x for nothing, too large and the stage still
times out.

Same hook as probe_device_ms.py (CachedXRTRuntime.run), so the figure is the dispatch, not the
aiecc build. c_in is set to ci_chunk so each trial is exactly one chunk.
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
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
import stage_shapes as ss

CI, K, STRIDE = 64, 16, 8          # stage 1 upsample geometry, chunked
L = 8
rng = np.random.default_rng(7)
print(f"conv_transpose  ci_chunk={CI} k={K} stride={STRIDE} L={L}", flush=True)
print(f"{'c_out':>6} {'streamed_B':>11} {'device_ms':>10} {'ms/KiB':>8}  result", flush=True)
for c_out in (32, 64, 128, 256):
    x = rng.standard_normal((CI, L)).astype(np.float32)
    w = (rng.standard_normal((CI, c_out, K)) * 0.02).astype(np.float32)
    _last.clear()
    try:
        wd.conv_transpose(x, w, np.zeros(c_out, np.float32), STRIDE, f"ct{c_out}",
                          ci_chunk=CI, resident_depth=1)
        res = "OK"
    except Exception as e:
        res = "TIMEOUT" if "TIMEOUT" in str(e) else type(e).__name__
    sb = c_out * (CI * K + 1) * 4
    ms = _last.get("ms", float("nan"))
    print(f"{c_out:6d} {sb:11d} {ms:10.1f} {ms/(sb/1024):8.3f}  {res}", flush=True)
