#!/usr/bin/env python3
"""Separate the two candidate mechanisms behind window_driver.conv's ERT timeout.

Wall time cannot do it: a `wd.conv` call is dominated by the aiecc BUILD, so the earlier probe's
4.8s->8.2s trend measured compilation, not the dispatch. This hooks CachedXRTRuntime.run, which
wraps `h.wait()` directly, so what it reports is the DEVICE dispatch and nothing else.

  time hypothesis: amdxdna's tdr_timeout_ms defaults to 2000 (aie2_ctx.c:31) and kills a job with no
                   progress inside that window. Predicts device ms rising toward ~2000 as the
                   operand grows, with the last passing point just under it.
  size hypothesis: a hard limit near ~2.08 MB of streamed operand. Predicts device ms stays small
                   (tens of ms) and flat right up to a cliff.

The two are numerically adjacent by coincidence (~2 MB vs 2000 ms), which is exactly why wall time
looked like it supported either.
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

CI, K, L = 128, 7, 20
rng = np.random.default_rng(4)
print(f"{'c_out':>6} {'streamed_B':>11} {'device_ms':>10}  result", flush=True)
for c_out in (64, 256, 512, 576, 584, 640, 1024):
    x = rng.standard_normal((CI, L)).astype(np.float32)
    w = (rng.standard_normal((c_out, CI, K)) * 0.02).astype(np.float32)
    _last.clear()
    try:
        wd.conv(x, w, np.zeros(c_out, np.float32), K, 1, f"dm{c_out}",
                ci_chunk=CI, resident_depth=1)
        res = "OK"
    except Exception as e:
        res = "TIMEOUT" if "TIMEOUT" in str(e) else type(e).__name__
    print(f"{c_out:6d} {c_out*(CI*K+1)*4:11d} {_last.get('ms', float('nan')):10.1f}  {res}",
          flush=True)
