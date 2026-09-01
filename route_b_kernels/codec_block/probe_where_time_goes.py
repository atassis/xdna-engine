#!/usr/bin/env python3
"""Account for ONE window_driver dispatch, end to end. No theory, just a breakdown.

Two explanations have been offered for the codec decoder taking ~1.8 s of wall per dispatch while
the device does ~22 ms of work: per-window xclbin rebuilds, and host-side numpy. bricklib's own
header prices a rebuild at ~150 ms, which leaves ~1.6 s unexplained -- so at least one of those is
wrong, or something else dominates. This measures instead of arguing.

Times, per wd.conv call, with the JIT cache OFF (the shipped default) and ON:
  total   -- the whole wd.conv call
  device  -- inside CachedXRTRuntime.run, i.e. the dispatch itself
  jit     -- inside iron.jit(...) construction/build
  rest    -- total - device - jit, i.e. host numpy + shim writing + BO setup
"""
import os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aie.utils.hostruntime.xrtruntime import hostruntime as hr
from aie import iron

acc = {"device": 0.0, "jit": 0.0, "ndev": 0, "njit": 0}

_orig_run = hr.CachedXRTRuntime.run
def _timed_run(self, *a, **k):
    t0 = time.perf_counter()
    try:
        return _orig_run(self, *a, **k)
    finally:
        acc["device"] += time.perf_counter() - t0
        acc["ndev"] += 1
hr.CachedXRTRuntime.run = _timed_run

_orig_jit = iron.jit
def _timed_jit(*a, **k):
    t0 = time.perf_counter()
    try:
        return _orig_jit(*a, **k)
    finally:
        acc["jit"] += time.perf_counter() - t0
        acc["njit"] += 1
iron.jit = _timed_jit

# window_driver puts bricks/_verify on sys.path, so it must be imported before bricklib.
# Both call `iron.jit(...)` by attribute at call time, so patching the module object reaches them.
import window_driver as wd
import bricklib
bricklib.iron.jit = _timed_jit

C_IN, K, L = 128, 7, 200                # ~3 windows at T=64, one ci chunk
rng = np.random.default_rng(5)
print(f"BRICK_JIT_CACHE={os.environ.get('BRICK_JIT_CACHE','<unset>')}  "
      f"c_in={C_IN} k={K} L={L} c_out=384", flush=True)

x = rng.standard_normal((C_IN, L)).astype(np.float32)
w = (rng.standard_normal((384, C_IN, K)) * 0.02).astype(np.float32)
for label in ("cold", "warm"):
    for kk in acc:
        acc[kk] = 0.0 if isinstance(acc[kk], float) else 0
    wd.reset_stats()
    t0 = time.perf_counter()
    wd.conv(x, w, np.zeros(384, np.float32), K, 1, f"wt_{label}", ci_chunk=C_IN, resident_depth=1)
    total = time.perf_counter() - t0
    n = wd.stats()["dispatches"]
    rest = total - acc["device"] - acc["jit"]
    print(f"  {label}: {n} dispatches in {total:7.2f} s  "
          f"-> {total/n*1e3:8.1f} ms/dispatch", flush=True)
    print(f"        device {acc['device']:7.2f} s ({acc['ndev']} calls)   "
          f"jit {acc['jit']:7.2f} s ({acc['njit']} calls)   rest {rest:7.2f} s", flush=True)
    print(f"        share: device {100*acc['device']/total:4.1f}%  "
          f"jit {100*acc['jit']/total:4.1f}%  rest {100*rest/total:4.1f}%", flush=True)
