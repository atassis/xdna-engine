#!/usr/bin/env python3
"""cProfile ONE window_driver conv call. Two theories about where the wall time goes -- per-window
xclbin rebuilds, then the disabled JIT cache -- were each measured and each explained only a
minority of it. ~89% of a 1.18 s dispatch is still unattributed with the cache ON, so stop
hypothesising and read the profile."""
import cProfile, io, pstats, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_driver as wd

C_IN, K, L = 128, 7, 136           # ~1 window
rng = np.random.default_rng(6)
x = rng.standard_normal((C_IN, L)).astype(np.float32)
w = (rng.standard_normal((384, C_IN, K)) * 0.02).astype(np.float32)
b = np.zeros(384, np.float32)

wd.conv(x, w, b, K, 1, "prof_warm", ci_chunk=C_IN, resident_depth=1)   # warm any one-time cost

pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
wd.conv(x, w, b, K, 1, "prof_run", ci_chunk=C_IN, resident_depth=1)
pr.disable()
print(f"profiled call: {time.perf_counter()-t0:.2f} s, {wd.stats()['dispatches']} dispatches\n")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28)
for line in s.getvalue().splitlines():
    if line.strip():
        print(line[:150])
