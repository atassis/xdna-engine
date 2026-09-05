#!/usr/bin/env python3
"""Does the JIT cache actually HIT for a repeated design? The earlier attempt at this used a
DIFFERENT tag for its two calls, so both were cold builds and the cache looked worthless (30%).
The profile since showed 83% of a dispatch is an aiecc subprocess at ~2 s per unique design, so a
real hit should be visible. Same tag, twice."""
import os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_driver as wd

C_IN, K, L = 128, 7, 136
rng = np.random.default_rng(7)
x = rng.standard_normal((C_IN, L)).astype(np.float32)
w = (rng.standard_normal((384, C_IN, K)) * 0.02).astype(np.float32)
b = np.zeros(384, np.float32)
print(f"BRICK_JIT_CACHE={os.environ.get('BRICK_JIT_CACHE','<unset>')}", flush=True)
for i in (1, 2, 3):
    t0 = time.perf_counter()
    wd.conv(x, w, b, K, 1, "samehit", ci_chunk=C_IN, resident_depth=1)   # SAME tag each time
    print(f"  call {i} (same tag): {time.perf_counter()-t0:6.2f} s", flush=True)
