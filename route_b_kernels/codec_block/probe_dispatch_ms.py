#!/usr/bin/env python3
"""Is the conv cliff a SIZE limit or the driver's 2000 ms TDR watchdog?

probe_head_conv_threshold{,2,3}.py established that window_driver.conv returns
ERT_CMD_STATE_TIMEOUT above ~1.9-2.1 MB of total streamed operand, with the boundary independent of
how those bytes factor into (tile count x tile size). That reads as a ~2 MiB size limit -- and
amdxdna's `tdr_timeout_ms` defaults to 2000 (aie2_ctx.c:31), so "about 2 MB" and "2000 ms" are
numerically adjacent for entirely unrelated reasons. Exactly the coincidence that cements a wrong
constant.

Discriminator: measure WALL TIME OF ONE DISPATCH as the operand grows, staying under the cliff. A
time limit predicts the last passing dispatch sits just under 2000 ms and the trend extrapolates to
the observed cliff. A size limit predicts dispatches are fast (tens of ms) right up to a hard edge.

c_in is set to ci_chunk so there is exactly ONE c_in chunk, and L is chosen to give ONE t-window,
so `wd.stats()["dispatches"]` is 1 per trial and wall time is that dispatch plus a fixed host cost.
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_driver as wd

CI, K, L = 128, 7, 20
rng = np.random.default_rng(3)

print(f"c_in=ci_chunk={CI} k={K} L={L}  (one c_in chunk, one t-window => 1 dispatch/trial)",
      flush=True)
print(f"{'c_out':>6} {'streamed':>10} {'dispatches':>10} {'wall_ms':>9}  result", flush=True)
prev = None
for c_out in (64, 128, 256, 384, 512, 576):
    x = rng.standard_normal((CI, L)).astype(np.float32)
    w = (rng.standard_normal((c_out, CI, K)) * 0.02).astype(np.float32)
    wd.reset_stats()
    t0 = time.time()
    try:
        wd.conv(x, w, np.zeros(c_out, np.float32), K, 1, f"ms{c_out}",
                ci_chunk=CI, resident_depth=1)
        res = "OK"
    except Exception as e:
        res = "TIMEOUT" if "TIMEOUT" in str(e) else type(e).__name__
    ms = (time.time() - t0) * 1e3
    n = wd.stats()["dispatches"]
    print(f"{c_out:6d} {c_out*(CI*K+1)*4:10d} {n:10d} {ms:9.1f}  {res}", flush=True)
