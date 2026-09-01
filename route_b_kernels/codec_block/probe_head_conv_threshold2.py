#!/usr/bin/env python3
"""Is the head conv's ERT timeout driven by the streamed TILE COUNT or by total streamed BYTES?

probe_head_conv_threshold.py found a sharp boundary at fixed ci_chunk=128, k=7: c_out<=512 returns,
c_out>=1024 times out. In window_driver.conv the streamed operand is one weight row per OUTPUT
channel, so c_out IS the tile count and (ci_chunk*k+1) floats IS the tile size -- the two are
independently movable, and which one binds decides the fix. If it is tile count, c_out must be
chunked like c_in already is; if it is bytes, a smaller ci_chunk buys the same headroom.

Part A pins bytes-per-tile and walks the count. Part B pins the count at a value known to FAIL and
shrinks the tile, which is the discriminator: if a failing count starts passing once the tile
shrinks, the count is not what binds.
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_driver as wd

C_IN, K, L = 1024, 7, 20
rng = np.random.default_rng(1)

def trial(tag, c_out, ci_chunk):
    x = rng.standard_normal((C_IN, L)).astype(np.float32)
    w = (rng.standard_normal((c_out, C_IN, K)) * 0.02).astype(np.float32)
    row_b = (ci_chunk * K + 1) * 4
    t0 = time.time()
    try:
        wd.conv(x, w, np.zeros(c_out, np.float32), K, 1, tag, ci_chunk=ci_chunk, resident_depth=1)
        r = "OK  "
    except Exception as e:
        r = "FAIL" if "TIMEOUT" in str(e) else f"ERR({type(e).__name__})"
    print(f"{tag:22s} c_out={c_out:5d} ci_chunk={ci_chunk:4d} tile={row_b:6d}B "
          f"total={c_out*row_b/1e6:7.2f}MB  {r}  {time.time()-t0:6.1f}s", flush=True)

print("--- A: fixed ci_chunk=128 (tile 3588 B), walk the tile COUNT ---", flush=True)
for c_out in (512, 640, 768, 896, 1024):
    trial(f"A_co{c_out}", c_out, 128)

print("--- B: fixed c_out=1024 (a count that FAILS above), shrink the TILE ---", flush=True)
for ci in (128, 64, 32, 16):
    trial(f"B_ci{ci}", 1024, ci)
