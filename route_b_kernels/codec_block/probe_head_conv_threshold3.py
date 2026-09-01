#!/usr/bin/env python3
"""Pin the streamed-operand size at which window_driver.conv starts returning ERT timeout.

probe_head_conv_threshold2.py showed the binding quantity is TOTAL streamed bytes, not tile count
and not tile size: 1024 tiles pass at 1796 B each (1.84 MB) and fail at 3588 B (3.67 MB), while 512
tiles pass at 3588 B (1.84 MB). Both observed passes sit at 1.84 MB and the first observed failure
at 2.30 MB, which brackets 2 MiB = 2097152 B.

This straddles that value from both sides at two DIFFERENT (count, tile) factorisations. If a single
byte figure separates pass from fail in both, the limit is on the operand's total size and nothing
else; if the two factorisations disagree, it is not a pure byte limit and this probe says so rather
than letting a tidy round number stand unearned.
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_driver as wd

C_IN, K, L = 1024, 7, 20
LIMIT = 2 * 1024 * 1024
rng = np.random.default_rng(2)

def trial(tag, c_out, ci_chunk):
    row_b = (ci_chunk * K + 1) * 4
    total = c_out * row_b
    x = rng.standard_normal((C_IN, L)).astype(np.float32)
    w = (rng.standard_normal((c_out, C_IN, K)) * 0.02).astype(np.float32)
    t0 = time.time()
    try:
        wd.conv(x, w, np.zeros(c_out, np.float32), K, 1, tag, ci_chunk=ci_chunk, resident_depth=1)
        r = "OK  "
    except Exception as e:
        r = "FAIL" if "TIMEOUT" in str(e) else f"ERR({type(e).__name__})"
    side = "under" if total < LIMIT else "OVER "
    print(f"{tag:12s} c_out={c_out:5d} tile={row_b:5d}B total={total:9d}B "
          f"({total/LIMIT:5.3f}x2MiB {side}) {r} {time.time()-t0:6.1f}s", flush=True)

# tile 3588 B: 2 MiB / 3588 = 584.4 -> 584 under, 585 over
print("--- factorisation 1: ci_chunk=128, tile 3588 B ---", flush=True)
trial("F1_under", 584, 128)
trial("F1_over", 585, 128)

# tile 1796 B: 2 MiB / 1796 = 1167.7 -> 1167 under, 1168 over
print("--- factorisation 2: ci_chunk=64, tile 1796 B ---", flush=True)
trial("F2_under", 1167, 64)
trial("F2_over", 1168, 64)
