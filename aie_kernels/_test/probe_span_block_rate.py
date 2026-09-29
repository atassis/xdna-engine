#!/usr/bin/env python3
"""Rate of the 3-core SPAN block: wall time vs strip height, fitted slope -> cycles per pixel.
The chain's floor is its slowest core at the same shape, which probe_span_core_rates.py
measures (the gate core). Cycles assume 1.8 GHz, printed. Prints, exits 0."""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa: E402
import block_design as B  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

WIDTH, HEIGHTS, TRIALS, CLOCK = 32, [64, 128, 192, 256, 320, 384], 7, 1.8e9

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
P = net.core_params(2)
rng = np.random.default_rng(0)
xs, ys = [], []
for h in HEIGHTS:
    design = B.build(WIDTH, h, P, HERE / "gen" / "span_block_rate", tag=f"spanrate{h}")
    rows = rng.integers(-40, 40, size=h * B.row_bytes(WIDTH), dtype=np.int64).astype(np.int8)
    xt = iron.tensor(rows, dtype=np.int8, device="npu")
    pt = [iron.tensor(P[k]["blob"], dtype=np.int8, device="npu") for k in ("c1", "c2", "c3")]
    yt = iron.zeros((h * B.row_bytes(WIDTH),), dtype=np.int8, device="npu")
    design(xt, *pt, yt)
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        design(xt, *pt, yt)
        ts.append(time.perf_counter() - t0)
    xs.append(h)
    ys.append(float(np.median(ts)))
    print(f"height {h:4d}: median {ys[-1] * 1e3:8.3f} ms")
slope, icpt = np.polyfit(xs, ys, 1)
cyc_px = slope / WIDTH * CLOCK
print(f"slope {slope * 1e6:.1f} us/row, intercept {icpt * 1e3:.3f} ms -> {cyc_px:.0f} cyc/px at "
      f"{CLOCK / 1e9:.1f} GHz")
