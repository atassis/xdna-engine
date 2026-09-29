#!/usr/bin/env python3
"""Whole-net rate at two MAIN_DEPTH values, same session, alternated -- the one cheap experiment
for the lockstep hypothesis in designs/span_sr/TRACE_RESULTS.md: windowed()'s fi.acquire(3) against
depth=4 leaves 1 free slot for the upstream producer, forcing near-lockstep sync every row across
the 22-stage chain. L1 forbids raising MAIN_DEPTH at W=32 (blocks 2-6 share silu_x/silu/gate kind
widths, all tight -- see net_design.py main_depth docstring); W=16 buys exactly +1 (4->5) before the
same wall. Runs both depths at W=16, alternated per height, and prints both fitted slopes.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa: E402
import net_design as N  # noqa: E402
import net_layout as NL  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

WIDTH, HEIGHTS, TRIALS, CLOCK = 16, [64, 128, 192, 256, 320], 7, 1.8e9
DEPTHS = [4, 5]

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
names = NL.stage_names("up")
rng = np.random.default_rng(0)
x_row, y_row = NL.layout("conv1", WIDTH).in_bytes, NL.layout("up", WIDTH).out_bytes

designs = {}
for md in DEPTHS:
    designs[md] = {h: N.build(WIDTH, h, NP, HERE / "gen" / f"span_net_depth{md}", tag=f"spannetd{md}h{h}",
                              main_depth=md)
                  for h in HEIGHTS}

results = {md: {"xs": [], "ys": []} for md in DEPTHS}
for h in HEIGHTS:
    x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    for md in DEPTHS:  # alternate depth within each height
        design = designs[md][h]
        args = [iron.tensor(x, dtype=np.int8, device="npu"),
               iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
               iron.zeros((h * y_row,), dtype=np.int8, device="npu")]
        design(*args)
        ts = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            design(*args)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        results[md]["xs"].append(h)
        results[md]["ys"].append(med)
        print(f"depth={md} height {h:4d}: median {med * 1e3:8.3f} ms", flush=True)

for md in DEPTHS:
    xs, ys = results[md]["xs"], results[md]["ys"]
    slope, icpt = np.polyfit(xs, ys, 1)
    cyc_px = slope / WIDTH * CLOCK
    print(f"depth={md}: slope {slope * 1e6:.1f} us/row, intercept {icpt * 1e3:.3f} ms -> "
         f"{cyc_px:.0f} cyc/px at {CLOCK / 1e9:.1f} GHz (W={WIDTH})", flush=True)
