#!/usr/bin/env python3
"""Whole-net rate at two b1c1->b1c2 fifo depths, same session, alternated -- tests the
b1c1/b1c2-alternation hypothesis in designs/span_sr/TRACE_RESULTS.md: DEPTH["b1c1"]=3 against
b1c2's windowed() 3-row acquire leaves at most 1 free slot for b1c1 to run ahead into. L1 forbids
depth=4 at W=32 (net_design.py's DEPTH docstring); W=16 admits up to depth=5 (compile-only sweep).
Runs both depths at W=16, alternated per height, and prints both fitted slopes -- same method as
probe_span_net_depth.py's MAIN_DEPTH sweep, applied to depths={"b1c1": d} instead.
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
DEPTHS = [3, 4]

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
names = NL.stage_names("up")
rng = np.random.default_rng(0)
x_row, y_row = NL.layout("conv1", WIDTH).in_bytes, NL.layout("up", WIDTH).out_bytes

designs = {}
for d in DEPTHS:
    designs[d] = {h: N.build(WIDTH, h, NP, HERE / "gen" / f"span_net_b1c1d{d}", tag=f"spannetb1c1d{d}h{h}",
                             depths={"b1c1": d})
                 for h in HEIGHTS}

results = {d: {"xs": [], "ys": []} for d in DEPTHS}
for h in HEIGHTS:
    x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    for d in DEPTHS:  # alternate depth within each height
        design = designs[d][h]
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
        results[d]["xs"].append(h)
        results[d]["ys"].append(med)
        print(f"b1c1_depth={d} height {h:4d}: median {med * 1e3:8.3f} ms", flush=True)

for d in DEPTHS:
    xs, ys = results[d]["xs"], results[d]["ys"]
    slope, icpt = np.polyfit(xs, ys, 1)
    cyc_px = slope / WIDTH * CLOCK
    print(f"b1c1_depth={d}: slope {slope * 1e6:.1f} us/row, intercept {icpt * 1e3:.3f} ms -> "
         f"{cyc_px:.0f} cyc/px at {CLOCK / 1e9:.1f} GHz (W={WIDTH})", flush=True)
