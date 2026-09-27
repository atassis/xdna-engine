#!/usr/bin/env python3
"""Whole-net rate at several SKIP_SLACK values, same session, alternated per height -- the dose-
response test for the skip-ring latency-throttle hypothesis in designs/span_sr/TRACE_RESULTS.md:
conv_1 (rows_ahead=20) cannot write row r+skip_depth until conv_cat has consumed row r, and the
join ring only admits skip_slack rows of headroom past the minimum. Prediction: whole-net cyc/px
falls roughly as 1/(effective slack) as SKIP_SLACK rises, until it hits the slowest core's own
rate (~406-433 cyc/px measured isolated, probe_span_core_rates.py), then goes flat.

W=32 (production width): unlike MAIN_DEPTH/b1c1-depth, the join ring is bound by MemTile (512 KB),
not L1 -- compile-only sweep found SKIP_SLACK<=25 fits (depth(conv_1)=45, 506.2/512 KiB), 26 fails.
So this runs at full production width, no W=16 fallback needed.
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

WIDTH, HEIGHTS, TRIALS, CLOCK = 32, [64, 128, 192, 256], 5, 1.8e9
SLACKS = [2, 4, 8, 16, 25]

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
names = NL.stage_names("up")
rng = np.random.default_rng(0)
x_row, y_row = NL.layout("conv1", WIDTH).in_bytes, NL.layout("up", WIDTH).out_bytes

designs = {}
for sk in SLACKS:
    print(f"[probe-skipslack] building slack={sk} across heights {HEIGHTS}...", flush=True)
    designs[sk] = {h: N.build(WIDTH, h, NP, HERE / "gen" / f"span_net_ss{sk}", tag=f"spannetss{sk}h{h}",
                              skip_slack=sk)
                  for h in HEIGHTS}

results = {sk: {"xs": [], "ys": []} for sk in SLACKS}
for h in HEIGHTS:
    x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    for sk in SLACKS:  # alternate slack within each height
        design = designs[sk][h]
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
        results[sk]["xs"].append(h)
        results[sk]["ys"].append(med)
        print(f"slack={sk:3d} height {h:4d}: median {med * 1e3:8.3f} ms", flush=True)

print("\n=== fitted slopes ===")
for sk in SLACKS:
    xs, ys = results[sk]["xs"], results[sk]["ys"]
    slope, icpt = np.polyfit(xs, ys, 1)
    cyc_px = slope / WIDTH * CLOCK
    print(f"slack={sk:3d}: slope {slope * 1e6:.1f} us/row, intercept {icpt * 1e3:.3f} ms -> "
         f"{cyc_px:.0f} cyc/px at {CLOCK / 1e9:.1f} GHz (W={WIDTH})", flush=True)
