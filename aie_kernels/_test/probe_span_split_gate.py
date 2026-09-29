#!/usr/bin/env python3
"""Whole-net rate, split vs unsplit b2c3 gate (BALANCE.md option (a)), same session, alternated
per height -- same method as probe_span_net_skipslack.py. b2c3's own compute drops from ~406-433
cyc/px (one core, 48 out channels) to a 32/16-channel pair (conv3x3_core processes output-channel
blocks in pairs, so COUT must be a multiple of 16 -- 48 has no even 24/24 split point).

Env: SPAN_EXPORT, SPAN_DEMO_DIR.
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
ARMS = {"unsplit": frozenset(), "split_b2c3": frozenset({"b2c3"})}

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
names = NL.stage_names("up")
rng = np.random.default_rng(0)
x_row, y_row = NL.layout("conv1", WIDTH).in_bytes, NL.layout("up", WIDTH).out_bytes

designs = {}
for arm, sg in ARMS.items():
    print(f"[probe-split-gate] building {arm} across heights {HEIGHTS}...", flush=True)
    designs[arm] = {h: N.build(WIDTH, h, NP, HERE / "gen" / f"span_net_{arm}",
                               tag=f"spansg{arm}h{h}", split_gate=sg)
                   for h in HEIGHTS}

results = {arm: {"xs": [], "ys": []} for arm in ARMS}
for h in HEIGHTS:
    x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    for arm in ARMS:  # alternate arm within each height
        design = designs[arm][h]
        args = [iron.tensor(x, dtype=np.int8, device="npu"),
               iron.tensor(N.weights_blob(NP, names, split_gate=ARMS[arm]), dtype=np.int8,
                          device="npu"),
               iron.zeros((h * y_row,), dtype=np.int8, device="npu")]
        design(*args)
        ts = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            design(*args)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        results[arm]["xs"].append(h)
        results[arm]["ys"].append(med)
        print(f"{arm:12s} height {h:4d}: median {med * 1e3:8.3f} ms", flush=True)

print("\n=== fitted slopes ===")
for arm in ARMS:
    xs, ys = results[arm]["xs"], results[arm]["ys"]
    slope, icpt = np.polyfit(xs, ys, 1)
    cyc_px = slope / WIDTH * CLOCK
    print(f"{arm:12s}: slope {slope * 1e6:.1f} us/row, intercept {icpt * 1e3:.3f} ms -> "
         f"{cyc_px:.0f} cyc/px at {CLOCK / 1e9:.1f} GHz (W={WIDTH})", flush=True)
