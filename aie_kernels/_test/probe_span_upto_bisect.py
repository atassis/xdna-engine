#!/usr/bin/env python3
"""Coordinator's chain-length bisection: discriminate core-chain-throttle vs. join/tail-throttle
by measuring whole-net cyc/px at several SPAN_UPTO prefixes, same session/conditions as the rest
of designs/span_sr/TRACE_RESULTS.md (W=32, SKIP_SLACK=8/default, alternated-per-height fitted
slope, same method as probe_span_net_depth.py / probe_span_net_skipslack.py).

Prefixes: b1c3 (4 cores), b3c3 (10 cores), b6c3 (19 cores), conv_2 (20 cores) -- all BEFORE
conv_cat in stage order, so no join is built for these (net_design.build() only wires the join
when "conv_cat" is in the stage list); conv_cat (21 cores, join) and up (22 cores, full net) add
the join and its tail. If the pace already appears by a short core-only prefix (b1c3/b3c3), the
throttle is in the main chain; if the core-only prefixes run near the gate rate (~406-430) and the
jump appears only at conv_cat/up, the throttle is the join/tail (MemTile(4,1) sharing the join +
a weight group, conv_cat's own cross-column read, or the join's fixed ring depth).
"""
import os
import subprocess
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
UPTOS = ["b1c3", "b3c3", "b6c3", "conv_2", "conv_cat", "up"]

try:
    pm = subprocess.run([sys.executable,
                         str(HERE.parents[1] / "scripts" / "npu_power_mode.py")],
                        capture_output=True, text=True, timeout=10).stdout.strip()
except Exception as e:
    pm = f"<npu_power_mode.py failed: {e}>"
print(f"[power mode] {pm}", flush=True)

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
rng = np.random.default_rng(0)
x_row = NL.layout("conv1", WIDTH).in_bytes
kind = dict(NL.STAGES)

designs, names_by_upto, y_row_by_upto, ncores_by_upto = {}, {}, {}, {}
for upto in UPTOS:
    names = NL.stage_names(upto)
    names_by_upto[upto] = names
    ncores_by_upto[upto] = len(names)
    y_row_by_upto[upto] = NL.layout(kind[upto], WIDTH).out_bytes
    print(f"[upto-bisect] building upto={upto} ({len(names)} cores) across heights {HEIGHTS}...",
         flush=True)
    designs[upto] = {h: N.build(WIDTH, h, NP, HERE / "gen" / f"upb_{upto}", tag=f"upb{upto}h{h}",
                                upto=upto)
                    for h in HEIGHTS}

results = {upto: {"xs": [], "ys": []} for upto in UPTOS}
for h in HEIGHTS:
    x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    for upto in UPTOS:  # alternate prefix within each height
        design = designs[upto][h]
        y_row = y_row_by_upto[upto]
        args = [iron.tensor(x, dtype=np.int8, device="npu"),
               iron.tensor(NL.weights_blob(NP, names_by_upto[upto]), dtype=np.int8, device="npu"),
               iron.zeros((h * y_row,), dtype=np.int8, device="npu")]
        design(*args)
        ts = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            design(*args)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        results[upto]["xs"].append(h)
        results[upto]["ys"].append(med)
        print(f"  [{upto}] height {h:4d}: median {med * 1e3:8.3f} ms", flush=True)

print("\n=== fitted rates ===", flush=True)
for upto in UPTOS:
    xs, ys = results[upto]["xs"], results[upto]["ys"]
    slope, icpt = np.polyfit(xs, ys, 1)
    cyc_px = slope / WIDTH * CLOCK
    print(f"upto={upto} ({ncores_by_upto[upto]} cores): slope {slope * 1e6:.1f} us/row, "
         f"intercept {icpt * 1e3:.3f} ms -> {cyc_px:.0f} cyc/px @ {CLOCK / 1e9:.1f} GHz (W={WIDTH})",
         flush=True)
