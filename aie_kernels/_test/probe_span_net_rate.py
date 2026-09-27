#!/usr/bin/env python3
"""Rate of the whole SPAN network on one 32-px strip: wall time vs strip height, fitted slope ->
cycles per LR pixel. The chain cannot beat its slowest core; compare with the gate core's kernel
rate from probe_span_core_rates.py. Cycles assume 1.8 GHz, printed. Prints, exits 0."""
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

WIDTH, HEIGHTS, TRIALS, CLOCK = 32, [64, 128, 192, 256, 320], 7, 1.8e9

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
names = NL.stage_names("up")
rng = np.random.default_rng(0)
x_row, y_row = NL.layout("conv1", WIDTH).in_bytes, NL.layout("up", WIDTH).out_bytes
xs, ys = [], []
for h in HEIGHTS:
    design = N.build(WIDTH, h, NP, HERE / "gen" / "span_net_rate", tag=f"spannetrate{h}")
    x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
            iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
            iron.zeros((h * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        design(*args)
        ts.append(time.perf_counter() - t0)
    xs.append(h)
    ys.append(float(np.median(ts)))
    print(f"height {h:4d}: median {ys[-1] * 1e3:8.3f} ms", flush=True)
slope, icpt = np.polyfit(xs, ys, 1)
print(f"slope {slope * 1e6:.1f} us/row, intercept {icpt * 1e3:.3f} ms -> "
      f"{slope / WIDTH * CLOCK:.0f} cyc per LR px at {CLOCK / 1e9:.1f} GHz")
