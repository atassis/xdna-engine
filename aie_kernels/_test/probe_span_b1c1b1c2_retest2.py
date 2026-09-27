#!/usr/bin/env python3
"""Continuation of probe_span_b1c1b1c2_retest.py (Part 1/2 completed; Part 2's d=4 traces and
Part 3 crashed the process, since compiling upto=b1c3 with trace_stages=["b1c1"] at depth=4 hits
the SAME L1 wall net_design.py's b1c1_depth docstring flags for the full net -- but at a SMALLER
overflow: `data_sizes={"b1c1": 4160}` (aiecc's own suggested fix, now a net_design.build() param)
fixes it for the short chain. Two parts:

  1. b1c1/b1c2 traces at depth=4 (upto=b1c3, W=32, data_sizes={"b1c1": 4160}) -- the corroboration
     the crash lost.
  2. b1c1 depth 3 vs 4 on the FULL NET (upto=up) at W=16, fitted rate (Part 3, never ran).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa: E402
from aie.utils.trace.config import TraceConfig  # noqa: E402
from aie.utils.trace.events import CoreEvent  # noqa: E402
import aie.utils as aie_utils  # noqa: E402
import net_design as N  # noqa: E402
import net_layout as NL  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

sys.path.insert(0, str(HERE))
from trace_span_net import summarize  # noqa: E402

HEIGHTS, TRIALS, CLOCK = [64, 128, 192, 256], 5, 1.8e9
EVENTS = [CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL,
         CoreEvent.MEMORY_STALL, CoreEvent.STREAM_STALL]
OUT_DIR = Path("/mnt/data/xdna/traces/span")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def trace_stage(stage, width, trace_h, upto, build_kw, tag):
    x_row = NL.layout("conv1", width).in_bytes
    kind = dict(NL.STAGES)
    names = NL.stage_names(upto)
    y_row = NL.layout(kind[names[-1]], width).out_bytes
    trace_txt = OUT_DIR / f"retest2_{tag}_{stage}.txt"
    trace_json = OUT_DIR / f"retest2_{tag}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(width, trace_h, NP, HERE / "gen" / f"retest2tr_{tag}_{stage}",
                     tag=f"retest2tr{tag}{stage}", upto=upto, trace_stages=[stage],
                     trace_config=tc, coretile_events=EVENTS, egress_shim_col=1, **build_kw)
    x = rng.integers(0, 256, size=(trace_h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((trace_h * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [{tag}/{stage}] EMPTY TRACE, skipping", flush=True)
        return None
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, width, f"{tag}_{stage}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"retest2_{tag}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
    print(f"  [{tag}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  compute+gap={cg:.1f}",
         flush=True)
    return res


print("\n=== Part 2 cont'd: b1c1/b1c2 traces at depth=4, upto=b1c3, W=32, "
     "data_sizes={'b1c1': 4160} ===", flush=True)
for stage in ("b1c1", "b1c2"):
    trace_stage(stage, 32, 128, "b1c3",
               {"depths": {"b1c1": 4}, "data_sizes": {"b1c1": 4160}}, "shortd4")

print("\n=== Part 3: b1c1 depth 3 vs 4 on the FULL NET, W=16 ===", flush=True)
WIDTH = 16
x_row = NL.layout("conv1", WIDTH).in_bytes
y_row = NL.layout("up", WIDTH).out_bytes
names = NL.stage_names("up")
designs = {}
for d in (3, 4):
    designs[d] = {h: N.build(WIDTH, h, NP, HERE / "gen" / f"retest2full_d{d}",
                             tag=f"retest2fulld{d}h{h}", depths={"b1c1": d})
                 for h in HEIGHTS}
results = {d: {"xs": [], "ys": []} for d in (3, 4)}
for h in HEIGHTS:
    x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    for d in (3, 4):
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
        print(f"  [fullw16/d{d}] height {h:4d}: median {med * 1e3:8.3f} ms", flush=True)
for d in (3, 4):
    xs, ys = results[d]["xs"], results[d]["ys"]
    slope, icpt = np.polyfit(xs, ys, 1)
    cyc_px = slope / WIDTH * CLOCK
    print(f"[fullw16/d{d}]: slope {slope * 1e6:.1f} us/row, intercept {icpt * 1e3:.3f} ms -> "
         f"{cyc_px:.0f} cyc/px @ {CLOCK / 1e9:.1f} GHz (W={WIDTH})", flush=True)

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
