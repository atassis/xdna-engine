#!/usr/bin/env python3
"""Phase 1d: test the fixed-per-row-cost hypothesis (T_row = a + b*W, i.e. pace in cyc/px =
b + a/W) on the SPAN_UPTO=b1c3 4-core prefix (net_design.py's own zero-slack/pace-setter
block, per TRACE_RESULTS.md's chain-length bisection). W in {16, 32} -- 48/64 fail L1 on tile
(0,3) at main's defaults (probe_span_w_sweep_compile.py), so this is the full compiling range.

Two independent measurements per W, same session:
  1. Same-process trace of b1c2 and b1c3 individually (one stage per dispatch, H=128), reporting
     compute_cyc_per_row / gap_cyc_per_row DIRECTLY (not divided by W) -- trace_span_net.summarize()
     already reports these un-normalized.
  2. Wall-clock fitted slope of the whole b1c3-prefix chain across heights (same method as
     probe_span_upto_bisect.py), converted to cyc/row (slope * CLOCK, not divided by W) as a
     cross-check -- this already isolates the marginal per-ROW cost from the dispatch's own
     one-time overhead, which is a DIFFERENT fixed cost than the one this hypothesis is about.

Fits T_row = a + b*W by solving the two-point system exactly (only 2 W admit at L1), for compute
alone and for compute+gap, per stage and from the wall-clock series.
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

UPTO = "b1c3"
WIDTHS = [16, 32]
STAGES = ["b1c2", "b1c3"]
TRACE_H = 128
HEIGHTS = [64, 128, 192, 256]
TRIALS = 5
CLOCK = 1.8e9
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
names = NL.stage_names(UPTO)
kind = dict(NL.STAGES)
y_row_by_w = {}

trace_rows = {}  # (w, stage) -> summarize() dict


def trace_stage(w, stage):
    x_row = NL.layout("conv1", w).in_bytes
    y_row = NL.layout(kind[UPTO], w).out_bytes
    trace_txt = OUT_DIR / f"wsweep_trace_w{w}_{stage}.txt"
    trace_json = OUT_DIR / f"wsweep_trace_w{w}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(w, TRACE_H, NP, HERE / "gen" / f"wsweeptr_w{w}_{stage}",
                     tag=f"wsweeptr{w}{stage}", upto=UPTO, trace_stages=[stage],
                     trace_config=tc, coretile_events=EVENTS, egress_shim_col=1)
    x = rng.integers(0, 256, size=(TRACE_H + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((TRACE_H * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [w{w}/{stage}] EMPTY TRACE, skipping", flush=True)
        return None
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, w, f"w{w}_{stage}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"wsweep_trace_w{w}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg_row = (res["compute_cyc_per_row"] or 0) + (res["gap_cyc_per_row"] or 0)
    print(f"  [w{w}/{stage}] compute {res['compute_cyc_per_row']} cyc/row "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_row']} cyc/row "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  compute+gap/row={cg_row:.1f}",
         flush=True)
    return res


print("\n=== Same-process trace, one stage per dispatch, per W ===", flush=True)
for w in WIDTHS:
    for stage in STAGES:
        trace_rows[(w, stage)] = trace_stage(w, stage)

print("\n=== Wall-clock secondary: b1c3-prefix chain, fitted slope over heights, per W ===",
     flush=True)
wallclock_cyc_row = {}
for w in WIDTHS:
    x_row = NL.layout("conv1", w).in_bytes
    y_row = NL.layout(kind[UPTO], w).out_bytes
    design = N.build(w, HEIGHTS[0], NP, HERE / "gen" / f"wsweepwc_w{w}", tag=f"wsweepwc{w}",
                     upto=UPTO)
    designs = {h: N.build(w, h, NP, HERE / "gen" / f"wsweepwc_w{w}", tag=f"wsweepwc{w}", upto=UPTO)
              for h in HEIGHTS}
    xs, ys = [], []
    for h in HEIGHTS:
        x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
        d = designs[h]
        args = [iron.tensor(x, dtype=np.int8, device="npu"),
               iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
               iron.zeros((h * y_row,), dtype=np.int8, device="npu")]
        d(*args)
        ts = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            d(*args)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        xs.append(h)
        ys.append(med)
        print(f"  [w{w}] height {h:4d}: median {med * 1e3:8.3f} ms", flush=True)
    slope, icpt = np.polyfit(xs, ys, 1)
    cyc_row = slope * CLOCK
    wallclock_cyc_row[w] = cyc_row
    print(f"[w{w}] slope {slope * 1e6:.2f} us/row, intercept {icpt * 1e3:.3f} ms -> "
         f"{cyc_row:.1f} cyc/row @ {CLOCK / 1e9:.1f} GHz (upto={UPTO})", flush=True)


def fit_ab(w0, t0, w1, t1):
    """T_row = a + b*W through two points."""
    b = (t1 - t0) / (w1 - w0)
    a = t0 - b * w0
    return a, b


print("\n=== Fit T_row = a + b*W (two-point, W=16 vs W=32) ===", flush=True)
w0, w1 = WIDTHS
for stage in STAGES:
    r0, r1 = trace_rows[(w0, stage)], trace_rows[(w1, stage)]
    if r0 is None or r1 is None:
        print(f"[{stage}] missing trace, skip fit", flush=True)
        continue
    ac, bc = fit_ab(w0, r0["compute_cyc_per_row"], w1, r1["compute_cyc_per_row"])
    cg0 = r0["compute_cyc_per_row"] + (r0["gap_cyc_per_row"] or 0)
    cg1 = r1["compute_cyc_per_row"] + (r1["gap_cyc_per_row"] or 0)
    acg, bcg = fit_ab(w0, cg0, w1, cg1)
    ag, bg = fit_ab(w0, r0["gap_cyc_per_row"] or 0, w1, r1["gap_cyc_per_row"] or 0)
    print(f"[{stage}] compute:      a={ac:8.1f} b={bc:6.2f}   "
         f"(row cost = {ac:.1f} + {bc:.2f}*W)", flush=True)
    print(f"[{stage}] gap:          a={ag:8.1f} b={bg:6.2f}", flush=True)
    print(f"[{stage}] compute+gap:  a={acg:8.1f} b={bcg:6.2f}", flush=True)

acw, bcw = fit_ab(w0, wallclock_cyc_row[w0], w1, wallclock_cyc_row[w1])
print(f"[wallclock upto={UPTO}] a={acw:8.1f} b={bcw:6.2f}", flush=True)

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
