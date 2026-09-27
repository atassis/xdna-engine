#!/usr/bin/env python3
"""Coordinator's re-test of the b1c1<->b1c2 alternation-sum hypothesis, now at SKIP_SLACK=8 (the
earlier refutation in TRACE_RESULTS.md ran at SKIP_SLACK=2, where the ring throttle set the pace
and could have masked this). DEPTH["b1c1"]=3 against b1c2's windowed() 3-row acquire leaves ZERO
producer slack -> b1c1 and b1c2 must alternate, predicting their computes ADD: 319.87 + 322.79 =
642.66 cyc/px (traced at slack=2, but compute is stage-fixed and should not depend on slack) vs.
the chain-bisection's measured pace at upto=b1c3 of 663 cyc/px (and 655-690 for longer chains).

Three parts, all W=32 SKIP_SLACK=8/default unless noted:
  1. Bisect upto in {conv_1, b1c1, b1c2, b1c3} (fitted whole-net rate) -- does the pace already
     appear exactly at b1c2 (2 cores), one core short of b1c3?
  2. Same-process A/B, depths={"b1c1": 3} vs 4, on upto=b1c3 (compile-only sweep found depth=4
     fits L1 here, unlike the full net) -- fitted rate + same-process trace of b1c1 and b1c2.
  3. Same-process A/B, depths={"b1c1": 3} vs 4, on the FULL NET (upto=up) at W=16 (depth=4 fails
     L1 at W=32 on the full net specifically -- see TRACE_RESULTS.md for the exact aiecc error and
     buffer-map analysis) -- fitted rate only (trace corroboration already done in part 2's W=32
     short chain, which is the more trustworthy instrument per this file's own standing method).
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
kind = dict(NL.STAGES)


def fitted_rate_sweep(label, width, configs):
    """configs: {name: (upto, build_kwargs)}. Alternates per height, returns {name: cyc_px}."""
    x_row = NL.layout("conv1", width).in_bytes
    designs, names_by, y_row_by = {}, {}, {}
    for name, (upto, kw) in configs.items():
        names = NL.stage_names(upto)
        names_by[name] = names
        y_row_by[name] = NL.layout(kind[upto], width).out_bytes
        designs[name] = {h: N.build(width, h, NP, HERE / "gen" / f"retest_{label}_{name}",
                                    tag=f"retest{label}{name}h{h}", upto=upto, **kw)
                        for h in HEIGHTS}
    results = {name: {"xs": [], "ys": []} for name in configs}
    for h in HEIGHTS:
        x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
        for name in configs:
            design = designs[name][h]
            y_row = y_row_by[name]
            args = [iron.tensor(x, dtype=np.int8, device="npu"),
                   iron.tensor(NL.weights_blob(NP, names_by[name]), dtype=np.int8, device="npu"),
                   iron.zeros((h * y_row,), dtype=np.int8, device="npu")]
            design(*args)
            ts = []
            for _ in range(TRIALS):
                t0 = time.perf_counter()
                design(*args)
                ts.append(time.perf_counter() - t0)
            med = float(np.median(ts))
            results[name]["xs"].append(h)
            results[name]["ys"].append(med)
            print(f"  [{label}/{name}] height {h:4d}: median {med * 1e3:8.3f} ms", flush=True)
    out = {}
    for name in configs:
        xs, ys = results[name]["xs"], results[name]["ys"]
        slope, icpt = np.polyfit(xs, ys, 1)
        cyc_px = slope / width * CLOCK
        out[name] = cyc_px
        print(f"[{label}/{name}]: slope {slope * 1e6:.1f} us/row, intercept {icpt * 1e3:.3f} ms -> "
             f"{cyc_px:.0f} cyc/px @ {CLOCK / 1e9:.1f} GHz (W={width})", flush=True)
    return out


def trace_stage(stage, width, trace_h, build_kw, tag):
    x_row = NL.layout("conv1", width).in_bytes
    y_row = NL.layout("up", width).out_bytes
    names = NL.stage_names("up")
    trace_txt = OUT_DIR / f"retest_{tag}_{stage}.txt"
    trace_json = OUT_DIR / f"retest_{tag}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(width, trace_h, NP, HERE / "gen" / f"retesttr_{tag}_{stage}",
                     tag=f"retesttr{tag}{stage}", trace_stages=[stage], trace_config=tc,
                     coretile_events=EVENTS, egress_shim_col=1, **build_kw)
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
    (OUT_DIR / f"retest_{tag}_{stage}_summary.json").write_text(__import__("json").dumps(res, indent=2))
    cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
    print(f"  [{tag}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  compute+gap={cg:.1f}",
         flush=True)
    return res


print("\n=== Part 1: bisect upto in {conv_1, b1c1, b1c2, b1c3}, W=32, SKIP_SLACK=8 ===", flush=True)
fitted_rate_sweep("bisect1", 32, {
    "conv_1": ("conv_1", {}), "b1c1": ("b1c1", {}), "b1c2": ("b1c2", {}), "b1c3": ("b1c3", {})})

print("\n=== Part 2: b1c1 depth 3 vs 4 on upto=b1c3, W=32 (fits L1 here) ===", flush=True)
fitted_rate_sweep("b1c1depth_short", 32, {
    "d3": ("b1c3", {"depths": {"b1c1": 3}}), "d4": ("b1c3", {"depths": {"b1c1": 4}})})
for d in (3, 4):
    for stage in ("b1c1", "b1c2"):
        trace_stage(stage, 32, 128, {"depths": {"b1c1": d}}, f"shortd{d}")

print("\n=== Part 3: b1c1 depth 3 vs 4 on the FULL NET, W=16 (depth=4 fails L1 at W=32) ===",
     flush=True)
fitted_rate_sweep("b1c1depth_full_w16", 16, {
    "d3": ("up", {"depths": {"b1c1": 3}}), "d4": ("up", {"depths": {"b1c1": 4}})})

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
