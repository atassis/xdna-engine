#!/usr/bin/env python3
"""Phase 1e: re-run the chain-length bisection with the TRUSTED instrument only.

TRACE_RESULTS.md's Phase 1a/1c wall-clock SPAN_UPTO bisections claimed the whole-net pace was
already set by block 1 (4 cores) -- but Phase 1d showed the wall-clock method disagrees with a
same-process trace by ~2x on this exact prefix (b1c3: 326 cyc/px traced vs. 553-679 cyc/px wall-
clock in the earlier bisections), and this file's own standing caveat says only same-dispatch
trace splits are trustworthy. So: repeat the bisection using ONLY trace_span_net.summarize(),
never a wall-clock fit.

Method: for each SPAN_UPTO prefix, build TWO designs (b1c2 is present in every prefix from
b1c3 onward, so it is the fixed cross-prefix probe; the prefix's own last stage is the second,
so a prefix-specific compute/gap signature is also visible). One stage traced per dispatch
(summarize() refuses/would-conflate multi-pid JSON otherwise -- TRACE_RESULTS.md Phase 1c).
SPAN_UPTO below "conv_cat" builds no join (net_design.build() only wires f_cat when "conv_cat"
is in the truncated stage list) -- so b1c3..conv_2 isolate the main chain alone; conv_cat/up add
the join+tail. W=32, H=128, same conditions as every other trace in this file.
"""
import json
import os
import subprocess
import sys
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

WIDTH, TRACE_H, CLOCK = 32, 128, 1.8e9
UPTOS = ["b1c3", "b2c3", "b3c3", "b4c3", "b5c3", "b6c3", "conv_2", "conv_cat", "up"]
PROBE = "b1c2"
EVENTS = [CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL,
         CoreEvent.MEMORY_STALL, CoreEvent.STREAM_STALL]
OUT_DIR = Path("/mnt/data/xdna/traces/span")
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    pm = subprocess.run([sys.executable, str(HERE.parents[1] / "scripts" / "npu_power_mode.py")],
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


def trace_stage(upto, stage):
    names = NL.stage_names(upto)
    x_row = NL.layout("conv1", WIDTH).in_bytes
    y_row = NL.layout(kind[upto], WIDTH).out_bytes
    trace_txt = OUT_DIR / f"phase1e_bisect_{upto}_{stage}.txt"
    trace_json = OUT_DIR / f"phase1e_bisect_{upto}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"p1ebisect_{upto}_{stage}",
                     tag=f"p1ebisect{upto}{stage}", upto=upto, trace_stages=[stage],
                     trace_config=tc, coretile_events=EVENTS, egress_shim_col=1)
    x = rng.integers(0, 256, size=(TRACE_H + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((TRACE_H * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [{upto}/{stage}] EMPTY TRACE, skipping", flush=True)
        return None
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, f"{upto}_{stage}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"phase1e_bisect_{upto}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg = (res["compute_cyc_per_px"] or 0) + (res["gap_cyc_per_px"] or 0)
    print(f"  [{upto}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  compute+gap={cg:.1f}",
         flush=True)
    return res


print("\n=== Phase 1e trace-only bisection: probe=b1c2 + prefix's own last stage, one stage per "
     "dispatch ===", flush=True)
rows = []
for upto in UPTOS:
    r_probe = trace_stage(upto, PROBE)
    last = upto  # prefix's own last stage IS `upto` by construction of stage_names()
    r_last = None if last == PROBE else trace_stage(upto, last)
    rows.append((upto, r_probe, r_last))

print("\n=== summary table ===", flush=True)
print(f"{'upto':<10}{'probe(b1c2) cg':>16}{'LOCK%':>8}{'last-stage':>12}{'last cg':>10}{'LOCK%':>8}")
for upto, r_probe, r_last in rows:
    cg_p = (r_probe['compute_cyc_per_px'] + r_probe['gap_cyc_per_px']) if r_probe else float("nan")
    lock_p = r_probe['by_event'].get('LOCK_STALL', {}).get('pct_of_span') if r_probe else None
    if r_last is not None:
        cg_l = r_last['compute_cyc_per_px'] + r_last['gap_cyc_per_px']
        lock_l = r_last['by_event'].get('LOCK_STALL', {}).get('pct_of_span')
        print(f"{upto:<10}{cg_p:>16.1f}{lock_p:>8}{upto:>12}{cg_l:>10.1f}{lock_l:>8}")
    else:
        print(f"{upto:<10}{cg_p:>16.1f}{lock_p:>8}{'(=probe)':>12}{'':>10}{'':>8}")

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
