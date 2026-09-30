#!/usr/bin/env python3
"""Phase 1c task 4: same-process trace of the opt-in b2c3 split gate
(net_design.build(split_gate={"b2c3"}), BALANCE.md option (a)) and its neighbours, now that the
whole-chain pace has dropped from ~660-690 to ~478-484 cyc/px (block-1 fix + gate-epilogue
inline, both landed on main). Previously only measured by whole-net repeat/fit
(probe_span_split_gate.py); tracing the split halves needed a small additive fix to
net_design._shim_gate_half (bracket=, same event0()/event1() convention as _shim, off by
default) since it had no bracket param at all.

Env: SPAN_EXPORT, SPAN_DEMO_DIR.
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
EVENTS = [CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL,
         CoreEvent.MEMORY_STALL, CoreEvent.STREAM_STALL]
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "lib"))
from data_root import XDNA_DATA  # noqa: E402
OUT_DIR = XDNA_DATA / "traces" / "span"
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
names = NL.stage_names("up")
rng = np.random.default_rng(0)
x_row, y_row = NL.layout("conv1", WIDTH).in_bytes, NL.layout("up", WIDTH).out_bytes
x = rng.integers(0, 256, size=(TRACE_H + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)


def run(tag, build_kw, stage):
    """One stage traced per dispatch (trace_span_net.py's method): summarize() aggregates ALL
    traced-core events in the JSON by event NAME, not by (pid, stage), so >1 stage per dispatch
    silently merges two cores' timelines into one bogus compute/gap split (found here: 2-stage
    b2c2+b2c3 dispatch gave IDENTICAL, >100%-compute, negative-gap numbers for both)."""
    trace_txt_stem = OUT_DIR / f"splitgate_{tag}_{stage}"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt_stem) + ".txt")
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"sgtr_{tag}_{stage}",
                     tag=f"sgtr{tag}{stage}", trace_stages=[stage], trace_config=tc,
                     coretile_events=EVENTS, egress_shim_col=1, **build_kw)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(N.weights_blob(NP, names, split_gate=build_kw.get("split_gate")),
                      dtype=np.int8, device="npu"),
           iron.zeros((TRACE_H * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    txt = Path(str(trace_txt_stem) + ".txt")
    if not txt.exists() or txt.stat().st_size == 0:
        print(f"  [{tag}/{stage}] EMPTY TRACE, skipping", flush=True)
        return
    json_path = OUT_DIR / f"splitgate_{tag}_{stage}.json"
    tc.trace_to_json(tc.physical_mlir_path, str(json_path))
    res = summarize(str(json_path), CLOCK / 1e9, WIDTH, f"{tag}_{stage}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"splitgate_{tag}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
    print(f"  [{tag}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  compute+gap={cg:.1f}",
         flush=True)


print("\n=== unsplit baseline: b2c2 (prev), b2c3 (gate), b3c1 (next) ===", flush=True)
run("unsplit", {}, "b2c2")
run("unsplit", {}, "b2c3")
run("unsplit", {}, "b3c1")

print("\n=== split b2c3: the two halves, one dispatch each ===", flush=True)
run("split", {"split_gate": frozenset({"b2c3"})}, "b2c3_lo")
run("split", {"split_gate": frozenset({"b2c3"})}, "b2c3_hi")

print("\n=== split b2c3: neighbours under the split build ===", flush=True)
run("split", {"split_gate": frozenset({"b2c3"})}, "b2c2")
run("split", {"split_gate": frozenset({"b2c3"})}, "b3c1")

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
