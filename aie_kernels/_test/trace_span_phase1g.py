#!/usr/bin/env python3
"""Phase 1g of the SPAN stall task (2026-09-27-npu-any-game-realtime.md): re-test
PROD_DEPTH/cat_cons_depth/skip_cons_depths now that Phase 1e/1f localized the ONLY remaining
throttle to the join build itself (conv_2 -> conv_cat, ~326 -> ~483.8 cyc/px on main 45ceaf1).
Phase 1a tested these same three levers and found them NULL, but that was BEFORE the block-1
b1c1/b1c2 fix landed -- the pace was still ~610-690 (block-1 ping-pong), so a lever that only
affects the join could not have shown up on that probe. See TRACE_RESULTS.md's Phase 1g section
for the full writeup, including three device-crash-driven trims to this lever list:
`cat_cons_depth=4`, `skip_cons_depths={"conv_1":4}`, and `skip_cons_depths={"b6c1":5}` all fit L1
UNTRACED (`compile_sweep_phase1g.py`) but overflow different tiles once tracing is enabled
anywhere in the build (network-wide instrumentation, not local to the traced core) -- dropped
from the traced sweep below, not resolved.

Traces conv_cat (the join emission stage, most directly wired to these levers) and b1c2 (the
fixed cross-prefix pace probe used in Phase 1e/1f), one stage per dispatch (this file's own
`summarize()` is pid-filtered but the convention is still one stage/dispatch per the Phase 1c
summarize() defect note).
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


def trace_stage(stage, build_kw, tag):
    trace_txt = OUT_DIR / f"p1g_{tag}_{stage}.txt"
    trace_json = OUT_DIR / f"p1g_{tag}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"p1gtr_{tag}_{stage}",
                     tag=f"p1gtr{tag}{stage}", trace_stages=[stage], trace_config=tc,
                     coretile_events=EVENTS, egress_shim_col=1, **build_kw)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((TRACE_H * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [{tag}/{stage}] EMPTY TRACE, skipping", flush=True)
        return None
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, f"{tag}_{stage}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"p1g_{tag}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    print(f"  [{tag}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  "
         f"compute+gap={res['compute_cyc_per_px'] + res['gap_cyc_per_px']:.1f}", flush=True)
    return res


print("\n=== Step 1: baseline traces, main defaults, full net (W=32 H=128) ===", flush=True)
for stage in ("conv_1", "b1c1", "b1c3", "b6c1", "conv_2", "conv_cat", "b1c2"):
    trace_stage(stage, {}, "base")

# cat_cons_depth=4, skip_cons_depths={"conv_1":4}/{"b6c1":5} fit L1 untraced
# (compile_sweep_phase1g.py) but each overflows a different tile once ANY trace_config is
# active in the build -- dropped here, see module docstring and TRACE_RESULTS.md Phase 1g.
LEVERS = [
    ("prod8", {"prod_depth": 8}),
    ("cd3", {"cat_cons_depth": 3}),
    ("sc_b1c3_5", {"skip_cons_depths": {"b1c3": 5}}),
    ("combo", {"prod_depth": 8, "cat_cons_depth": 3}),
    ("combo2", {"prod_depth": 8, "cat_cons_depth": 3, "skip_cons_depths": {"b1c3": 5}}),
]

print("\n=== Step 2: lever A/B on conv_cat (join emission, primary indicator) ===", flush=True)
for tag, kw in LEVERS:
    trace_stage("conv_cat", kw, tag)

print("\n=== Step 3: combo2, also re-check b1c2 (pace probe) ===", flush=True)
trace_stage("b1c2", dict(LEVERS[-1][1]), "combo2")

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
