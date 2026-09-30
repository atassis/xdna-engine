#!/usr/bin/env python3
"""Phase 1i: re-trace the epilogue-cut kinds after the restrict/C3_LOOP_RANGE edits to
conv3x3_u8.cc's LUT/LUT16/gate epilogue gather loops (apply_lut_inplace, apply_lut16, the gate
inner loop). Same conditions as Phase 1c's baseline (main defaults, W=32, H=128, one stage per
dispatch): b1c1 (silu16), b1c2 (silu_i16), b1c3 (gate), b2c2 (silu). Compare against Phase 1c's
table in TRACE_RESULTS.md: b1c1 266.35, b1c2 324.01, b1c3 315.29, b2c2 251.09 (compute cyc/px).
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

TODO = os.environ.get("PHASE1I_STAGES", "b1c1,b1c2,b1c3,b2c2").split(",")
print(f"[phase1i] {len(TODO)} stages: {TODO}", flush=True)

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

results = {}
for stage in TODO:
    trace_txt = OUT_DIR / f"phase1i_{stage}.txt"
    trace_json = OUT_DIR / f"phase1i_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"p1i_{stage}",
                     tag=f"phase1i{stage}", trace_stages=[stage], trace_config=tc,
                     coretile_events=EVENTS, egress_shim_col=1)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((TRACE_H * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [{stage}] EMPTY TRACE, skipping", flush=True)
        results[stage] = None
        continue
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, stage)
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"phase1i_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
    print(f"  [{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  "
         f"compute+gap={cg:.1f}", flush=True)
    results[stage] = res

print("\n=== summary ===", flush=True)
for stage, res in results.items():
    if res is None:
        print(f"  {stage}: EMPTY TRACE", flush=True)
        continue
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
    print(f"  {stage}: compute={res['compute_cyc_per_px']:.1f} gap={res['gap_cyc_per_px']:.1f} "
         f"LOCK_STALL={lock}% compute+gap={cg:.1f}", flush=True)

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
