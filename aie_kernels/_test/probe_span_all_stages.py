#!/usr/bin/env python3
"""Coordinator follow-up on probe_span_stall_levers.py: 8/22 stages were traced
(conv_1, b1c1, b1c2, b3c3, b6c1, conv_2, conv_cat, up), all landing at the same ~610-630
cyc/px compute+gap pace, LOCK_STALL-dominated -- every buffer-depth lever tried (SKIP_SLACK,
MAIN_DEPTH, b1c1->b1c2, PROD_DEPTH, cat_cons_depth) is null at b3c3. Coordinator's alternative:
the pace-setter is one of the 14 UNTRACED stages (low LOCK_STALL, compute near the pace) or a
non-core resource (shared DMA channel / MemTile pool), since latency alone cannot explain a
buffer-insensitive throughput ceiling.

Traces the 14 remaining stages one at a time (one dispatch each, same conditions as before:
W=32, H=128, SKIP_SLACK=8/default, prod_depth=2/default, cat_cons_depth=2/default).
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

ALREADY_TRACED = {"conv_1", "b1c1", "b1c2", "b3c3", "b6c1", "conv_2", "conv_cat", "up"}
ALL_STAGES = NL.stage_names("up")
TODO = [s for s in ALL_STAGES if s not in ALREADY_TRACED]
print(f"[all-stages] {len(TODO)} untraced stages: {TODO}", flush=True)

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

mlir_saved = False
results = {}
for stage in TODO:
    trace_txt = OUT_DIR / f"stall_allstages_{stage}.txt"
    trace_json = OUT_DIR / f"stall_allstages_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"allst_{stage}",
                     tag=f"allstages{stage}", trace_stages=[stage], trace_config=tc,
                     coretile_events=EVENTS, egress_shim_col=1)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((TRACE_H * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not mlir_saved and tc.physical_mlir_path:
        mlir_dst = OUT_DIR / "span_net_physical.mlir"
        mlir_dst.write_text(Path(tc.physical_mlir_path).read_text())
        print(f"[all-stages] saved physical MLIR (from {stage}'s build) -> {mlir_dst}", flush=True)
        mlir_saved = True
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [{stage}] EMPTY TRACE, skipping", flush=True)
        results[stage] = None
        continue
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, stage)
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"stall_allstages_{stage}_summary.json").write_text(json.dumps(res, indent=2))
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
