#!/usr/bin/env python3
"""Phase 1e targeted experiment: the join's MemTile is shared with weight group 3 (TRACE_RESULTS.md
Phase 1a, "MemTile(4,1) is the shared resource" -- 10/12 DMA channels used there vs. 7/12 on every
other weight-group MemTile), and the pace-jump bisection (trace_span_upto_bisect.py) now shows the
whole-net pace steps from ~326 to ~483 cyc/px exactly when the join enters (conv_2, no join, 20
cores: 323.3; conv_cat, +join, 21 cores: 481.3) -- so the join is where to look, not block 1.

Experiment: net_design.build(join_tile=...) pins the join off the shared MemTile onto a dedicated
one (compile_span_join_tile.py already confirmed Tile(1,1) and Tile(5,1) both fit L1/compile clean).
Same-process trace of conv_cat and b1c2 (both present at every prefix), full net (upto=up), W=32,
H=128, one stage per dispatch, baseline (AnyMemTile, lands on (4,1)) vs. both candidates.
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
from aie.iron.device import Tile  # noqa: E402
from aie.utils.trace.config import TraceConfig  # noqa: E402
from aie.utils.trace.events import CoreEvent  # noqa: E402
import aie.utils as aie_utils  # noqa: E402
import net_design as N  # noqa: E402
import net_layout as NL  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

sys.path.insert(0, str(HERE))
from trace_span_net import summarize  # noqa: E402

WIDTH, TRACE_H, CLOCK, UPTO = 32, 128, 1.8e9, "up"
STAGES = ["conv_cat", "b1c2"]
CASES = [("baseline_AnyMemTile", None), ("tile1_1", Tile(1, 1)), ("tile5_1", Tile(5, 1))]
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
names = NL.stage_names(UPTO)
kind = dict(NL.STAGES)
x_row = NL.layout("conv1", WIDTH).in_bytes
y_row = NL.layout(kind[UPTO], WIDTH).out_bytes


def trace_stage(label, jt, stage):
    trace_txt = OUT_DIR / f"phase1e_jointile_{label}_{stage}.txt"
    trace_json = OUT_DIR / f"phase1e_jointile_{label}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"p1ejt_{label}_{stage}",
                     tag=f"p1ejt{label}{stage}", upto=UPTO, trace_stages=[stage],
                     trace_config=tc, coretile_events=EVENTS, egress_shim_col=1, join_tile=jt)
    x = rng.integers(0, 256, size=(TRACE_H + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((TRACE_H * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [{label}/{stage}] EMPTY TRACE, skipping", flush=True)
        return None
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, f"{label}_{stage}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"phase1e_jointile_{label}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg = (res["compute_cyc_per_px"] or 0) + (res["gap_cyc_per_px"] or 0)
    print(f"  [{label}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  compute+gap={cg:.1f}",
         flush=True)
    return res


print("\n=== Phase 1e join-tile experiment: full net, one stage per dispatch ===", flush=True)
for label, jt in CASES:
    for stage in STAGES:
        trace_stage(label, jt, stage)

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
