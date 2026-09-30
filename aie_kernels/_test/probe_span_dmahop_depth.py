#!/usr/bin/env python3
"""Coordinator's DMA-hop hypothesis: every main-path hop has depth 4 against a 3-row window (1
free slot). On a SHARED-MEMORY hop (adjacent tiles, same column -- confirmed from the compiled
MLIR: block 1's conv_1/b1c1/b1c2/b1c3 all sit on tile column 0) the freed slot costs no copy. On
a DMA-crossing hop (confirmed from the same MLIR: e.g. b1c3 (0,5) -> b2c1 (1,2) allocates TWO
separate buffer arrays, `b1c3_skip_1_cons_buff_*` on b2c1's OWN tile (1,2) plus `b1c3_skip_buff_*`
on b1c3's tile (0,5) -- unlike the shared-memory case where one array on one tile serves both
sides) the freed slot must be refilled by an actual DMA transfer after the consumer releases a
row, so that transfer latency sits on the critical path every row.

Column-crossing main-path hops (from the physical MLIR's col,row placement, TRACE_RESULTS.md
Phase 1a follow-up table): b1c3->b2c1, b3c1->b3c2, b4c2->b4c3, b5c3->b6c1. b1c3 is also a skip
source (CAT_SOURCES), so its main-path consumer depth is keyed by `skip_cons_depths`, not
`depths` (see net_design.py's `fi = out[prev].cons(skip_cons_depths.get(prev, main_depth) if
prev in skips else depth[prev])`); the other three are plain stages, keyed by `depths`. Compile-
only sweep (this file's sibling shell one-liner) already confirmed depth+1 on all four fits L1 at
W=32. This traces b1c2, b3c3, conv_1 (representative: block-1 shared-mem-only, mid-chain crossing
one DMA hop already, and the network's own DMA-fed head) before/after, same process.

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

DMAHOP_KW = {"depths": {"b3c1": 5, "b4c2": 5, "b5c3": 5}, "skip_cons_depths": {"b1c3": 5}}

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
    trace_txt = OUT_DIR / f"dmahop_{tag}_{stage}.txt"
    trace_json = OUT_DIR / f"dmahop_{tag}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"dmahop_{tag}_{stage}",
                     tag=f"dmahop{tag}{stage}", trace_stages=[stage], trace_config=tc,
                     coretile_events=EVENTS, egress_shim_col=1, **build_kw)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(N.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((TRACE_H * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [{tag}/{stage}] EMPTY TRACE, skipping", flush=True)
        return None
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, f"{tag}_{stage}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"dmahop_{tag}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
    print(f"  [{tag}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  compute+gap={cg:.1f}",
         flush=True)
    return res


print("\n=== baseline (all main-path depths=4, current default) ===", flush=True)
for stage in ("conv_1", "b1c2", "b3c3"):
    trace_stage(stage, {}, "baseline")

print("\n=== DMA-crossing hops +1 consumer depth (b1c3->b2c1, b3c1->b3c2, "
     "b4c2->b4c3, b5c3->b6c1) ===", flush=True)
for stage in ("conv_1", "b1c2", "b3c3"):
    trace_stage(stage, DMAHOP_KW, "dmahop")

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
