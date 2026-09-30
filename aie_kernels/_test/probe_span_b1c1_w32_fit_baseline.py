#!/usr/bin/env python3
"""Baseline-config trace corroboration for probe_span_b1c1_w32_fit.py: that run's wall-clock A/B
(553 vs 561 cyc/px) was too close/noisy to read, but its SAME-PROCESS trace of the new config
landed b1c1/b1c2/b3c3 all within 1 cyc/px of each other at ~483 -- decisively lower than the
610-690 cyc/px range measured in earlier (pre-main-merge) sessions. Traces the CURRENT baseline
(depth=3, no fix) on the full net, same session conventions, so the improvement is read off the
same trustworthy instrument on both sides instead of against a stale cross-session number.
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
rng = np.random.default_rng(0)
names = NL.stage_names("up")
x_row, y_row = NL.layout("conv1", WIDTH).in_bytes, NL.layout("up", WIDTH).out_bytes
x = rng.integers(0, 256, size=(TRACE_H + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)


def trace_stage(stage, build_kw, tag):
    trace_txt = OUT_DIR / f"w32base_{tag}_{stage}.txt"
    trace_json = OUT_DIR / f"w32base_{tag}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"w32basetr_{tag}_{stage}",
                     tag=f"w32basetr{tag}{stage}", trace_stages=[stage], trace_config=tc,
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
    (OUT_DIR / f"w32base_{tag}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
    print(f"  [{tag}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  compute+gap={cg:.1f}",
         flush=True)
    return res


print("\n=== Same-process trace at the CURRENT BASELINE (depth=3, no fix) ===", flush=True)
for stage in ("b1c1", "b1c2", "b3c3"):
    trace_stage(stage, {}, "baseline")

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
