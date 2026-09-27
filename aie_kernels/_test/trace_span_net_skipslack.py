#!/usr/bin/env python3
"""Corroborates probe_span_net_skipslack.py's wall-clock dose-response with a within-dispatch
trace: b3c3's compute/gap/LOCK_STALL at SKIP_SLACK=2 (baseline) vs. 25 (max that fits the 512 KB
MemTile join ring), same W=32, H=128, one process. If the skip-ring throttle hypothesis is right,
b3c3's LOCK_STALL should collapse toward its own isolated compute (~406-433 cyc/px) at slack=25.
See designs/span_sr/TRACE_RESULTS.md.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa: E402
from aie.utils.trace.config import TraceConfig  # noqa: E402
from aie.utils.trace.events import CoreEvent  # noqa: E402
import net_design as N  # noqa: E402
import net_layout as NL  # noqa: E402
import span_int as S  # noqa: E402
import test_span_int as TS  # noqa: E402

sys.path.insert(0, str(HERE))
from trace_span_net import summarize  # noqa: E402

WIDTH, HEIGHT, CLOCK = 32, 128, 1.8e9
SLACKS = (2, 25)
EVENTS = [CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL,
         CoreEvent.MEMORY_STALL, CoreEvent.STREAM_STALL]

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
out_dir = Path(os.environ.get("TRACE_OUT_DIR", "trace-out"))
out_dir.mkdir(parents=True, exist_ok=True)

for sk in SLACKS:
    trace_txt = out_dir / f"trace_span_net_skipslack_b3c3_ss{sk}.txt"
    trace_json = out_dir / f"trace_span_net_skipslack_b3c3_ss{sk}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, HEIGHT, NP, HERE / "gen" / f"trss{sk}", tag=f"trssb3c3{sk}",
                     trace_stages=["b3c3"], trace_config=tc, coretile_events=EVENTS,
                     egress_shim_col=1, skip_slack=sk)
    rng = np.random.default_rng(0)
    x_row, y_row = NL.layout("conv1", WIDTH).in_bytes, NL.layout("up", WIDTH).out_bytes
    x = rng.integers(0, 256, size=(HEIGHT + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, NL.stage_names("up")), dtype=np.int8, device="npu"),
           iron.zeros((HEIGHT * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"skip_slack={sk}: EMPTY TRACE, skipping")
        continue
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, f"b3c3_ss{sk}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    print(f"skip_slack={sk}: compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  "
         f"compute+gap={res['compute_cyc_per_px'] + res['gap_cyc_per_px']:.1f}", flush=True)
