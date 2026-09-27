#!/usr/bin/env python3
"""Phase 1h: re-test the SKIP_SLACK dose-response now that block 1 (b1c1<->b1c2) is fixed and the
join is the established sole throttle (Phase 1e/1f/1g). Traces conv_1 (the skip source itself,
rows_ahead=20) and b1c2 (the whole-chain pace probe used throughout Phase 1c/1e/1f/1g) at
SKIP_SLACK in {8, 12, 16, 20, 25}, one stage per dispatch, same process, W=32 H=128.
See designs/span_sr/TRACE_RESULTS.md, Phase 1h.
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
SLACKS = (8, 12, 16, 20, 25)
STAGES = ("conv_1", "b1c2")
EVENTS = [CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL,
         CoreEvent.MEMORY_STALL, CoreEvent.STREAM_STALL]

cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
net = S.Span(os.environ["SPAN_EXPORT"])
net.quantize(cal)
NP = net.net_params()
out_dir = Path(os.environ.get("TRACE_OUT_DIR", "trace-out"))
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for sk in SLACKS:
    for stage in STAGES:
        trace_txt = out_dir / f"trace_span_1h_{stage}_ss{sk}.txt"
        trace_json = out_dir / f"trace_span_1h_{stage}_ss{sk}.json"
        tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
        design = N.build(WIDTH, HEIGHT, NP, HERE / "gen" / f"tr1h_{stage}_{sk}",
                         tag=f"tr1h{stage}{sk}", trace_stages=[stage], trace_config=tc,
                         coretile_events=EVENTS, egress_shim_col=1, skip_slack=sk)
        rng = np.random.default_rng(0)
        x_row, y_row = NL.layout("conv1", WIDTH).in_bytes, NL.layout("up", WIDTH).out_bytes
        x = rng.integers(0, 256, size=(HEIGHT + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
        args = [iron.tensor(x, dtype=np.int8, device="npu"),
               iron.tensor(NL.weights_blob(NP, NL.stage_names("up")), dtype=np.int8, device="npu"),
               iron.zeros((HEIGHT * y_row,), dtype=np.int8, device="npu")]
        design(*args)
        if not trace_txt.exists() or trace_txt.stat().st_size == 0:
            print(f"skip_slack={sk} stage={stage}: EMPTY TRACE, skipping")
            continue
        tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
        res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, f"{stage}_ss{sk}")
        lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
        cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
        rows.append((sk, stage, res["compute_cyc_per_px"], res["gap_cyc_per_px"], lock, cg))
        print(f"skip_slack={sk} stage={stage}: compute {res['compute_cyc_per_px']} cyc/px "
             f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
             f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  "
             f"compute+gap={cg:.1f}", flush=True)

print("\n== summary ==")
for sk, stage, c, g, lock, cg in rows:
    print(f"{sk}\t{stage}\t{c:.2f}\t{g:.2f}\t{lock}\t{cg:.1f}")
