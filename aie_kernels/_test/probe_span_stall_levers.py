#!/usr/bin/env python3
"""Phase 1a of the SPAN stall task (2026-09-27-npu-any-game-realtime.md): after SKIP_SLACK
2->8, b3c3 still shows 40% LOCK_STALL at slack=25. Three untested suspects from
designs/span_sr/TRACE_RESULTS.md:

  1. PROD_DEPTH=2 on the four skip sources (conv_1, conv_2, b1c3, b6c1) -- their output
     ObjectFifo has TWO consumers (next main-path core, depth=main_depth=4; and the join's
     MemTile ring, depth=skip_depths[src]) and a broadcast producer can only run as far ahead
     as its SLOWER consumer. Compile-only sweep (compile_sweep_prod_cat.py, not committed --
     scratch) found prod_depth up to 16 fits L1 at W=32, unlike MAIN_DEPTH.
  2. conv_cat's own read-ahead into the join ring (`f_cat.cons(depth=2)`, now `cat_cons_depth`);
     compile-only sweep found depth=3 fits, depth=4 fails L1.
  3. Untraced skip-related stages b6c1 (mid-chain skip source) and conv_2 (feeds the join a
     row ahead of the others) -- do they show the same signature as every other traced stage?

Same-process trace-only A/B (the decisive instrument per TRACE_RESULTS.md's skip-ring section,
cheaper than the wall-clock net-rate fit and not contention-sensitive in its SPLIT even though
the box is shared): b3c3 retraced at each lever's default vs. its most permissive compiling
value, W=32, H=128, SKIP_SLACK=8 (current default) held fixed throughout.
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
OUT_DIR = Path("/mnt/data/xdna/traces/span")
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
    trace_txt = OUT_DIR / f"stall_{tag}_{stage}.txt"
    trace_json = OUT_DIR / f"stall_{tag}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"stalltr_{tag}_{stage}",
                     tag=f"stalltr{tag}{stage}", trace_stages=[stage], trace_config=tc,
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
    (OUT_DIR / f"stall_{tag}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    print(f"  [{tag}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  "
         f"compute+gap={res['compute_cyc_per_px'] + res['gap_cyc_per_px']:.1f}", flush=True)
    return res


print("\n=== Step 1: untraced skip-related stages, defaults "
     "(prod_depth=2, cat_cons_depth=2, skip_slack=8) ===", flush=True)
for stage in ("b6c1", "conv_2"):
    trace_stage(stage, {}, "base")

print("\n=== Step 2: PROD_DEPTH A/B on b3c3 (2 -> 8) ===", flush=True)
for pd in (2, 8):
    trace_stage("b3c3", {"prod_depth": pd}, f"pd{pd}")

print("\n=== Step 3: cat_cons_depth A/B on b3c3 (2 -> 3) ===", flush=True)
for cd in (2, 3):
    trace_stage("b3c3", {"cat_cons_depth": cd}, f"cd{cd}")

print("\n=== Step 4: PROD_DEPTH=8 + cat_cons_depth=3 combined, b3c3 ===", flush=True)
trace_stage("b3c3", {"prod_depth": 8, "cat_cons_depth": 3}, "combo")

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
