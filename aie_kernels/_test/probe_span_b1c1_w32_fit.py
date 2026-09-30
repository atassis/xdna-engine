#!/usr/bin/env python3
"""Coordinator's fitted lever combo for b1c1<->b1c2 depth=4 at FULL production width (W=32):
`depths={"b1c1": 4}` alone overflows tile (0,3) by 2112 B (see TRACE_RESULTS.md's byte-budget
section). Two cheap, measured (not guessed) fixes close it with 192 B to spare, no LUT/weight
change needed:

  1. `data_sizes={"b1c1": 4160}` -- the LUT's own measured static-data size (aiecc's
     `checkDataSizeRequirements`), explicitly reserved so automatic buffer placement does not
     starve it.
  2. `skip_cons_depths={"conv_1": 3}` -- conv_1's own broadcast-fifo consumer depth into b1c1
     (normally hardcoded to main_depth=4) dropped to 3, saving one 2304 B buffer slot on tile
     (0,3). conv_1's own compute is only ~40 cyc/px (measured), so a zero-slack alternation
     between conv_1 and b1c1 gives a ~360 cyc/px ceiling on that link alone -- comfortably under
     the ~406-433 cyc/px gate rate, so it should not become the new bottleneck (checked below by
     tracing b1c1 itself, not assumed).

Full-net (upto=up) W=32 A/B (baseline vs. this combo), alternated per height, >=3 heights,
medians -- then same-process trace of b1c1/b1c2/b3c3 at the new config.
"""
import json
import os
import subprocess
import sys
import time
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

WIDTH, HEIGHTS, TRIALS, CLOCK = 32, [64, 128, 192, 256], 5, 1.8e9
EVENTS = [CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL,
         CoreEvent.MEMORY_STALL, CoreEvent.STREAM_STALL]
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "lib"))
from data_root import XDNA_DATA  # noqa: E402
OUT_DIR = XDNA_DATA / "traces" / "span"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_KW = {"depths": {"b1c1": 4}, "data_sizes": {"b1c1": 4160},
         "skip_cons_depths": {"conv_1": 3}}

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

print("\n=== Full-net W=32 A/B: baseline vs. b1c1-depth4 fit combo, alternated per height ===",
     flush=True)
configs = {"baseline": {}, "d4fit": NEW_KW}
designs = {name: {h: N.build(WIDTH, h, NP, HERE / "gen" / f"w32fit_{name}",
                             tag=f"w32fit{name}h{h}", **kw)
                  for h in HEIGHTS}
          for name, kw in configs.items()}
results = {name: {"xs": [], "ys": []} for name in configs}
for h in HEIGHTS:
    x = rng.integers(0, 256, size=(h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    for name in configs:
        design = designs[name][h]
        args = [iron.tensor(x, dtype=np.int8, device="npu"),
               iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
               iron.zeros((h * y_row,), dtype=np.int8, device="npu")]
        design(*args)
        ts = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            design(*args)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        results[name]["xs"].append(h)
        results[name]["ys"].append(med)
        print(f"  [{name}] height {h:4d}: median {med * 1e3:8.3f} ms", flush=True)
for name in configs:
    xs, ys = results[name]["xs"], results[name]["ys"]
    slope, icpt = np.polyfit(xs, ys, 1)
    cyc_px = slope / WIDTH * CLOCK
    print(f"[{name}]: slope {slope * 1e6:.1f} us/row, intercept {icpt * 1e3:.3f} ms -> "
         f"{cyc_px:.0f} cyc/px @ {CLOCK / 1e9:.1f} GHz (W={WIDTH})", flush=True)


def trace_stage(stage, build_kw, tag):
    trace_h = 128
    trace_txt = OUT_DIR / f"w32fit_{tag}_{stage}.txt"
    trace_json = OUT_DIR / f"w32fit_{tag}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, trace_h, NP, HERE / "gen" / f"w32fittr_{tag}_{stage}",
                     tag=f"w32fittr{tag}{stage}", trace_stages=[stage], trace_config=tc,
                     coretile_events=EVENTS, egress_shim_col=1, **build_kw)
    x = rng.integers(0, 256, size=(trace_h + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((trace_h * y_row,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [{tag}/{stage}] EMPTY TRACE, skipping", flush=True)
        return None
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, f"{tag}_{stage}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"w32fit_{tag}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
    print(f"  [{tag}/{stage}] compute {res['compute_cyc_per_px']} cyc/px "
         f"({res['compute_pct_of_span']}%)  gap {res['gap_cyc_per_px']} cyc/px "
         f"({res['gap_pct_of_span']}%)  LOCK_STALL {lock}% of span  compute+gap={cg:.1f}",
         flush=True)
    return res


print("\n=== Same-process trace at the new W=32 config: b1c1, b1c2, b3c3 ===", flush=True)
for stage in ("b1c1", "b1c2", "b3c3"):
    trace_stage(stage, NEW_KW, "d4fit")

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()
