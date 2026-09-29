#!/usr/bin/env python3
"""INT8_BLOCK1.md steps 3-4, one device session: gate both b1_modes on the full net (22 cores,
W=32), then same-process trace b1c2 and b1c3 (gate) one stage per dispatch, int16 vs int8.
Power mode printed, not pinned (npu_power_mode.py, same convention as every other span probe).
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

WIDTH, HEIGHT, TRACE_H, CLOCK = 32, 64, 128, 1.8e9
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

cal, test = TS.load_split(os.environ["SPAN_DEMO_DIR"])


def gate(b1_mode):
    b1_int8 = b1_mode == "int8"
    net = S.Span(os.environ["SPAN_EXPORT"], b1_mode=b1_mode)
    net.quantize(cal)
    crop = test[:, :HEIGHT, :WIDTH]
    T = net.int_tensors(crop)
    NP = net.net_params()
    names = NL.stage_names("up")
    x = NL.conv1_rows(crop, net.mean255).reshape(-1).view(np.int8)
    xt = iron.tensor(x, dtype=np.int8, device="npu")
    wt = iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu")
    out_bytes = NL.layout(NL.stages(b1_int8)["up"], WIDTH).out_bytes
    yt = iron.zeros((HEIGHT * out_bytes,), dtype=np.int8, device="npu")
    design = N.build(WIDTH, HEIGHT, NP, HERE / "gen" / f"b1int8gate_{b1_mode}",
                     tag=f"b1int8gate{b1_mode}", b1_int8=b1_int8)
    design(xt, wt, yt)
    got = NL.unpack_out(yt.numpy(), "up", HEIGHT, WIDTH, b1_int8=b1_int8).astype(np.int64)
    ref = T[NL.golden(b1_int8)["up"][0]].astype(np.int64)
    mism = int((got != ref).sum())
    print(f"[gate b1_mode={b1_mode}] up (22 cores), {WIDTH}x{HEIGHT}: exact "
         f"{ref.size - mism}/{ref.size} -> {'PASS' if mism == 0 else 'FAIL'}", flush=True)
    return mism == 0, NP


def trace_stage(b1_mode, NP, stage):
    b1_int8 = b1_mode == "int8"
    trace_txt = OUT_DIR / f"b1int8_{b1_mode}_{stage}.txt"
    trace_json = OUT_DIR / f"b1int8_{b1_mode}_{stage}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = N.build(WIDTH, TRACE_H, NP, HERE / "gen" / f"b1int8tr_{b1_mode}_{stage}",
                     tag=f"b1int8tr{b1_mode}{stage}", b1_int8=b1_int8, trace_stages=[stage],
                     trace_config=tc, coretile_events=EVENTS, egress_shim_col=1)
    x_row = NL.layout("conv1", WIDTH).in_bytes
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(TRACE_H + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    names = NL.stage_names("up")
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, names), dtype=np.int8, device="npu"),
           iron.zeros((TRACE_H * NL.layout("up", WIDTH).out_bytes,), dtype=np.int8, device="npu")]
    design(*args)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        print(f"  [{b1_mode}/{stage}] EMPTY TRACE, skipping", flush=True)
        return None
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, WIDTH, f"{b1_mode}_{stage}")
    lock = res["by_event"].get("LOCK_STALL", {}).get("pct_of_span")
    (OUT_DIR / f"b1int8_{b1_mode}_{stage}_summary.json").write_text(json.dumps(res, indent=2))
    cg = res["compute_cyc_per_px"] + res["gap_cyc_per_px"]
    print(f"  [{b1_mode}/{stage}] compute {res['compute_cyc_per_px']} cyc/px  "
         f"gap {res['gap_cyc_per_px']} cyc/px  LOCK_STALL {lock}% of span  compute+gap={cg:.1f}",
         flush=True)
    return res


ok_all = True
NPs = {}
for mode in ("int16", "int8"):
    ok, NP = gate(mode)
    ok_all &= ok
    NPs[mode] = NP

print("\n=== same-process trace: b1c2, b1c3 (gate), int16 vs int8 ===", flush=True)
for mode in ("int16", "int8"):
    for stage in ("b1c2", "b1c3"):
        trace_stage(mode, NPs[mode], stage)

if aie_utils.DefaultNPURuntime is not None:
    aie_utils.DefaultNPURuntime.cleanup()

sys.exit(0 if ok_all else 1)
