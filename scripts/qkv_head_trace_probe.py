#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hardware trace of ONE fused qkv_head_dp invocation -- where its transport time actually goes.

WHY THIS OP. A depth sweep of the fused decode (bench_layer_arms.py, 5 arms, residuals +/-0.015 ms)
splits the step exactly:

    t = 1.4601 ms/layer * L + 5.886 ms      bytes = 34.7737 MB/layer * L + 311.49 MB

so the per-TOKEN part (the lm_head GEMV) moves its 311.49 MB at 52.92 GB/s -- the measured
52.69 GB/s pure-read fabric ceiling -- while the per-LAYER body moves its 34.77 MB at 23.82 GB/s,
45% of it. Same dispatch, same token, same silicon. qkv_head_dp is the largest single stream in
that layer body (8.389 MB of concatenated Wqkv) and it reproduces the deficit STANDALONE at
~420 us = ~20 GB/s, so the question is answerable on one op without the graph.

The census route is closed: bytes, configures, descriptors, host and compute are each eliminated
with a number (four-configures-a-layer-came-off-without-a-new-kernel). What survives is DMA
transfer LATENCY rather than issue cost, and objectFIFO lock round-trips -- both trace questions.
_default_coretile_events() already carries exactly the events that separate them: PORT_RUNNING_0/1
(the two input DMA channels), PORT_RUNNING_2 (output), LOCK_STALL, MEMORY_STALL, INSTR_EVENT_0/1.

  IRON_TRACE_NTILES=1 python scripts/qkv_head_trace_probe.py [--tsi 4] [--trace-size 65536]

Needs the NPU free (fuser -v /dev/accel/accel0; do NOT stop npu-vox, it is the owner's dictation).
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

import torch

from iron.common import AIEContext
from iron.operators.qkv_head_dp.op import QKVHeadDataParallel
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel
from aie.utils.trace import TraceConfig
import aie.utils as aie_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsi", type=int, default=4)
    ap.add_argument("--trace-size", type=int, default=65536)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--outdir", default="artifacts/trace_qkv_head")
    a = ap.parse_args()

    D, HD, Hq, Hkv, N, S = 1024, 128, 16, 8, a.cols, 512
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    txt, js = str(out / "trace.txt"), str(out / "trace.json")

    # IRON_TRACE_SIZE is what _trace.maybe_enable_trace() reads; set it BEFORE compile so the
    # design is built with the trace tiles wired in. A separate build dir keeps the cached
    # non-traced xclbin from being picked up instead.
    os.environ["IRON_TRACE_SIZE"] = str(a.trace_size)
    os.environ.setdefault("IRON_TRACE_NTILES", "1")
    bd = Path(__file__).resolve().parents[2] / "build" / f"qkv_head_dp_trace_tsi{a.tsi}_S{S}_n{N}"
    op = QKVHeadDataParallel(D=D, HD=HD, Hq=Hq, Hkv=Hkv, max_seq=S, num_aie_columns=N,
                             tile_size_input=a.tsi, kv_offset_parameter=None,
                             context=AIEContext(build_dir=bd))
    print(f"[trace] compiling {op.name} with trace_size={a.trace_size} ...", flush=True)
    op.compile()

    tc = TraceConfig(trace_size=a.trace_size, trace_file=txt)
    k = NPUKernel(xclbin_path=op.xclbin_artifact.filename,
                  kernel_name=op.xclbin_artifact.kernel_name,
                  insts_path=op.insts_artifact.filename,
                  trace_config=tc)

    torch.manual_seed(0)
    args = []
    for s in op.get_arg_spec():
        if s.direction in ("in", "inout"):
            args.append(XRTTensor.from_torch(torch.randn(*s.shape, dtype=torch.bfloat16) * 0.05))
        else:
            args.append(XRTTensor(s.shape, dtype=s.dtype))

    for i in range(a.reps):
        k(*args)
        print(f"[trace] rep {i}: trace.txt {os.path.getsize(txt) if os.path.exists(txt) else 0} B",
              flush=True)

    phys = tc.physical_mlir_path
    print("[trace] physical mlir:", phys)
    tc.trace_to_json(phys, js)
    ev = json.load(open(js))
    print(f"[trace] parsed {len(ev)} events -> {js}")
    summarize(ev)
    aie_utils.DefaultNPURuntime.cleanup()


def summarize(ev):
    """Per-event-name occupancy over the traced span, in CYCLES (clock-free, per the doctrine)."""
    ts = [e["ts"] for e in ev if "ts" in e]
    if not ts:
        print("[trace] no timestamped events"); return
    span = max(ts) - min(ts)
    print(f"\n[trace] span {span} cycles ({min(ts)}..{max(ts)})")

    # Events arrive as start/stop pairs per name; accumulate time-in-state per name.
    openat, total, count = {}, collections.Counter(), collections.Counter()
    for e in sorted(ev, key=lambda x: x.get("ts", 0)):
        nm, ph, t = e.get("name"), e.get("ph"), e.get("ts")
        if nm is None or t is None:
            continue
        if ph == "B":
            openat[nm] = t
        elif ph == "E" and nm in openat:
            total[nm] += t - openat.pop(nm); count[nm] += 1
    if not total:
        names = collections.Counter(e.get("name") for e in ev)
        print("[trace] no B/E pairs; event-name histogram:", dict(names.most_common(15)))
        return
    print(f"\n{'event':32} {'cycles':>10} {'% of span':>10} {'n':>6}")
    for nm, c in total.most_common():
        print(f"{nm:32} {c:10} {100.0*c/span if span else 0:9.1f}% {count[nm]:6}")


if __name__ == "__main__":
    sys.exit(main())
