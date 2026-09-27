#!/usr/bin/env python3
"""Hardware trace of the whole SPAN x2 net (22 cores, one design): per-core compute vs. gap,
to attribute the measured 1528 cyc/px (probe_span_net_rate.py) against the slowest single
core's ~430 cyc/px (probe_span_core_rates.py) -- a 3.5x this instrument is meant to localise.

Mechanism: net_design.build(trace_stages=..., trace_config=...) brackets exactly the named
stages' kernel call with event0()/event1() and calls Program.enable_trace on those Workers only
(same two-half mechanism as aie_kernels/_test/bricklib.py's _build_streamed_traced / this repo's
proven working precedent, designs/codec_block/trace_conv_dispatch.py). event0/event1 give a
DISJOINT per-row compute/gap split (one core, one thread); LOCK_STALL/MEMORY_STALL/STREAM_STALL
are the CORE-side stall proxies -- there is no literal "DMA busy" core event (see
trace_conv_dispatch.py gotcha 4).

Traces at most 2 stages per run (a traced tile needs a free CORE-tile South egress port --
method-profile-a-brick-with-enable-trace.md gotcha 1 -- and 2 cores keeps shim/channel pressure
low against the net's own 5 weight-group + x/y channels).

Usage (device held under the NPU lock):
  python3 trace_span_net.py --stages conv_1,conv_cat --out trace-out
"""
import argparse
import json
import os
import sys
from collections import defaultdict
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

CLOCK = 1.8e9
# Sparse set (mirrors trace_conv_dispatch.py's CORETILE_EVENTS_SPARSE): event0()/event1()
# bracket every row so INSTR_EVENT_0/1 give a per-row compute/gap split; the three stall
# classes are the direct target of the "objectFIFO depth/lock stalls" and "inter-column DMA
# hops" hypotheses in the task.
EVENTS = [CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL,
         CoreEvent.MEMORY_STALL, CoreEvent.STREAM_STALL]
STALL_EVENTS = ("LOCK_STALL", "MEMORY_STALL", "STREAM_STALL")


def intervals(events):
    """Chrome-trace B/E pairs -> {(pid, name): [(start, end), ...]}. Verbatim from
    trace_conv_dispatch.py: an unclosed B at the buffer's end is dropped, not extrapolated."""
    out = defaultdict(list)
    open_at = {}
    for e in events:
        name, ph = e.get("name"), e.get("ph")
        if ph not in ("B", "E") or "ts" not in e:
            continue
        key = (e.get("pid"), name)
        if ph == "B":
            open_at[key] = e["ts"]
        elif key in open_at:
            out[key].append((open_at.pop(key), e["ts"]))
    return out


def union_cycles(ivs):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(ivs):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s) if cur_s is not None else 0


def summarize(trace_json_path, clock_ghz, w, stage):
    ev = json.load(open(trace_json_path))
    ts = [e["ts"] for e in ev if "ts" in e]
    if not ts:
        return {"stage": stage, "events": len(ev), "error": "no timestamped events in trace JSON"}
    span = max(ts) - min(ts)
    iv = intervals(ev)

    by_event = {}
    for (_pid, name), pairs in iv.items():
        d = by_event.setdefault(name, {"cycles": 0, "count": 0, "_pairs": []})
        d["cycles"] += sum(e - s for s, e in pairs)
        d["count"] += len(pairs)
        d["_pairs"].extend(pairs)
    for name, d in by_event.items():
        d["pct_of_span"] = round(100 * d["cycles"] / span, 2) if span else None
        d["mean_cycles"] = round(d["cycles"] / d["count"], 2) if d["count"] else None

    stall_pairs = [p for n in STALL_EVENTS for p in by_event.get(n, {}).get("_pairs", [])]
    stall_union = union_cycles(stall_pairs)
    for d in by_event.values():
        d.pop("_pairs")

    e0 = sorted(e["ts"] for e in ev if e.get("name") == "INSTR_EVENT_0" and e.get("ph") == "B")
    e1 = sorted(e["ts"] for e in ev if e.get("name") == "INSTR_EVENT_1" and e.get("ph") == "B")
    n_iters = min(len(e0), len(e1))
    compute = [b - a for a, b in zip(e0[:n_iters], e1[:n_iters])]
    gaps = [e0[i + 1] - e1[i] for i in range(n_iters - 1)]

    def cyc_ms(c):
        return round(c / (clock_ghz * 1e9) * 1e3, 4)

    total_compute, total_gap = sum(compute), sum(gaps)
    return {
        "stage": stage, "clock_ghz": clock_ghz, "span_cycles": span, "span_ms": cyc_ms(span),
        "n_rows": n_iters, "by_event": by_event,
        "stall_union_cycles": stall_union,
        "stall_union_pct_of_span": round(100 * stall_union / span, 2) if span else None,
        "compute_cyc_per_row": round(total_compute / n_iters, 1) if n_iters else None,
        "compute_cyc_per_px": round(total_compute / n_iters / w, 2) if n_iters else None,
        "gap_cyc_per_row": round(total_gap / max(n_iters - 1, 1), 1) if n_iters > 1 else None,
        "gap_cyc_per_px": round(total_gap / max(n_iters - 1, 1) / w, 2) if n_iters > 1 else None,
        "compute_pct_of_span": round(100 * total_compute / span, 2) if span else None,
        "gap_pct_of_span": round(100 * total_gap / span, 2) if span else None,
    }


def main(o):
    out_dir = Path(o.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = o.stages.split(",")
    if len(stages) > 2:
        sys.exit("trace-span-net: at most 2 traced stages per run (see module docstring)")

    cal, _ = TS.load_split(os.environ["SPAN_DEMO_DIR"])
    net = S.Span(os.environ["SPAN_EXPORT"])
    net.quantize(cal)
    NP = net.net_params()

    trace_txt = out_dir / f"trace_span_net_{'_'.join(stages)}.txt"
    trace_json = out_dir / f"trace_span_net_{'_'.join(stages)}.json"
    tc = TraceConfig(trace_size=o.trace_size, trace_file=str(trace_txt))

    design = N.build(o.width, o.height, NP, HERE / "gen" / "trace_span_net",
                     tag=f"tracespannet{o.height}", trace_stages=stages, trace_config=tc,
                     coretile_events=EVENTS, egress_shim_col=o.egress_shim_col)

    rng = np.random.default_rng(0)
    x_row, y_row = NL.layout("conv1", o.width).in_bytes, NL.layout("up", o.width).out_bytes
    x = rng.integers(0, 256, size=(o.height + 2) * x_row, dtype=np.int64).astype(np.uint8).view(np.int8)
    args = [iron.tensor(x, dtype=np.int8, device="npu"),
           iron.tensor(NL.weights_blob(NP, NL.stage_names("up")), dtype=np.int8, device="npu"),
           iron.zeros((o.height * y_row,), dtype=np.int8, device="npu")]
    design(*args)

    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        sys.exit(f"trace-span-net: empty/missing {trace_txt} -- trace packet never reached the "
                f"shim (try a different --egress-shim-col, or a HIGHER-row stage: gotcha 1, "
                f"method-profile-a-brick-with-enable-trace.md)")
    print(f"[trace-span-net] {trace_txt} {trace_txt.stat().st_size} B", flush=True)
    if not tc.physical_mlir_path:
        sys.exit("trace-span-net: trace_config.physical_mlir_path never set (compile did not run)")
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))

    for i, stage in enumerate(stages):
        stage_json = trace_json if len(stages) == 1 else out_dir / f"trace_span_net_{stage}_only.json"
        # trace_to_json merges all traced tiles into one stream keyed by pid; re-decode per-pid
        # isn't separated by this call, so with 2 stages traced together the summary below
        # reports the MERGED stream unless the caller re-runs single-stage. Documented, not
        # silently wrong: see the printed warning.
        pass
    res = summarize(str(trace_json), o.clock_ghz, o.width, "+".join(stages))
    (out_dir / f"trace_span_net_{'_'.join(stages)}_summary.json").write_text(json.dumps(res, indent=2))
    if len(stages) > 1:
        print("[trace-span-net] WARNING: >1 stage traced in one run merges both cores' events "
             "into one stream (trace_to_json has no per-tile pid split here) -- the per-row "
             "compute/gap split below is only trustworthy for a SINGLE traced stage. Re-run "
             "with one stage at a time to attribute individually.", flush=True)

    print(f"\n=== stages={'+'.join(stages)} span={res.get('span_cycles')} cyc "
         f"{res.get('span_ms')} ms @ {o.clock_ghz} GHz, {res.get('n_rows')} rows ===")
    if "compute_cyc_per_px" in res and res["compute_cyc_per_px"] is not None:
        print(f"  compute: {res['compute_cyc_per_px']} cyc/px  ({res['compute_pct_of_span']}% of span)")
        print(f"  gap:     {res['gap_cyc_per_px']} cyc/px  ({res['gap_pct_of_span']}% of span)")
        print(f"  stall union (LOCK|MEMORY|STREAM): {res['stall_union_pct_of_span']}% of span")
    for name, d in sorted(res.get("by_event", {}).items(), key=lambda kv: -kv[1]["cycles"]):
        print(f"    {name:<16} {d['cycles']:>10} cyc {d['pct_of_span']:>6}%  n={d['count']}")
    print(f"\nwrote {out_dir}/trace_span_net_{'_'.join(stages)}_summary.json")

    if aie_utils.DefaultNPURuntime is not None:
        aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default="conv_1", help="comma-separated stage name(s), max 2")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--out", default=os.environ.get("TRACE_OUT_DIR", "trace-out"))
    p.add_argument("--trace-size", type=int, default=1048576)
    p.add_argument("--egress-shim-col", type=int, default=1)
    p.add_argument("--clock-ghz", type=float, default=1.8)
    main(p.parse_args())
