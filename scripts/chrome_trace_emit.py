#!/usr/bin/env python3
"""Chrome Trace Event Format adapter for mlir-aie on-chip hardware traces.

`TraceConfig.trace_to_json()` (mlir-aie/python/utils/trace/parse.py) already emits literal
Chrome Trace JSON, but its `ts` is a monotonic AIE-core CYCLE counter starting at 0 per capture
(not microseconds), and every real capture on disk traces exactly one tile -- NUM_EVENTS=8 caps
one core per run (gemm-traced-window-is-half-the-command). This adapter:

  1. Turns each capture's B/E interval pairs into complete ("X") events in MICROSECONDS, scaled
     by an explicit --clock-hz. The AIE clock is an 8-level DPM ladder, not a constant (see
     decode-perop-aie-clock) -- there is no honest default to bake in here.
  2. Treats each input file as one hardware context (one pid). A single file just gets rescaled
     in place. Two or more files are laid out back-to-back on ONE time axis, separated by an
     explicit --switch-us gap plus a global "context_switch" instant marker, so a hw-context
     transition renders as a visible, labeled gap instead of two captures both starting their
     own clock at cycle 0 and overlapping.
  3. Optionally adds a coarse wall-clock reference row from --clip-seconds NAME=SECONDS (e.g. a
     per-clip `[enc] ... {:.3}s` span that is already real seconds -- ts_us = seconds * 1e6).

Within one hw-context pid, tid rows come straight from the capture's own thread_name metadata
(INSTR_EVENT_*/INSTR_VECTOR = core issue, PORT_RUNNING_* = DMA stream, *_STALL = stall) -- that
already IS the kernel/stream split this project cares about; nothing is invented here. If a
single capture traces more than one tile, its tiles are folded into that same one pid with
tid = "tile / event" rows, because multiple tiles inside one dispatch share one hw context.

    python3 scripts/chrome_trace_emit.py artifacts/trace_gemm_512x1024x1024_64x32x128_4c_modalid.json \\
        --clock-hz 1.8e9 --out artifacts/chrome_trace_gemm.json
"""
import argparse
import json
import sys


def intervals(events):
    """B/E pairs -> {(pid, tid, name): [(start_cycles, end_cycles), ...]}.
    An unclosed B at buffer end is dropped, matching gemm_trace_probe.py's own `intervals()`:
    the trace buffer truncates mid-run and a guessed close would inflate whatever was open."""
    out = {}
    open_at = {}
    for e in events:
        ph = e.get("ph")
        if ph not in ("B", "E") or "ts" not in e:
            continue
        key = (e.get("pid"), e.get("tid"), e.get("name"))
        if ph == "B":
            open_at[key] = e["ts"]
        elif key in open_at:
            out.setdefault(key, []).append((open_at.pop(key), e["ts"]))
    return out


def capture_to_x_events(events, clock_hz, new_pid, label):
    """One mlir-aie trace_to_json capture -> (X events, span_us) on a single logical pid.
    span_us is measured from this capture's own min/max ts, so callers can chain captures
    on a shared timeline without assuming they all started at cycle 0 at the same instant."""
    scale = 1e6 / clock_hz
    proc_names, thread_names = {}, {}
    for e in events:
        if e.get("ph") != "M":
            continue
        if e["name"] == "process_name":
            proc_names[e.get("pid")] = e["args"]["name"]
        elif e["name"] == "thread_name":
            thread_names[(e.get("pid"), e.get("tid"))] = e["args"]["name"]

    ivs_by_key = intervals(events)
    if not ivs_by_key:
        return [], 0.0

    multi_tile = len(proc_names) > 1
    min_ts = min(s for ivs in ivs_by_key.values() for s, _ in ivs)
    max_ts = max(e for ivs in ivs_by_key.values() for _, e in ivs)

    out, seen_tids = [], {}
    for (pid, tid, name), ivs in ivs_by_key.items():
        out_tid = pid * 1000 + tid if multi_tile else tid
        seen_tids[out_tid] = (pid, tid)
        for start, end in ivs:
            out.append({
                "name": name, "ph": "X", "pid": new_pid, "tid": out_tid,
                "ts": round((start - min_ts) * scale, 3),
                "dur": round((end - start) * scale, 3),
                "args": {"cycles": end - start},
            })

    meta = [{"name": "process_name", "ph": "M", "pid": new_pid, "args": {"name": label}}]
    for out_tid, (pid, tid) in sorted(seen_tids.items()):
        tname = thread_names.get((pid, tid), str(tid))
        if multi_tile:
            tname = f"{proc_names.get(pid, pid)} / {tname}"
        meta.append({"name": "thread_name", "ph": "M", "pid": new_pid, "tid": out_tid,
                     "args": {"name": tname}})
    return meta + out, (max_ts - min_ts) * scale


def context_switch_marker(ts_us, from_label, to_label, gap_us):
    """Global instant event at a hw-context boundary -- ph 'i' with scope 'g' draws a marker
    across every row in chrome://tracing, on top of the gap already left in the time axis."""
    return {
        "name": "hw_context_switch", "ph": "i", "s": "g", "ts": round(ts_us, 3),
        "cat": "context_switch",
        "args": {"from": from_label, "to": to_label, "gap_us": round(gap_us, 3)},
    }


def clip_seconds_row(pid, name, seconds):
    """A single coarse wall-clock bar spanning `seconds`, its own top-level pid -- a reference
    to compare the sum of hw-context rows against, not scaled: it is already real seconds."""
    return [
        {"name": "process_name", "ph": "M", "pid": pid, "args": {"name": f"wall-clock: {name}"}},
        {"name": "thread_name", "ph": "M", "pid": pid, "tid": 0, "args": {"name": "clip"}},
        {"name": name, "ph": "X", "pid": pid, "tid": 0, "ts": 0.0,
         "dur": round(seconds * 1e6, 3), "args": {"seconds": seconds}},
    ]


def build(capture_paths, clock_hz, labels, switch_us, clip_seconds):
    trace_events = []
    offset_us = 0.0
    prev_label = None
    for i, path in enumerate(capture_paths):
        events = json.load(open(path))
        label = labels[i] if i < len(labels) else path
        x_events, span_us = capture_to_x_events(events, clock_hz, new_pid=i, label=label)
        for e in x_events:
            if "ts" in e:
                e["ts"] = round(e["ts"] + offset_us, 3)
        trace_events += x_events

        if i > 0:
            gap = switch_us if switch_us is not None else 0.0
            trace_events.append(context_switch_marker(offset_us, prev_label, label, gap))
        offset_us += span_us
        if i + 1 < len(capture_paths):
            offset_us += switch_us if switch_us is not None else 0.0
        prev_label = label

    for j, spec in enumerate(clip_seconds):
        name, seconds = spec.split("=", 1)
        trace_events += clip_seconds_row(pid=-(j + 1), name=name, seconds=float(seconds))

    other = {"generator": "scripts/chrome_trace_emit.py", "clock_hz": clock_hz,
             "sources": capture_paths}
    if len(capture_paths) > 1:
        other["switch_us"] = switch_us
        other["switch_us_source"] = "user-provided" if switch_us is not None else "unmeasured-placeholder (0)"
    return {"traceEvents": trace_events, "otherData": other}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("captures", nargs="+", help="one or more mlir-aie trace_to_json JSON files")
    p.add_argument("--clock-hz", type=float, required=True,
                    help="AIE core clock in Hz used to rescale cycles->us; no default on "
                         "purpose (DPM ladder, not a constant -- see decode-perop-aie-clock)")
    p.add_argument("--out", required=True)
    p.add_argument("--label", action="append", default=[],
                    help="hw-context label per capture, in order (default: the file path)")
    p.add_argument("--switch-us", type=float, default=None,
                    help="gap in microseconds between consecutive captures, i.e. the measured "
                         "hw-context switch tax; if omitted with >1 capture the gap is 0 and "
                         "the output is tagged unmeasured-placeholder rather than guessing")
    p.add_argument("--clip-seconds", action="append", default=[], metavar="NAME=SECONDS",
                    help="add a coarse wall-clock reference row, already in real seconds")
    o = p.parse_args(argv)

    if len(o.label) not in (0, len(o.captures)):
        p.error(f"--label given {len(o.label)} times but {len(o.captures)} captures were passed")
    if len(o.captures) > 1 and o.switch_us is None:
        print("warning: >1 capture merged with no --switch-us; gap set to 0 and tagged "
              "unmeasured-placeholder in otherData", file=sys.stderr)

    doc = build(o.captures, o.clock_hz, o.label, o.switch_us, o.clip_seconds)
    json.dump(doc, open(o.out, "w"))
    n_x = sum(1 for e in doc["traceEvents"] if e.get("ph") == "X")
    print(f"[chrome-trace] {len(o.captures)} capture(s), {len(doc['traceEvents'])} events "
          f"({n_x} complete) -> {o.out}")


if __name__ == "__main__":
    main()
