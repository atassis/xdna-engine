#!/usr/bin/env python3
"""Hardware trace of ONE codec conv dispatch: where does the on-device time actually go?

probe_vec_device_ms.py and probe_device_ms.py settled COMPUTE vs OVERHEAD at the tile-rate level
(vectorising bought 23x, so the scalar core was compute-bound at ITS rate) -- but that comparison
never looked inside a dispatch. Measured separately, one VECTORISED conv dispatch (c_in=128, k=7,
c_out=384, T=64 -- 384 objectFIFO tiles of 897 f32 / 3588 B each) costs 65.6 ms on device, against
a 0.76 ms compute floor (22.02e6 MAC / (16 MAC-lanes x 1.8e9 Hz)) and a 0.026 ms DRAM-bandwidth
floor (1,377,792 B / 52.7e9 B/s). ~65 of the 65.6 ms is neither: this instrument puts a trace on
that dispatch and reports which of {core compute, lock-acquire stall, DMA/stream-interface stall,
unaccounted} the missing ~99% actually is, with cycle-level evidence instead of another rate.

MECHANISM. `bricklib._build_streamed`'s generator declares zero CompileTime[T] params, so handing
`iron.jit(...)` a `trace_config=TraceConfig(...)` (the FAILED attempt, probe_trace_conv.py) makes
`CallableDesign.__call__` inject a `trace_size` compile kwarg the generator's signature has no slot
for -- `CompilableDesign` rejects it before a single MLIR op is built. bricklib.py now carries
`_build_streamed_traced`, a separate generator with `trace_size: CompileTime[int] = 0` wired to
`Program.enable_trace(...)` (see its docstring for the two-half mechanism: enable_trace bakes the
MLIR-side hardware config, iron.jit's trace_config is the host-side buffer alloc/read-back). This
script installs `_traced_build_streamed(...)` onto the `bricklib` module object for the duration of
ONE `window_driver.conv()` call, so window_driver's own windowing/shim-generation code runs
completely unmodified -- what gets traced is bit-identical to a production dispatch, not a
hand-rolled stand-in.

FOUR GOTCHAS hit getting this working (first working trace on this rail):
  1. `trace_config`'s injected compile kwarg is a NAME MATCH against the generator's own
     `CompileTime[T]` parameters (compilabledesign.py's Guard 2-B) -- the generator must declare
     `trace_size` itself; passing `trace_config` to `iron.jit` is not sufficient on its own.
  2. Patch `bricklib._build_streamed` (the module attribute `window_driver._get_design` actually
     calls), not `iron.jit` -- the failed attempt patched `iron.jit`, which IS reachable, but by
     the time it runs the generator being wrapped is already `_build_streamed`'s (param-less) one;
     injecting trace_config there is exactly what trips gotcha 1.
  3. `input_with_addresses.mlir` (what `TraceConfig.trace_to_json` needs as `mlir_file`) is written
     unconditionally whenever `aiecc` gets a `work_dir` (compile/utils.py: `--get-input-with-
     addresses` is appended whenever `--tmpdir` is) -- no extra `aiecc_flags` needed, but the path
     is `<kernel_dir>/input_with_addresses.mlir` and `CallableDesign` only stamps it onto
     `trace_config.physical_mlir_path` inside `_compile_and_build_kernel`, which runs on the
     FIRST actual dispatch (`design(...)`/`CallableDesign.__call__`) -- an eager `.compile()` call
     alone does NOT set it (verified directly: `.compile()`-only on this exact traced design built
     a real xclbin at both egress_shim_col=0 and =1, but left `physical_mlir_path` `None`). So
     read it after the dispatch, never before, and never rely on `.compile()` alone for it.
  4. There is no CORE-tile event literally named "DMA active". `Program.enable_trace`'s
     `coretile_events` are the CORE's own 8 trace-event monitor slots; genuine DMA/shim-port
     activity is a SEPARATE (untraced here) `shimtile_events`/`PortEvent` axis. STREAM_STALL --
     the core stalled on its own stream-switch/DMA-interface port -- is the closest CORE-side proxy
     for "waiting on the fifo" and is what this script reports under that name; it is not a literal
     DMA-utilization measurement. Extending to `PortEvent`-based shim/memtile tracing is future
     work, not done here (would need device-side verification of the WireBundle/channel mapping
     this script cannot perform).

EVENT SET. `CORETILE_EVENTS` below reuses mlir-aie's own
programming_examples/ml/layernorm/residual_add_iron.py `DENOM_CHECK_CORETILE_EVENTS` verbatim (the
one 8-slot set in this codebase already measured to decode on this box), for two reasons: it
reserves INSTR_EVENT_0/INSTR_EVENT_1, which conv_1d.cc's own `event0()`/`event1()` calls (both
`conv_1d_causal_core` and `conv_1d_causal_core_vec`, read directly off the kernel source, not
assumed) bracket around EACH of the n_tiles per-tile kernel calls -- giving a per-tile-iteration
compute/gap split for free -- and its remaining six slots are exactly the stall/compute categories
the hypothesis needs. Event codes are numerically identical between CoreEventAIE2 (what the
`CoreEvent` alias resolves to) and CoreEventAIE2P (this box's actual Gorgon Point / npu2 silicon)
for all 8 names used here -- checked directly against this instance's `_aie_enum_gen`, not assumed
from the AIE2/AIE2P architectures being similar.

SANITY CHECK (item 3 of the task). The trace's own total captured span, converted to time at the
STATED clock, must land near the wall-clock this SAME run measures around the dispatch call --
disagreement means the trace is measuring something other than the dispatch (truncated buffer, a
stale MLIR, a routing accident). The AIE core clock is an 8-level DPM ladder; ~1.8 GHz is the
canonical MEASURED operating point on this project (not the ~1.53 GHz back-computed marketing
figure), so --clock-ghz defaults to 1.8 and every cycle->ms conversion below names it explicitly.

Usage (main session, device held under the NPU lock -- see bricks/_verify/run.sh for the env this
needs: instance PYTHONPATH, AIECC_PATH, PEANO_INSTALL_DIR):
  python3 trace_conv_dispatch.py --tile-floats 897 --c-out 384 --out $TRACE_OUT_DIR
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bricks" / "_verify"))

import aie.iron as iron
from aie.iron.device import NPU2
from aie.utils.trace.config import TraceConfig
from aie.utils.trace.events import CoreEvent
import aie.utils as aie_utils

import bricklib
import window_driver as wd

# Precedes any compile(): bricklib._build_streamed[_traced] targets iron.get_current_device(),
# which PROBES the runtime (and can hang/error) if nothing is bound yet -- same reasoning as
# export_codec_artifacts.py's own module docstring. NPU2() is a static device descriptor consumed
# by MLIR generation; this line never opens /dev/accel0.
iron.set_current_device(NPU2())

# See the module docstring's EVENT SET section: verbatim DENOM_CHECK_CORETILE_EVENTS from
# residual_add_iron.py.
CORETILE_EVENTS = [
    CoreEvent.INSTR_EVENT_0,
    CoreEvent.INSTR_EVENT_1,
    CoreEvent.INSTR_VECTOR,
    CoreEvent.LOCK_STALL,
    CoreEvent.MEMORY_STALL,
    CoreEvent.STREAM_STALL,
    CoreEvent.INSTR_LOAD,
    CoreEvent.INSTR_STORE,
]
# STREAM_STALL is this script's proxy for "core stalled waiting on the fifo" -- see gotcha 4.
STALL_EVENTS = ("LOCK_STALL", "MEMORY_STALL", "STREAM_STALL")


def _install_traced_builder(trace_config, egress_shim_col):
    """Rebind `bricklib._build_streamed` (the module attribute `window_driver._get_design`
    resolves at CALL time) to the traced builder, closing over `trace_config`/`egress_shim_col`.
    Returns the ORIGINAL so the caller can restore it -- this process only ever runs one traced
    dispatch, but leaving the module patched is a footgun for anything imported after this.
    """
    orig = bricklib._build_streamed

    def _build_traced(symbol, shim, n_tiles, in_tile, out_tile, resident_len, compile_flags,
                      in_dt, out_dt, resident_dt, resident_depth=2, stack_size=None):
        return bricklib._build_streamed_traced(
            symbol, shim, n_tiles, in_tile, out_tile, resident_len, compile_flags,
            in_dt, out_dt, resident_dt, trace_config, resident_depth=resident_depth,
            stack_size=stack_size, coretile_events=CORETILE_EVENTS,
            egress_shim_col=egress_shim_col)

    bricklib._build_streamed = _build_traced
    return orig


def intervals(events):
    """Chrome-trace B/E pairs -> {(pid, name): [(start, end), ...]}.

    Adapted from xdna-engine/scripts/gemm_trace_probe.py's `intervals()` (same repo, same
    convention -- see docs/where-time-goes.md for the broader accounting style this mirrors): an
    unclosed B at the end of the buffer is dropped, not extrapolated, since the trace buffer
    truncates mid-run and a guessed tail would silently inflate whatever event was open.
    """
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
    """Cycles covered by at least one interval -- the honest total when intervals can overlap
    (e.g. LOCK_STALL and MEMORY_STALL both open in the same cycle). Same routine as
    gemm_trace_probe.py's."""
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


def summarize(trace_json_path, clock_ghz):
    """Cycle-level occupancy summary for a ONE-core, ONE-dispatch trace capture.

    Two complementary views, both from the same event stream:
      * per-event totals (cycles/pct_of_span/count/mean) across the whole capture, same shape as
        gemm_trace_probe.py's `summarize()` -- can OVERLAP (a cycle can be inside more than one
        state), so these do not sum to the span.
      * per-TILE-ITERATION compute/gap split, from event0()/event1() pairs -- DISJOINT by
        construction (one core, one thread, one kernel call at a time), so this partition is the
        trustworthy one for "how much of the span was this tile's own compute".
    """
    ev = json.load(open(trace_json_path))
    ts = [e["ts"] for e in ev if "ts" in e]
    if not ts:
        return {"events": len(ev), "error": "no timestamped events in trace JSON"}
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
    # Gap i is everything the core loop does OUTSIDE the kernel call between tile i and i+1:
    # objectFifo acquire/release plus whatever DMA/lock wait that entails -- the direct measure
    # of the hypothesis under test.
    gaps = [e0[i + 1] - e1[i] for i in range(n_iters - 1)]
    lead = (e0[0] - min(ts)) if e0 else None
    trail = (max(ts) - e1[-1]) if e1 else None

    total_compute = sum(compute)
    total_gap = sum(gaps) + (lead or 0) + (trail or 0)

    def cyc_ms(c):
        return round(c / (clock_ghz * 1e9) * 1e3, 4)

    return {
        "clock_ghz": clock_ghz,
        "span_cycles": span,
        "span_ms": cyc_ms(span),
        "by_event": by_event,
        "stall_union_cycles": stall_union,   # LOCK_STALL | MEMORY_STALL | STREAM_STALL, de-duped
        "stall_union_pct_of_span": round(100 * stall_union / span, 2) if span else None,
        "n_tile_iterations": n_iters,
        "compute": {
            "total_cycles": total_compute,
            "total_ms": cyc_ms(total_compute),
            "pct_of_span": round(100 * total_compute / span, 2) if span else None,
            "min_cycles": min(compute) if compute else None,
            "mean_cycles": round(sum(compute) / len(compute), 2) if compute else None,
            "max_cycles": max(compute) if compute else None,
        },
        "gap": {  # per the module docstring: acquire/release + lock/DMA wait, NOT kernel compute
            "total_cycles": total_gap,
            "total_ms": cyc_ms(total_gap),
            "pct_of_span": round(100 * total_gap / span, 2) if span else None,
            "mean_cycles": round(sum(gaps) / len(gaps), 2) if gaps else None,
            "max_cycles": max(gaps) if gaps else None,
            "lead_cycles": lead,
            "trail_cycles": trail,
        },
    }


def main(o):
    out_dir = Path(o.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (o.tile_floats - 1) % o.k != 0:
        sys.exit(f"--tile-floats {o.tile_floats} is not c_in*{o.k}+1 for any integer c_in "
                 f"(pass --k to match a different kernel width)")
    c_in = (o.tile_floats - 1) // o.k
    c_out = o.c_out
    tag = f"trctile{c_in}k{o.k}co{c_out}"

    trace_txt = out_dir / f"trace_conv_dispatch_{tag}.txt"
    trace_json = out_dir / f"trace_conv_dispatch_{tag}.json"
    summary_path = out_dir / f"trace_conv_dispatch_{tag}_summary.json"

    tc = TraceConfig(trace_size=o.trace_size, trace_file=str(trace_txt))
    orig_build = _install_traced_builder(tc, o.egress_shim_col)
    try:
        # L=T chosen so the outer window loop in window_driver._conv_chunk runs exactly ONCE:
        # ctx=(k-1)*dilation, step=T-ctx, M=L-ctx=T-ctx=step -- range(0, step, step) is one
        # iteration -- matching probe_head_conv_threshold.py's own established recipe for
        # isolating a single dispatch. ci_chunk=c_in keeps the outer channel-chunk loop to one
        # iteration too. VERIFIED device-free (window_driver.BUILD_ONLY=True, no dispatch):
        # L=T+ctx gives wd.stats()['dispatches']==2, not 1 -- an earlier draft of this line had
        # that off-by-one-window bug; L=T is the version that measured dispatches==1.
        L = wd.T
        rng = np.random.default_rng(o.seed)
        x = rng.standard_normal((c_in, L)).astype(np.float32)
        w = (rng.standard_normal((c_out, c_in, o.k)) * 0.02).astype(np.float32)
        bias = np.zeros(c_out, np.float32)

        print(f"[trace-conv] c_in={c_in} k={o.k} c_out={c_out} T={wd.T} tile_floats={o.tile_floats} "
             f"CONV_VEC={wd.CONV_VEC} trace_size={o.trace_size} egress_shim_col={o.egress_shim_col}",
             flush=True)

        t0 = time.perf_counter()
        wd.conv(x, w, bias, o.k, o.dilation, tag, ci_chunk=c_in, resident_depth=1)
        wall_ms = (time.perf_counter() - t0) * 1e3
        n_dispatches = wd.stats()["dispatches"]
        print(f"[trace-conv] {n_dispatches} dispatch(es), host wall {wall_ms:.2f} ms", flush=True)
        if n_dispatches != 1:
            print(f"[trace-conv] WARNING: expected exactly 1 dispatch, got {n_dispatches} -- "
                 f"the trace covers ALL of them merged, not one; the per-tile split below still "
                 f"holds per iteration but 'ONE dispatch' framing does not.", flush=True)
    finally:
        bricklib._build_streamed = orig_build

    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        sys.exit(f"[trace-conv] empty or missing trace file {trace_txt} -- the trace packet "
                f"never reached the shim (check --egress-shim-col against this design's "
                f"objectFIFO column placement; see bricklib._build_streamed_traced's docstring)")
    print(f"[trace-conv] trace.txt {trace_txt.stat().st_size} B", flush=True)

    if not tc.physical_mlir_path:
        sys.exit("[trace-conv] trace_config.physical_mlir_path was never set -- compile() did not "
                 "run (see gotcha 3 in the module docstring)")
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))

    res = summarize(str(trace_json), o.clock_ghz)
    res.update(tag=tag, c_in=c_in, k=o.k, c_out=c_out, T=wd.T, tile_floats=o.tile_floats,
              host_wall_ms=round(wall_ms, 3), trace_size=o.trace_size)

    if "error" not in res:
        diff_pct = (100 * (res["span_ms"] - o.expected_wall_ms) / o.expected_wall_ms
                   if o.expected_wall_ms else None)
        res["sanity_check"] = {
            "assumed_clock_ghz": o.clock_ghz,
            "trace_span_ms": res["span_ms"],
            "host_wall_ms": round(wall_ms, 3),
            "reference_wall_ms": o.expected_wall_ms,
            "pct_diff_vs_reference": round(diff_pct, 1) if diff_pct is not None else None,
            "note": ("reference_wall_ms is the 2026-08-31 measurement at the DEFAULT shape "
                    "(tile_floats=897, c_out=384); only meaningful when this run used those "
                    "defaults -- compare trace_span_ms against host_wall_ms (THIS run, any "
                    "shape) for the real self-consistency check."),
        }

    summary_path.write_text(json.dumps(res, indent=2))
    print(f"\n=== {tag}: span={res.get('span_cycles')} cyc = {res.get('span_ms')} ms "
         f"@ {o.clock_ghz} GHz ===")
    if "sanity_check" in res:
        sc = res["sanity_check"]
        print(f"  host wall (this run):  {sc['host_wall_ms']} ms")
        print(f"  trace span (this run): {sc['trace_span_ms']} ms  "
             f"({'AGREES' if abs(sc['trace_span_ms'] - sc['host_wall_ms']) / sc['host_wall_ms'] < 0.25 else 'DISAGREES -- trace may be measuring something else'} "
             f"with host wall, tolerance 25%)")
        if sc["pct_diff_vs_reference"] is not None:
            print(f"  vs 2026-08-31 reference {sc['reference_wall_ms']} ms: "
                 f"{sc['pct_diff_vs_reference']:+.1f}%")
    if "compute" in res:
        c, g = res["compute"], res["gap"]
        print(f"  {res['n_tile_iterations']} tile iterations (event0/event1 pairs)")
        print(f"  compute (kernel body, event0->event1): {c['total_cycles']:>10} cyc "
             f"{c['total_ms']:>8} ms  {c['pct_of_span']:>5}% of span  "
             f"mean/tile={c['mean_cycles']} min={c['min_cycles']} max={c['max_cycles']}")
        print(f"  gap (acquire/release + stall, between tiles): {g['total_cycles']:>10} cyc "
             f"{g['total_ms']:>8} ms  {g['pct_of_span']:>5}% of span  "
             f"mean/tile={g['mean_cycles']} max={g['max_cycles']} "
             f"lead={g['lead_cycles']} trail={g['trail_cycles']}")
        print(f"  stall union (LOCK|MEMORY|STREAM, de-duped): {res['stall_union_cycles']:>10} cyc "
             f"{res['stall_union_pct_of_span']:>5}% of span")
    print("\n  per-event totals (can overlap -- do not sum to span):")
    for name, d in sorted(res.get("by_event", {}).items(), key=lambda kv: -kv[1]["cycles"]):
        print(f"    {name:<16} {d['cycles']:>10} cyc {d['pct_of_span']:>6}%  "
             f"n={d['count']:<5} mean={d['mean_cycles']}")
    print(f"\nwrote {summary_path}")

    if aie_utils.DefaultNPURuntime is not None:
        aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tile-floats", type=int, default=897,
                   help="streamed row width (c_in*k+1); default matches the measured 897/3588B "
                        "tile that motivated this instrument")
    p.add_argument("--c-out", type=int, default=384, help="n_tiles (objectFIFO tile count)")
    p.add_argument("--k", type=int, default=7, help="conv kernel width (must divide tile-floats-1)")
    p.add_argument("--dilation", type=int, default=1)
    p.add_argument("--out", default="$TRACE_OUT_DIR",
                   help="output dir for trace.txt/.json/summary.json")
    p.add_argument("--trace-size", type=int, default=262144, help="host trace buffer, bytes")
    p.add_argument("--egress-shim-col", type=int, default=1,
                   help="see bricklib._build_streamed_traced's docstring -- 1 is the "
                        "proven-on-device precedent value; a compile-only check found 0 also "
                        "compiles cleanly for this shape, so try it if 1 fails on device")
    p.add_argument("--clock-ghz", type=float, default=1.8,
                   help="AIE core clock assumed for cycle->ms; 1.8 is this project's canonical "
                        "MEASURED value, not the ~1.53 GHz datasheet back-computation")
    p.add_argument("--expected-wall-ms", type=float, default=65.6,
                   help="reference wall time for the sanity check, informational only unless "
                        "--tile-floats/--c-out are left at their defaults")
    p.add_argument("--seed", type=int, default=21)
    main(p.parse_args())
