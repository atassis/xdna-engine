#!/usr/bin/env python3
"""Trace-confirm (or refute) the iterate_bds join tax found by TRACE_RESULTS.md Phase 1e,
with the TRUSTED instrument -- same-process hardware trace of the join CONSUMER core, not
wall-clock (bench_iterate_bds.py's wall-clock A/B flipped sign on the single-source cases, per
this file's own standing caveat that only same-dispatch trace splits are trustworthy).

Isolates the join exactly as bench_iterate_bds.py does (NSRC producers -> one MemTile join pool
-> 1 consumer, SPAN's own row size/depth), but routes the consumer's per-row touch through a
tiny ExternalFunction kernel (touch.cc) bracketed with event0()/event1(), the same convention
net_design.py's _shim(bracket=True) uses for SPAN itself -- pure-Python IRON worker bodies have
no event0()/event1() binding (only ExternalFunction kernel bodies do), so this is the minimum
change needed to make the consumer traceable at all.

CASES: NSRC=4 at SPAN's own join row size (2304 B), depth in {4, 8}. depth=4 fits the 48-BD cap
both ways (2*NSRC*depth <= 48); depth=8 needs 64 BDs per-slot (over the cap) so the OFF arm at
depth=8 is compile-checked, not assumed -- if it fails, only the ON arm is measured there and
that is reported, not silently skipped.

Run (device held under the NPU lock):
  cd aie_kernels/_test && ./run.sh trace_iterate_bds.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyMemTile  # noqa: E402
from aie.iron.kernel import ExternalFunction  # noqa: E402
from aie.helpers.taplib import TensorTiler2D  # noqa: E402
from aie.utils.trace.config import TraceConfig  # noqa: E402
from aie.utils.trace.events import CoreEvent  # noqa: E402
import aie.utils as aie_utils  # noqa: E402

from trace_span_net import summarize  # noqa: E402 -- reuse the pid-filtered instrument

CLOCK = 1.8e9
GEN = HERE / "gen" / "trace_iterate_bds"
GEN.mkdir(parents=True, exist_ok=True)
KERN_SRC = GEN / "touch.cc"
EVENTS = [CoreEvent.INSTR_EVENT_0, CoreEvent.INSTR_EVENT_1, CoreEvent.LOCK_STALL,
         CoreEvent.MEMORY_STALL, CoreEvent.STREAM_STALL]


def ty(n):
    return np.ndarray[(n,), np.dtype[np.int8]]


def write_touch_kernel():
    KERN_SRC.write_text(
        '#include <aie_api/aie.hpp>\n#include <stdint.h>\n'
        'extern "C" void touch_row(int8_t *l0, int8_t *o) {\n'
        '  event0();\n'
        '  o[0] = l0[0];\n'
        '  event1();\n'
        '}\n')
    return KERN_SRC


def build(nsrc, row_bytes, depth, iterate, h, tag, trace_config=None):
    """Same join shape as bench_iterate_bds.py's build(), consumer touch routed through the
    traced ExternalFunction kernel above so the consumer core can carry event0()/event1()."""
    kern_src = write_touch_kernel()

    def design(y: Out, *, trace_size: CompileTime[int] = 0):
        f_cat = ObjectFifo(ty(row_bytes * nsrc), name="cat_in", depth=depth,
                           iterate_bds=iterate)
        subs = f_cat.prod().join(
            [i * row_bytes for i in range(nsrc)],
            obj_types=[ty(row_bytes)] * nsrc, depths=[depth] * nsrc,
            names=[f"src{i}" for i in range(nsrc)], tile=AnyMemTile)
        f_out = ObjectFifo(ty(4), name="res_out", depth=2)

        touch = ExternalFunction("touch_row", source_file=str(kern_src),
                                 arg_types=[ty(row_bytes * nsrc), ty(4)])

        def prod_body(fo, val):
            for _ in range_(h):
                o = fo.acquire(1)
                o[0] = val
                fo.release(1)

        def cons_body(fi, fo, k):
            for _ in range_(h):
                e = fi.acquire(1)
                o = fo.acquire(1)
                k(e, o)
                fi.release(1)
                fo.release(1)

        workers = [Worker(lambda fo, v=i: prod_body(fo, v), fn_args=[subs[i].prod()])
                  for i in range(nsrc)]
        cons_wk = Worker(cons_body, fn_args=[f_cat.cons(depth=2), f_out.prod(), touch])
        workers.append(cons_wk)

        y_tap = TensorTiler2D.group_tiler((h, 4), (1, 4), (h, 1))[0]

        def sequence(y_, hy):
            hy.drain(y_, y_tap, wait=True)

        rt = Runtime(sequence, [ty(h * 4), f_out.cons()])
        prog = Program(iron.get_current_device(), rt, workers=workers)
        if trace_size:
            prog.enable_trace(trace_size=trace_size, workers=[cons_wk],
                              coretile_events=EVENTS, egress_shim_col=1)
        return prog.resolve_program()

    design.__name__ = design.__qualname__ = tag
    return iron.jit(design, use_cache=True, trace_config=trace_config)


def trace_once(nsrc, row_bytes, depth, iterate, h, out_dir, repeat_idx):
    tag = f"tib_n{nsrc}_r{row_bytes}_d{depth}_{'it' if iterate else 'sl'}_h{h}"
    trace_txt = out_dir / f"{tag}.txt"
    trace_json = out_dir / f"{tag}.json"
    tc = TraceConfig(trace_size=1048576, trace_file=str(trace_txt))
    design = build(nsrc, row_bytes, depth, iterate, h, tag, trace_config=tc)
    args = [iron.zeros((h * 4,), dtype=np.int8, device="npu")]
    design(*args)  # first dispatch: compile + warm up
    design(*args)  # repeat_idx-th measured dispatch (fresh trace buffer each call)
    if not trace_txt.exists() or trace_txt.stat().st_size == 0:
        return {"tag": tag, "error": "empty/missing trace file"}
    if not tc.physical_mlir_path:
        return {"tag": tag, "error": "physical_mlir_path never set"}
    tc.trace_to_json(tc.physical_mlir_path, str(trace_json))
    res = summarize(str(trace_json), CLOCK / 1e9, row_bytes, tag)
    res["repeat"] = repeat_idx
    return res


def main():
    out_dir = Path("trace-out-iterate-bds")
    out_dir.mkdir(exist_ok=True)
    H = 128
    REPEATS = 3
    # (nsrc, row_bytes, depth) -- SPAN's own join row size, depth=4 (fits 48-BD cap both ways),
    # depth=8 (needs iterate_bds; OFF is compile-checked, not assumed to fit).
    cases = [(4, 2304, 4), (4, 2304, 8)]

    all_results = {}
    for nsrc, rb, depth in cases:
        for iterate in (True, False):
            key = (nsrc, rb, depth, iterate)
            reps = []
            for r in range(REPEATS):
                # alternate ON/OFF within the repeat loop below in main(), not here; this
                # function is called in an alternated order by the driver loop.
                pass
            all_results[key] = reps

    # Alternate ON/OFF per repeat (box-drift control, same convention as every wall-clock A/B
    # in TRACE_RESULTS.md) rather than running all of one arm then all of the other.
    for nsrc, rb, depth in cases:
        for r in range(REPEATS):
            for iterate in (True, False):
                key = (nsrc, rb, depth, iterate)
                print(f"[trace-iterate-bds] nsrc={nsrc} row={rb} depth={depth} "
                     f"iterate={iterate} repeat={r}", flush=True)
                try:
                    res = trace_once(nsrc, rb, depth, iterate, H, out_dir, r)
                except Exception as e:  # noqa: BLE001 -- record and continue (e.g. depth=8 OFF
                                        # may fail to compile: 48-BD cap), report don't crash
                    res = {"error": str(e), "iterate": iterate, "depth": depth}
                all_results.setdefault(key, []).append(res)
                (out_dir / "all_results.json").write_text(
                    json.dumps({str(k): v for k, v in all_results.items()}, indent=2))

    print("\n=== summary: consumer core compute+gap cyc/row, iterate_bds ON vs OFF ===")
    for nsrc, rb, depth in cases:
        for iterate in (True, False):
            key = (nsrc, rb, depth, iterate)
            vals = [r.get("compute_cyc_per_px", None) for r in all_results[key]
                   if "error" not in r]
            gaps = [r.get("gap_cyc_per_px", None) for r in all_results[key] if "error" not in r]
            errs = [r for r in all_results[key] if "error" in r]
            if vals and gaps:
                tot = [v + g for v, g in zip(vals, gaps)]
                print(f"nsrc={nsrc} row={rb}B depth={depth} iterate={iterate}: "
                     f"compute+gap median {np.median(tot):.1f} cyc/row "
                     f"(n={len(tot)}, raw={tot})")
            if errs:
                print(f"  {len(errs)} error(s): {errs[0].get('error')}")

    if aie_utils.DefaultNPURuntime is not None:
        aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    main()
