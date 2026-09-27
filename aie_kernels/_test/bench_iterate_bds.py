#!/usr/bin/env python3
"""Isolated iterate_bds A/B, per TRACE_RESULTS.md Phase 1e's flagged next step: does the join's
self-looping BD (`XAie_DmaSetBdIteration`, one physical BD re-armed per firing) carry a per-row
tax over the ordinary per-slot BD chain (a physical BD per depth slot, pre-resident in the BD
table)? Locks are ruled out already (mlir-aie AIEObjectFifoLowerDMAs.cpp:143-150: both lowerings
acquire/release 1 per firing, one lock pair per source -- see the coordinator report).

Uses `ObjectFifo.join()` with NSRC producers -> one MemTile pool -> 1 consumer, exactly SPAN's own
join mechanism (net_design.py's `f_cat.prod().join(...)`), but with a trivial 1-byte-touch kernel
(no C++ shim) so the whole per-row cost is DMA/lock, not compute -- isolates the join transport
itself from conv_cat's own arithmetic.

NSRC=1 is (a): single producer -> MemTile -> consumer, at SPAN's own join row sizes (11520 B,
2304 B) and depth in {4, 8}. NSRC=4 is (b): a real 4-way join at a depth that fits the 48-BD cap
both ways (2*NSRC*depth <= 48 => depth<=6 for NSRC=4; the ORIGINAL commit measured "182 BDs" at
NSRC=4,depth=22, i.e. ~2*NSRC*depth, consistent).

Method: build() below is `net_design.build`-shaped (an `iron.jit`-wrapped design fn), varied over
ITERATE (True/False) and H (row count), wall-clock fitted-slope over several H, TRIALS medians
per H -- the same method as probe_span_net_skipslack.py, alternated per H across the two ITERATE
arms to control for box drift.

Run: cd aie_kernels/_test && SPAN_EXPORT=... SPAN_DEMO_DIR=... NPU_LOCK_SH=... NPU_WAIT_S=1800 \
    XRT_INI_PATH=... ./run.sh bench_iterate_bds.py
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "designs" / "span_sr"))
import aie.iron as iron  # noqa: E402
from aie.iron import CompileTime, ObjectFifo, Out, Program, Runtime, Worker  # noqa: E402
from aie.iron.controlflow import range_  # noqa: E402
from aie.iron.device import AnyMemTile  # noqa: E402
from aie.helpers.taplib import TensorTiler2D  # noqa: E402

CLOCK = 1.8e9
GEN = HERE / "gen" / "bench_iterate_bds"
GEN.mkdir(parents=True, exist_ok=True)


def ty(n):
    return np.ndarray[(n,), np.dtype[np.int8]]


def build(nsrc, row_bytes, depth, iterate, h, tag):
    """nsrc producers, each writing `row_bytes` int8 rows for h rows, joined on one MemTile
    pool (ObjectFifo(..., iterate_bds=iterate)) into one consumer that touches 1 byte/row and
    drains it to host. No host input: producers write a compile-time constant, so the whole
    design has one shim channel (the h-byte result out), isolating the join's own DMA/lock cost.
    """

    def design(y: Out, *, trace_size: CompileTime[int] = 0):
        f_cat = ObjectFifo(ty(row_bytes * nsrc), name="cat_in", depth=depth,
                           iterate_bds=iterate)
        subs = f_cat.prod().join(
            [i * row_bytes for i in range(nsrc)],
            obj_types=[ty(row_bytes)] * nsrc, depths=[depth] * nsrc,
            names=[f"src{i}" for i in range(nsrc)], tile=AnyMemTile)
        f_out = ObjectFifo(ty(4), name="res_out", depth=2)  # dma_bd needs len % 4 == 0

        def prod_body(fo, val):
            for _ in range_(h):
                o = fo.acquire(1)
                o[0] = val
                fo.release(1)

        def cons_body(fi, fo):
            for _ in range_(h):
                e = fi.acquire(1)
                o = fo.acquire(1)
                o[0] = e[0]
                fi.release(1)
                fo.release(1)

        workers = [Worker(lambda fo, v=i: prod_body(fo, v), fn_args=[subs[i].prod()])
                  for i in range(nsrc)]
        workers.append(Worker(cons_body, fn_args=[f_cat.cons(depth=2), f_out.prod()]))

        y_tap = TensorTiler2D.group_tiler((h, 4), (1, 4), (h, 1))[0]

        def sequence(y_, hy):
            hy.drain(y_, y_tap, wait=True)

        rt = Runtime(sequence, [ty(h * 4), f_out.cons()])
        prog = Program(iron.get_current_device(), rt, workers=workers)
        return prog.resolve_program()

    design.__name__ = design.__qualname__ = tag
    return iron.jit(design, use_cache=True)


def measure(nsrc, row_bytes, depth, iterate, heights, trials=5):
    tag = f"bib_n{nsrc}_r{row_bytes}_d{depth}_{'it' if iterate else 'sl'}"
    print(f"[bench] building {tag} for heights {heights}...", flush=True)
    designs = {h: build(nsrc, row_bytes, depth, iterate, h, f"{tag}h{h}") for h in heights}
    xs, ys = [], []
    for h in heights:
        design = designs[h]
        args = [iron.zeros((h * 4,), dtype=np.int8, device="npu")]
        design(*args)  # warm up / first-dispatch compile cost
        ts = []
        for _ in range(trials):
            t0 = time.perf_counter()
            design(*args)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        xs.append(h)
        ys.append(med)
        print(f"  {tag} h={h:4d}: median {med * 1e3:8.3f} ms", flush=True)
    slope, icpt = np.polyfit(xs, ys, 1)
    cyc_row = slope * CLOCK
    print(f"[bench] {tag}: slope {slope * 1e6:.2f} us/row, intercept {icpt * 1e3:.3f} ms -> "
         f"{cyc_row:.1f} cyc/row @ {CLOCK / 1e9:.1f} GHz", flush=True)
    return cyc_row


HEIGHTS = [100, 200, 300, 400]

CASES_A = [(1, rb, d) for rb in (11520, 2048) for d in (4, 8)]
CASES_B = [(4, 2304, 4)]

if __name__ == "__main__":
    results = {}
    print("\n=== (a) single producer -> MemTile -> consumer, iterate_bds on vs off ===", flush=True)
    print("=== (b) 4-producer join -> 1 consumer, depth=4 (fits 48-BD cap both ways) ===", flush=True)
    for nsrc, rb, d in CASES_A + CASES_B:
        for it in (True, False):  # alternate on/off per case to control for box drift
            results[(nsrc, rb, d, it)] = measure(nsrc, rb, d, it, HEIGHTS)

    print("\n=== summary: cyc/row, iterate_bds on vs off, and the delta ===")
    for nsrc, rb, d in CASES_A + CASES_B:
        on = results[(nsrc, rb, d, True)]
        off = results[(nsrc, rb, d, False)]
        print(f"nsrc={nsrc} row={rb:6d}B depth={d}: iterate_bds ON {on:8.1f} cyc/row  "
             f"OFF {off:8.1f} cyc/row  delta {on - off:+8.1f} ({(on / off - 1) * 100:+.1f}%)")
