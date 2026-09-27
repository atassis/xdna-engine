#!/usr/bin/env python3
"""Compile-only (no device) probe for spec 2026-09-28-span-frame-width-design.md's variant B:
"spill long-lead skips to DDR ... shim-fed subfifos into a shallow join." Tests two things at
the IRON/aiecc level, independent of net_design.py's own kernels:

  1. Does ObjectFifoHandle.join() accept a subfifo whose PRODUCER endpoint is a shim (fed by
     the Runtime sequence via .fill(), per ObjectFifo.prod()'s own tile= doc) rather than a
     core -- i.e. can conv_cat's ring be fed by a DDR spill buffer instead of only by a core's
     broadcast?
  2. Does aiecc accept two INDEPENDENT shim channels on the SAME DDR buffer -- one draining a
     core's output into it (write), one filling a join subfifo back out of it later (read) --
     ordered with a TaskGroup wait rather than racing?

Topology: 2 producer cores (dummy passthrough) -> 2 shim WRITE drains into 2 DDR spill buffers
-> (ordered by TaskGroup.finish()) 2 shim READ fills into a 2-way join on one MemTile -> 1
consumer core (dummy combine) -> shim drain to y. No device execution; success = aiecc places
this through address allocation.
"""
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.iron.runtime.taskgroup import TaskGroup
from aie.iron.device import AnyMemTile, AnyShimTile
from aie.helpers.taplib import TensorTiler2D

HERE = Path(__file__).resolve().parent
GEN = HERE / "gen"
GEN.mkdir(exist_ok=True)

ROW = 64      # bytes/row, one aie_api vector width -- keeps the dummy kernels trivial
H = 8         # rows in the strip
LEAD = 2      # rows the join's read trails the write by (variant B's "lead rows later")


def _write_kernel(path, sym, expr):
    path.write_text(
        f'#include <stdint.h>\n'
        f'extern "C" void {sym}(int8_t *in, int8_t *out) {{\n'
        f'  for (int i = 0; i < {ROW}; i++) out[i] = {expr};\n'
        f'}}\n')
    return path


def ty(n):
    return np.ndarray[(n,), np.dtype[np.int8]]


def main():
    p0 = _write_kernel(GEN / "spill_join_p0.cc", "spill_join_p0", "in[i]")
    p1 = _write_kernel(GEN / "spill_join_p1.cc", "spill_join_p1", "in[i]")
    comb = _write_kernel(GEN / "spill_join_comb.cc", "spill_join_comb",
                         f"(int8_t)(in[i] + in[{ROW} + i])")

    def design(x0: In, x1: In, spill0: In, spill1: In, y: Out):
        row_t, cat_t = ty(ROW), ty(2 * ROW)
        k0 = ExternalFunction("spill_join_p0", source_file=str(p0), arg_types=[row_t, row_t])
        k1 = ExternalFunction("spill_join_p1", source_file=str(p1), arg_types=[row_t, row_t])
        kc = ExternalFunction("spill_join_comb", source_file=str(comb), arg_types=[cat_t, row_t])

        x0f = ObjectFifo(row_t, name="x0")
        x1f = ObjectFifo(row_t, name="x1")
        w0f = ObjectFifo(row_t, name="w0", depth=4)   # producer -> shim WRITE (to spill0)
        w1f = ObjectFifo(row_t, name="w1", depth=4)

        def passthrough(fi, fo, k):
            for _ in range_(H):
                e = fi.acquire(1)
                o = fo.acquire(1)
                k(e, o)
                fo.release(1)
                fi.release(1)

        wk0 = Worker(passthrough, fn_args=[x0f.cons(), w0f.prod(), k0])
        wk1 = Worker(passthrough, fn_args=[x1f.cons(), w1f.prod(), k1])

        # The join: its two subfifos' CONSUMER side is wired to cat_in by join() itself; the
        # PRODUCER side is left to the caller -- here bound to a shim (spill READ), never a
        # core, which is the thing under test (spec's "whether an IRON join accepts
        # shim-fed subfifos").
        cat_in = ObjectFifo(cat_t, name="cat_in", depth=4, iterate_bds=True)
        r0f, r1f = cat_in.prod().join(
            [0, ROW], obj_types=[row_t, row_t], depths=[4, 4], names=["r0", "r1"],
            tile=AnyMemTile)

        yf = ObjectFifo(row_t, name="y")

        def combine(fi, fo, k):
            for _ in range_(H):
                e = fi.acquire(1)
                o = fo.acquire(1)
                k(e, o)
                fo.release(1)
                fi.release(1)

        wkc = Worker(combine, fn_args=[cat_in.cons(), yf.prod(), kc])

        row_tap = TensorTiler2D.group_tiler((H, ROW), (1, ROW), (H, 1))[0]

        def sequence(x0_, x1_, sp0_, sp1_, y_, x0h, x1h, w0h, w1h, r0h, r1h, yh):
            tg0 = TaskGroup()
            x0h.fill(x0_, row_tap, group=tg0)
            x1h.fill(x1_, row_tap, group=tg0)
            # Write phase: drain each producer's output to its DDR spill buffer, WAIT for
            # completion (variant B's "core -> shim -> DDR").
            w0h.drain(sp0_, row_tap, wait=True, group=tg0)
            w1h.drain(sp1_, row_tap, wait=True, group=tg0)
            tg0.finish()
            # Read phase: fill the join's two subfifos back FROM the same DDR buffers --
            # variant B's "return lead rows later into a shallow join". LEAD is not encoded
            # here (whole-buffer fill/drain, like every other Runtime call in this repo);
            # a row-granular version would slice row_tap per LEAD-row chunk instead. See
            # FRAME.md for why that gap is left open.
            tg1 = TaskGroup()
            r0h.fill(sp0_, row_tap, group=tg1)
            r1h.fill(sp1_, row_tap, group=tg1)
            yh.drain(y_, row_tap, wait=True, group=tg1)
            tg1.finish()

        rt = Runtime(sequence, [
            ty(H * ROW), ty(H * ROW), ty(H * ROW), ty(H * ROW), ty(H * ROW),
            x0f.prod(), x1f.prod(), w0f.cons(tile=AnyShimTile), w1f.cons(tile=AnyShimTile),
            r0f.prod(tile=AnyShimTile), r1f.prod(tile=AnyShimTile), yf.cons(),
        ])
        prog = Program(iron.get_current_device(), rt, workers=[wk0, wk1, wkc])
        return prog.resolve_program()

    design.__name__ = design.__qualname__ = "span_spill_join_probe"
    return iron.jit(design, use_cache=True)


if __name__ == "__main__":
    try:
        d = main()
        d.compile()
        print("[span_spill_join] OK: shim-fed join subfifo + DDR write-then-read compiles "
             "clean through aiecc address allocation", flush=True)
    except Exception as e:
        print(f"[span_spill_join] FAIL: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
