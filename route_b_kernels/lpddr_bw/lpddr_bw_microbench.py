# lpddr_bw_microbench.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, xdna2-asr-engine.
#
# ===========================================================================
# PURE-DMA LPDDR BANDWIDTH MICROBENCHMARK (no compute, no GEMM tiling).
#
# WHY: the encoder GEMM measured ~57 GB/s *effective* DMA ([[encoder-dma-occupancy]]),
# but that number is contaminated by tiling/dispatch/BD overhead and re-reads. This
# design streams a large CONTIGUOUS buffer L3(LPDDR) <-> L1 through a minimal objectFIFO
# with ZERO arithmetic, so the regression t = c0 + bytes / BW measures the *silicon's
# achievable LPDDR bandwidth* directly -- the number the KB currently has only as a
# datasheet figure (~120 GB/s; optimization-map open gap #2, hw-envelope). The result
# either opens a ~2x headroom lever (if achievable ~= 120) or revises the project's
# "45x above the bandwidth floor" thesis downward (if achievable ~= 60). Both forks are
# high-value, so this is a measure-first gate, not an optimization.
#
# DESIGN (deliberately the smallest dataflow that moves bytes):
#   mode=read  : L3 -> L1.   rt.fill(of.prod(), src, wait=True); a worker drains L1 by
#                acquire/release (NO kernel call -> no compute). Dispatch blocks on the
#                fill DMA completing => times the pure L3->L1 read of `bytes`.
#   mode=write : L1 -> L3.   a worker produces into of (acquire/release, garbage payload);
#                rt.drain(of.cons(), dst, wait=True) times the pure L1->L3 write.
#   mode=rdwr  : L3 -> L1 -> L3 round-trip via ObjectFifo.forward() (this is exactly the
#                upstream passthrough_dmas reference design, known-good). Moves `bytes`
#                each direction concurrently; aggregate = 2*bytes / t.
#
# Multi-column (`--cols C`): declare C independent per-column chains, each with its OWN
# runtime buffer arg and disjoint L3 region; aiecc's place-tiles pass maps them onto C
# columns to run concurrently. Tests whether aggregate LPDDR bandwidth scales with shim
# DMAs (the encoder uses 8 columns). cols=1 is the clean single-stream number; cols=8 the
# aggregate. (Column spread is confirmed post-placement on a healthy fork instance.)
#
# TOOLCHAIN: this targets the FORK place-tiles model (atassis/mlir-aie, the project's single
# blessed toolchain via toolchain.lock / toolchain_up.sh) -- bare resolve_program(), logical
# tiles placed by the compiler pass, NO Python-side placer and NO explicit Tile() pinning.
# Do NOT run this against the stale wheel python (old Python-placer model).
#
# No compute tile arithmetic is emitted in ANY mode: workers only acquire/release the
# objectFIFO (the DMA engines, not the VPU, do all the work). That is the whole point --
# we are measuring the data-movement brick in isolation.
#
# CONFOUNDERS -- why the harness sweeps `--line` (BD/transfer granularity) and `--depth`
# (objectFIFO double/triple-buffering) in ADDITION to transfer size:
#   * Too-small a `line` => many small BDs => per-BD setup dominates and the shim DMA never
#     reaches streaming bandwidth (the same per-transfer overhead the encoder GEMM pays).
#   * Too-shallow a `depth` (e.g. 1) => the DMA stalls waiting for the consumer to release a
#     buffer => the L3<->L1 pipe is not kept full => measured BW under-reports the silicon.
# Reporting a single (depth=2, 4 KB line) point risks measuring ~the GEMM's ~57 GB/s again
# and FALSELY concluding "120 GB/s is fiction". The achievable LPDDR bandwidth is the PEAK
# over a line x depth sweep (large contiguous BDs, depth >= 2), not any single config. So
# the headline number is max-over-(line,depth) of the bytes-regression slope.
#
# The harness (scripts/lpddr_bw_microbench_harness.py) computes buffer count/sizes from
# (mode, cols, bytes) and dispatches the built xclbin `iters` times via pyxrt, timing the
# median. Compute tiles are NPU2 rows 2..; shim row 0. Element type int32 (4 B).
# ===========================================================================
import argparse
import sys

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, NPU2Col1

ELEM = np.int32
ELEM_BYTES = 4


def _consume_fn(of_cons):
    # Pure DMA sink: pull one object off the fifo and release it. No arithmetic, no
    # kernel -- the compute tile only ticks the objectFIFO locks so the L3->L1 DMA drains.
    elem = of_cons.acquire(1)
    of_cons.release(1)


def _produce_fn(of_prod):
    # Pure DMA source: claim one object and release it (contents undefined -- we are
    # measuring L1->L3 write bandwidth, not correctness). No arithmetic, no kernel.
    elem = of_prod.acquire(1)
    of_prod.release(1)


def build_module(dev, mode, total_bytes, line_bytes, cols, depth):
    assert mode in ("read", "write", "rdwr")
    assert total_bytes % (cols * line_bytes) == 0, (
        f"bytes ({total_bytes}) must divide evenly into cols*line "
        f"({cols}*{line_bytes})"
    )
    per_col_bytes = total_bytes // cols
    per_col_elems = per_col_bytes // ELEM_BYTES
    line_elems = line_bytes // ELEM_BYTES
    assert per_col_elems % line_elems == 0

    line_ty = np.ndarray[(line_elems,), np.dtype[ELEM]]
    col_ty = np.ndarray[(per_col_elems,), np.dtype[ELEM]]

    workers = []
    fifos_in = []   # producer handles the runtime fills (read / rdwr)
    fifos_out = []  # consumer handles the runtime drains (write / rdwr)

    for c in range(cols):
        # Fork place-tiles model: NO explicit Tile placement. We declare `cols` independent
        # objectFIFO chains (+ workers) and aiecc's place-tiles pass maps them onto distinct
        # columns/shim DMAs (the same idiom the whole_array generator uses for its per-column
        # A/B/C fifos). cols=1 is the clean single-stream number; cols=8 the aggregate.
        of_in = ObjectFifo(line_ty, name=f"in{c}", depth=depth)

        if mode == "rdwr":
            of_out = of_in.cons().forward()
            fifos_in.append(of_in)
            fifos_out.append(of_out)
        elif mode == "read":
            workers.append(Worker(_consume_fn, [of_in.cons()]))
            fifos_in.append(of_in)
        else:  # write
            of_out = ObjectFifo(line_ty, name=f"out{c}", depth=depth)
            workers.append(Worker(_produce_fn, [of_out.prod()]))
            fifos_out.append(of_out)

    # Runtime sequence: one buffer arg per active L3 region. read=C inputs; write=C
    # outputs; rdwr=C inputs then C outputs. The harness mirrors this argument order.
    n_in = len(fifos_in)
    n_out = len(fifos_out)
    arg_types = [col_ty] * (n_in + n_out)

    rt = Runtime()
    with rt.sequence(*arg_types) as args:
        if not isinstance(args, tuple):
            args = (args,)
        for w in workers:
            rt.start(w)
        ai = 0
        # Fills (read + rdwr): wait=True on the LAST fill so the dispatch blocks until the
        # full L3->L1 read has landed.
        for i, of_in in enumerate(fifos_in):
            wait = (mode == "read") and (i == n_in - 1)
            rt.fill(of_in.prod(), args[ai], wait=wait)
            ai += 1
        # Drains (write + rdwr): wait=True on the last drain so the dispatch blocks until
        # the full L1->L3 write (and, for rdwr, the round-trip) has completed.
        for i, of_out in enumerate(fifos_out):
            wait = (i == n_out - 1)
            rt.drain(of_out.cons(), args[ai], wait=wait)
            ai += 1

    # Fork place-tiles model: bare resolve_program() -- aiecc's place-tiles pass assigns
    # physical tiles to the logical objectFIFOs/workers (NO Python-side placer).
    return Program(dev, rt).resolve_program()


def main():
    p = argparse.ArgumentParser(description="Pure-DMA LPDDR bandwidth microbenchmark")
    p.add_argument("--mode", choices=["read", "write", "rdwr"], default="rdwr")
    p.add_argument("--bytes", type=int, default=4 * 1024 * 1024,
                   help="total LPDDR bytes per direction (across all columns)")
    p.add_argument("--line", type=int, default=4096,
                   help="objectFIFO transfer granularity in bytes (BD size)")
    p.add_argument("--cols", type=int, default=1, help="number of shim columns (1..8)")
    p.add_argument("--depth", type=int, default=2, help="objectFIFO depth (double-buffer)")
    p.add_argument("--dev", choices=["npu2"], default="npu2")
    a = p.parse_args()

    dev = NPU2() if a.cols > 1 else NPU2Col1()
    module = build_module(dev, a.mode, a.bytes, a.line, a.cols, a.depth)
    print(module)


if __name__ == "__main__":
    sys.exit(main())
