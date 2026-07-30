#
# Device-side f32 scaled residual-add dataflow (whole-block-resident fusion).
#
# out[t,D] = a[t,D] + scale*b[t,D], all f32. BOTH inputs are per-row tiled (a = running
# activation, b = sub-layer output); `scale` is BAKED as a compile-time literal (the AIE2
# 2-input-DMA limit is consumed by a and b, so no runtime scale channel -> one xclbin per
# scale). 8 cores, rows_per_core = T/8, one [D] row per core_body iteration -- mirrors
# acc_add_iron.py (2 input DMA channels: a, b).
#
# Runtime sequence (a, b, out) -> kernel arg group_ids g3,g4,g5; scale/cols are baked scalars.
# Driven from Rust by run_matmul8(3, instr, n, bo_a, bo_b, bo_out, dummy_tmp, dummy_trace).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_
from aie.utils.trace.events import CoreEvent

# Explicit coretile_events override (2026-07-30): names the ~65-68% of traced span the 8
# HARDWARE-DEFAULT counters left uncharacterized on this same brick, same span, measured via
# a device trace histogram across resadd/glu. Keeps INSTR_VECTOR/
# MEMORY_STALL/STREAM_STALL/LOCK_STALL for direct comparability to that run; swaps out
# INSTR_EVENT_0/1 (function-entry/exit markers, already ~0.02% each) and PORT_RUNNING_0/1
# (DMA port activity, already characterized, MOVEMENT not CORE side) for CASCADE_STALL
# (completes the 4-way stall family -- GROUP_STALL siblings are MEMORY/STREAM/CASCADE/LOCK),
# INSTR_LOAD + INSTR_STORE (the only per-instruction-type events for scalar load/store --
# AIE2/2p exposes no generic "scalar ALU" event, load/store is the closest to
# "address-generation activity"), and ACTIVE (the core-status-group event that is the
# genuine busy/not-disabled/not-stalled signal -- its complement is genuine core idle).
# Numeric codes from aie.dialects._aie_enum_gen.CoreEventAIE2P (TableGen-generated from the
# aie-rt headers per aie/utils/trace/events/__init__.py's docstring); CoreEventAIE2 (what
# CoreEvent aliases to) carries IDENTICAL numeric codes for all 8 of these names, so which
# alias is used does not change what gets programmed into the hardware monitor slots.
# NOTE: dropping INSTR_EVENT_0/1 means get_trace_summary.py's invocation-boundary parse
# (which pairs INSTR_EVENT_0/1 B-events to find "cycles/invocation") no longer applies to
# this capture -- use total captured span instead.
BAND_NAMING_CORETILE_EVENTS = [
    CoreEvent.INSTR_VECTOR,
    CoreEvent.MEMORY_STALL,
    CoreEvent.STREAM_STALL,
    CoreEvent.LOCK_STALL,
    CoreEvent.CASCADE_STALL,
    CoreEvent.INSTR_LOAD,
    CoreEvent.INSTR_STORE,
    CoreEvent.ACTIVE,
]


def residual_add(dev, sequence_length, embedding_dim, scale, trace_size, n_cores=8):
    assert sequence_length % n_cores == 0, "rows must split evenly across 8 cores"
    assert embedding_dim % 16 == 0, "residual_add_row<16> vectorizes cols by 16"

    f32 = np.float32
    total = sequence_length * embedding_dim
    rows_per_core = sequence_length // n_cores

    a_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]
    b_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]

    of_a = [ObjectFifo(a_chunk, name=f"a_{i}") for i in range(n_cores)]
    of_b = [ObjectFifo(b_chunk, name=f"b_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(out_chunk, name=f"out_{i}") for i in range(n_cores)]

    kern = Kernel(
        "residual_add_row", "residual_add.o",
        [a_chunk, b_chunk, out_chunk, np.float32, np.int32],
    )

    taps_a = TensorTiler2D.simple_tiler((sequence_length, embedding_dim), (rows_per_core, embedding_dim))
    taps_b = TensorTiler2D.simple_tiler((sequence_length, embedding_dim), (rows_per_core, embedding_dim))
    taps_out = TensorTiler2D.simple_tiler((sequence_length, embedding_dim), (rows_per_core, embedding_dim))

    def core_body(of_a, of_b, of_out, add):
        for _ in range_(rows_per_core):
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            eo = of_out.acquire(1)
            add(ea, eb, eo, scale, embedding_dim)  # scale baked as a literal
            of_a.release(1)
            of_b.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, fn_args=[of_a[i].cons(), of_b[i].cons(), of_out[i].prod(), kern])
        for i in range(n_cores)
    ]

    a_ty = np.ndarray[(total,), np.dtype[f32]]
    b_ty = np.ndarray[(total,), np.dtype[f32]]
    out_ty = np.ndarray[(total,), np.dtype[f32]]

    def sequence(a, b, out, a_prods, b_prods, out_conses):
        for i in range(n_cores):
            a_prods[i].fill(a, taps_a[i])
            b_prods[i].fill(b, taps_b[i])
        for i in range(n_cores):
            out_conses[i].drain(out, taps_out[i], wait=True)

    rt = Runtime(
        sequence,
        [
            a_ty,
            b_ty,
            out_ty,
            [of_a[i].prod() for i in range(n_cores)],
            [of_b[i].prod() for i in range(n_cores)],
            [of_out[i].cons() for i in range(n_cores)],
        ],
    )
    prog = Program(dev, rt, workers=workers)
    # Per-op occupancy instrument (wired 2026-07-30, connecting the trace_size knob this file
    # already accepted but never called -- see ln_affine_cast_iron.py for the established recipe).
    # trace_size=0 (the production build) leaves the design byte-identical.
    if trace_size:
        # Trace ONE worker, not all n_cores. Tracing every worker collides on the shim's south
        # ports ("aie.masterset op targets same destination South: 3") once n_cores gets large
        # enough to saturate them -- build the traced variant with -n/--cores reduced (cores=1
        # is representative of the per-row MATH, not production wall time; see the enable_trace
        # recipe established for ln_affine_cast_iron.py, which this mirrors).
        prog.enable_trace(
            trace_size=trace_size, workers=[workers[0]], egress_shim_col=1,
            coretile_events=BAND_NAMING_CORETILE_EVENTS,
        )
    return prog.resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-s", "--scale", required=True, dest="scale")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
p.add_argument("-n", "--cores", required=False, dest="cores", default=8)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(residual_add(dev, int(opts.rows), int(opts.cols), float(opts.scale), int(opts.trace_size),
                   int(opts.cores)))
