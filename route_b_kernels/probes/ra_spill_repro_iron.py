#
# Minimal non-Parakeet IRON design for the RA/spill-around-call codegen repro.
#
# A tiny 2-core, depth-2 objectfifo per-row loop: [rows, cols] f32 -> f32, rows split
# across n_cores, each core loops over its rows acquiring a fresh (ping-pong) output
# tile per row. This is the SMALLEST dataflow that exposes BOTH faces of the bug:
#   * the heavy body (ra_spill_repro.cc, RA_HOLD=1) spills a vector across a noinline
#     `jl` -> HANG; and
#   * even/odd rows land on the two ping-pong buffers of the depth-2 output fifo, so a
#     surviving-but-corrupt build shows the even-row corruption signature.
# No Parakeet weights, no exp2f. Mirrors silu_iron.py minus the activation math.
#
# 2-tensor runtime sequence (in[cols], out[cols]) -> host ABI opcode 3, in=gid3, out=gid4.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def ra_spill(dev, rows, cols, n_cores):
    assert rows % n_cores == 0, "rows must split evenly across cores"
    assert cols % 16 == 0, "ra_spill_row<16> vectorizes cols by 16"

    f32 = np.float32
    total = rows * cols
    rows_per_core = rows // n_cores
    row_chunk = np.ndarray[(cols,), np.dtype[f32]]  # one row of `cols`, f32 (in and out)

    # depth-2 objectfifos (default) -> even/odd rows alternate the two ping-pong buffers.
    of_in = [ObjectFifo(row_chunk, name=f"in_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(row_chunk, name=f"out_{i}") for i in range(n_cores)]

    kern = Kernel("ra_spill_row", "ra_spill_repro.o", [row_chunk, row_chunk, np.int32])

    taps = TensorTiler2D.simple_tiler((rows, cols), (rows_per_core, cols))

    def core_body(of_in, of_out, fn):
        for _ in range_(rows_per_core):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            fn(ei, eo, cols)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, fn_args=[of_in[i].cons(), of_out[i].prod(), kern])
        for i in range(n_cores)
    ]

    rt = Runtime()
    tensor_ty = np.ndarray[(total,), np.dtype[f32]]
    with rt.sequence(tensor_ty, tensor_ty) as (a_in, c_out):
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_in[i].prod(), a_in, taps[i])
        for i in range(n_cores):
            rt.drain(of_out[i].cons(), c_out, taps[i], wait=True)
    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-n", "--ncores", required=False, dest="ncores", default=2)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(ra_spill(dev, int(opts.rows), int(opts.cols), int(opts.ncores)))
