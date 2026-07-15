#
# Device-side SiLU dataflow (Conformer conv-module post-dwconv activation).
#
# out[r, cols] = silu(in[r, cols]) = in * sigmoid(in). Per-row, 8 cores (rows split
# evenly). Symmetric elementwise (in width == out width == cols), so this mirrors
# glu_iron.py minus the a/g split. Consumes the dwconv output [C=rows, T=cols] f32
# and emits [C, T] f32 (the post-dwconv SiLU). SEPARATE single-op-loop brick (NOT a
# dwconv epilogue) -- immune to the fused-epilogue per-channel-loop miscompile.
#
# 2-channel runtime sequence (in[cols], out[cols]) -> host ABI 1=instr, 3=in, 4=out,
# 5=tmp, 6=ctrl, 7=trace, driven from Rust by
#   run_matmul8(3, instr, n, in_bo, out_bo, dummy_c, dummy_tmp, dummy_tr).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def silu(dev, rows, cols, trace_size):
    n_cores = 8
    assert rows % n_cores == 0, "rows must split evenly across 8 cores"
    assert cols % 16 == 0, "silu_row<16> vectorizes cols by 16"

    f32 = np.float32
    total = rows * cols
    rows_per_core = rows // n_cores
    row_chunk = np.ndarray[(cols,), np.dtype[f32]]  # one row of `cols`, f32 (in and out)

    of_in = [ObjectFifo(row_chunk, name=f"in_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(row_chunk, name=f"out_{i}") for i in range(n_cores)]

    kern = Kernel("silu_row", "silu_brick.o", [row_chunk, row_chunk, np.int32])

    taps = TensorTiler2D.simple_tiler((rows, cols), (rows_per_core, cols))

    def core_body(of_in, of_out, silu_fn):
        for _ in range_(rows_per_core):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            silu_fn(ei, eo, cols)
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
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(silu(dev, int(opts.rows), int(opts.cols), int(opts.trace_size)))
