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


def residual_add(dev, sequence_length, embedding_dim, scale, trace_size):
    n_cores = 8
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

    rt = Runtime()
    a_ty = np.ndarray[(total,), np.dtype[f32]]
    b_ty = np.ndarray[(total,), np.dtype[f32]]
    out_ty = np.ndarray[(total,), np.dtype[f32]]
    with rt.sequence(a_ty, b_ty, out_ty) as (a, b, out):
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_a[i].prod(), a, taps_a[i])
            rt.fill(of_b[i].prod(), b, taps_b[i])
        for i in range(n_cores):
            rt.drain(of_out[i].cons(), out, taps_out[i], wait=True)
    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-s", "--scale", required=True, dest="scale")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(residual_add(dev, int(opts.rows), int(opts.cols), float(opts.scale), int(opts.trace_size)))
