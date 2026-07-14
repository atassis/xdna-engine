#
# Device-side affine + f32->bf16 cast dataflow (resident-rails LN affine seam).
#
# out[t,D] = (in[t,D] * gamma[D] + beta[D]) -> bf16.  gamma/beta are [D] f32 params, the
# SAME for every row -> broadcast to all 8 cores (each core rt.fill'd the full [D] once,
# acquired once before its row loop). Mirrors ctx_ln_iron.py's 8-core per-row structure.
#
# Runtime sequence (x, gamma, beta, out) -> kernel arg group_ids g3,g4,g5,g6; driven from
# Rust by run_matmul8(3, instr, n, bo_x, bo_gamma, bo_beta, bo_out, dummy_trace).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def affine_cast(dev, sequence_length, embedding_dim, trace_size):
    n_cores = 8
    assert sequence_length % n_cores == 0, "rows must split evenly across 8 cores"
    assert embedding_dim % 16 == 0, "affine_cast_row<16> vectorizes cols by 16"

    f32 = np.float32
    total = sequence_length * embedding_dim
    gb_len = 2 * embedding_dim  # [gamma | beta] packed on ONE DMA input channel

    rows_per_core = sequence_length // n_cores
    in_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]
    gb_chunk = np.ndarray[(gb_len,), np.dtype[f32]]
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

    of_in = [ObjectFifo(in_chunk, name=f"in_{i}") for i in range(n_cores)]
    of_gb = [ObjectFifo(gb_chunk, name=f"gb_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(out_chunk, name=f"out_{i}") for i in range(n_cores)]

    kern = Kernel(
        "affine_cast_row", "affine_cast.o",
        [in_chunk, gb_chunk, out_chunk, np.int32],
    )

    taps_in = TensorTiler2D.simple_tiler((sequence_length, embedding_dim), (rows_per_core, embedding_dim))
    taps_out = TensorTiler2D.simple_tiler((sequence_length, embedding_dim), (rows_per_core, embedding_dim))

    def core_body(of_in, of_gb, of_out, affine):
        egb = of_gb.acquire(1)  # [gamma|beta] acquired ONCE, reused across this core's rows
        for _ in range_(rows_per_core):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            affine(ei, egb, eo, embedding_dim)
            of_in.release(1)
            of_out.release(1)
        of_gb.release(1)

    workers = [
        Worker(core_body, fn_args=[of_in[i].cons(), of_gb[i].cons(), of_out[i].prod(), kern])
        for i in range(n_cores)
    ]

    rt = Runtime()
    x_ty = np.ndarray[(total,), np.dtype[f32]]
    gb_ty = np.ndarray[(gb_len,), np.dtype[f32]]
    out_ty = np.ndarray[(total,), np.dtype[bfloat16]]
    with rt.sequence(x_ty, gb_ty, out_ty) as (x, gb, out):
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_in[i].prod(), x, taps_in[i])
            rt.fill(of_gb[i].prod(), gb)  # full [gamma|beta] to every core (broadcast)
        for i in range(n_cores):
            rt.drain(of_out[i].cons(), out, taps_out[i], wait=True)
    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(affine_cast(dev, int(opts.rows), int(opts.cols), int(opts.trace_size)))
