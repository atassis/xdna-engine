#
# Device-side f32 -> bf16 cast dataflow (resident-rails seam primitive).
#
# Mirrors ctx_ln_iron.py's 8-core per-row structure EXACTLY (rows split across 8
# cores, each core processes its rows over `cols`), but the kernel is the
# elementwise cast (cast_f32_bf16.cc) and the OUTPUT ObjectFifo is bf16 (half the
# bytes). Input f32, output bf16. Used to bridge an f32 producer (ctxLN) to the
# bf16-in whole_array matmul with the activation staying device-side.
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


def cast_ffn(dev, sequence_length, embedding_dim, trace_size):
    n_cores = 8
    assert sequence_length % n_cores == 0, "rows must split evenly across 8 cores"
    assert embedding_dim % 16 == 0, "cast_f32_bf16_row<16> vectorizes cols by 16"

    f32 = np.float32
    total = sequence_length * embedding_dim
    in_dtype = np.ndarray[(total,), np.dtype[f32]]
    out_dtype = np.ndarray[(total,), np.dtype[bfloat16]]

    rows_per_core = sequence_length // n_cores
    in_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]       # one f32 row of D
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]  # one bf16 row of D

    of_in = [ObjectFifo(in_chunk, name=f"in_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(out_chunk, name=f"out_{i}") for i in range(n_cores)]

    cast_kernel = Kernel(
        "cast_f32_bf16_row", "cast_f32_bf16.o", [in_chunk, out_chunk, np.int32]
    )

    taps_in = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )
    taps_out = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )

    def core_body(of_in, of_out, cast):
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            cast(elem_in, elem_out, embedding_dim)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, fn_args=[of_in[i].cons(), of_out[i].prod(), cast_kernel])
        for i in range(n_cores)
    ]

    rt = Runtime()
    with rt.sequence(in_dtype, out_dtype) as (a_in, c_out):
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_in[i].prod(), a_in, taps_in[i])
        for i in range(n_cores):
            rt.drain(of_out[i].cons(), c_out, taps_out[i], wait=True)
    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(cast_ffn(dev, int(opts.rows), int(opts.cols), int(opts.trace_size)))
