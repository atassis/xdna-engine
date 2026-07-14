#
# Deinterleave + f32->bf16 cast (resident-rails fc1->fc2 seam, Variant B).
#
# Input  fc1 output [T, DFF] f32 (row-major, DFF = n_chunks*KRES).
# Output [n_chunks, T, KRES] bf16 CHUNK-MAJOR: chunk c contiguous at offset c*T*KRES, so the fc2
# K-split can feed each K=KRES chunk as a device sub-buffer (Bo::sub) -- NO host round-trip, no
# strided matmul A. ONE dispatch: the kernel is the plain round-nearest cast (cast_f32_bf16_row);
# the chunk-major reorg is done entirely by the OUTPUT drain TensorAccessPattern (1 input, 1 strided
# output channel -- so the 2-output-DMA wall does not apply).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
from aie.iron.controlflow import range_

KRES = 1024  # fc2 K-chunk width (resident matmul K)


def deint_cast(dev, sequence_length, embedding_dim, trace_size):
    n_cores = 8
    assert sequence_length % n_cores == 0, "rows must split evenly across 8 cores"
    assert embedding_dim % KRES == 0, "DFF must be a whole number of KRES chunks"
    n_chunks = embedding_dim // KRES

    f32 = np.float32
    total = sequence_length * embedding_dim
    in_dtype = np.ndarray[(total,), np.dtype[f32]]
    out_dtype = np.ndarray[(total,), np.dtype[bfloat16]]  # holds [n_chunks, T, KRES] chunk-major

    rows_per_core = sequence_length // n_cores
    in_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]       # one f32 row of DFF
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]  # one bf16 row of DFF (reorged on drain)

    of_in = [ObjectFifo(in_chunk, name=f"in_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(out_chunk, name=f"out_{i}") for i in range(n_cores)]

    # reuse the plain round-nearest cast kernel (writes [DFF] bf16 row-major to the fifo)
    cast_kernel = Kernel(
        "cast_f32_bf16_row", "cast_f32_bf16.o", [in_chunk, out_chunk, np.int32]
    )

    taps_in = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )

    # OUTPUT: chunk-major drain. For core i (rows [i*rpc : i*rpc+rpc]) the [rpc, DFF] fifo tile,
    # interpreted as [rpc, n_chunks, KRES], is written to out[n_chunks, T, KRES] as
    #   out[c, i*rpc+r, j] = fifo[r, c*KRES + j]
    # => access (r, c, j) with dest strides (KRES, T*KRES, 1), base offset i*rpc*KRES.
    def out_tap(i):
        return TensorAccessPattern(
            (n_chunks, sequence_length, KRES),
            i * rows_per_core * KRES,
            [rows_per_core, n_chunks, KRES],
            [KRES, sequence_length * KRES, 1],
        )

    def core_body(of_in, of_out, cast):
        for _ in range_(rows_per_core):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            cast(ei, eo, embedding_dim)
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
            rt.drain(of_out[i].cons(), c_out, out_tap(i), wait=True)
    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(deint_cast(dev, int(opts.rows), int(opts.cols), int(opts.trace_size)))
