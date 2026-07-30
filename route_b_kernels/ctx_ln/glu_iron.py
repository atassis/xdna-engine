#
# Device-side GLU dataflow (Conformer conv-module gate step).
#
# out[t, D] = a[t, D] * sigmoid(g[t, D]),  where pw1's output row is [a | g], i.e.
# in[t, 2D] = [a(0..D) | g(D..2D)]. Consumes pw1's on-chip [T, 2D] f32 -> emits
# [T, D] f32. Per-row, 8 cores (rows split evenly). Mirrors affine_cast_iron.py /
# ctx_ln_iron.py's per-row 8-core structure, but the input row is 2*D wide and the
# output row is D wide (asymmetric elementwise).
#
# 2-channel runtime sequence (a_in[2D], c_out[D]) -> host ABI args 1=instr, 3=in,
# 4=out, 5=tmp, 6=ctrl, 7=trace -- driven from Rust by
#   run_matmul8(3, instr, n, in_bo, out_bo, dummy_c, dummy_tmp, dummy_tr)
# (output read from the b/out slot; same ABI as ctx_ln / run_npu_layernorm.py).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from ml_dtypes import bfloat16  # noqa: F401 (kept parallel to affine_cast_iron; output is f32)

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def glu(dev, sequence_length, embedding_dim, trace_size, n_cores=8):
    assert sequence_length % n_cores == 0, "rows must split evenly across 8 cores"
    assert embedding_dim % 16 == 0, "glu_row<16> vectorizes cols by 16"

    f32 = np.float32
    d_in = 2 * embedding_dim          # pw1 output row: [a | g]
    total_in = sequence_length * d_in
    total_out = sequence_length * embedding_dim

    rows_per_core = sequence_length // n_cores
    in_chunk = np.ndarray[(d_in,), np.dtype[f32]]           # one row of 2D, f32
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]  # one row of D, f32

    of_in = [ObjectFifo(in_chunk, name=f"in_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(out_chunk, name=f"out_{i}") for i in range(n_cores)]

    kern = Kernel("glu_row", "glu.o", [in_chunk, out_chunk, np.int32])

    taps_in = TensorTiler2D.simple_tiler((sequence_length, d_in), (rows_per_core, d_in))
    taps_out = TensorTiler2D.simple_tiler((sequence_length, embedding_dim), (rows_per_core, embedding_dim))

    def core_body(of_in, of_out, glu_fn):
        for _ in range_(rows_per_core):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            glu_fn(ei, eo, embedding_dim)  # cols = D (output width; input is 2D)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, fn_args=[of_in[i].cons(), of_out[i].prod(), kern])
        for i in range(n_cores)
    ]

    in_ty = np.ndarray[(total_in,), np.dtype[f32]]
    out_ty = np.ndarray[(total_out,), np.dtype[f32]]

    def sequence(a_in, c_out, in_prods, out_conses):
        for i in range(n_cores):
            in_prods[i].fill(a_in, taps_in[i])
        for i in range(n_cores):
            out_conses[i].drain(c_out, taps_out[i], wait=True)

    rt = Runtime(
        sequence,
        [
            in_ty,
            out_ty,
            [of_in[i].prod() for i in range(n_cores)],
            [of_out[i].cons() for i in range(n_cores)],
        ],
    )
    prog = Program(dev, rt, workers=workers)
    # Per-op occupancy instrument (wired 2026-07-30; trace_size was already plumbed through this
    # file's CLI/signature but never connected -- see ln_affine_cast_iron.py for the recipe).
    # trace_size=0 (the production build) leaves the design byte-identical.
    if trace_size:
        # Trace ONE worker; tracing all n_cores collides on the shim's south ports once n_cores
        # saturates them. Build the traced variant with -n/--cores reduced (cores=1 is
        # representative of the per-row MATH, not production wall time).
        prog.enable_trace(trace_size=trace_size, workers=[workers[0]], egress_shim_col=1)
    return prog.resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
p.add_argument("-n", "--cores", required=False, dest="cores", default=8)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(glu(dev, int(opts.rows), int(opts.cols), int(opts.trace_size), int(opts.cores)))
