#
# ctxLN — encoder LayerNorm on the NPU array (Step D, internal notes §4).
#
# Derived from mlir-aie/programming_examples/ml/layernorm/layernorm.py, with TWO changes:
#   * dtype bf16 -> f32 (docs/05 "never re-expand": the encoder LN is f32 in/out; the host
#     downstream — residual add / next matmul / SiLU — consumes f32 with no re-expand tax).
#   * kernel layer_norm (bf16, unstable E[x²]-mean²) -> layer_norm_2pass_f32 (route_b_kernels/
#     aie_kernels/ln_2pass.cc): per-row f32 two-pass centered variance, matching the host
#     reference npu-asr-host/src/lib.rs `layer_norm_normalize` exactly. NORMALIZE-ONLY; the
#     affine γ,β is applied on the host for the 4 affine LN sites (exact, cheap).
#
# 8 cores; the `rows` (T=400) are split across cores (rows_per_core = rows/8 = 50); each core
# normalizes its rows independently over `cols` (D=768). The 2-arg runtime sequence (a_in, c_out)
# yields the standard host ABI args 1=instr,3=in,4=out,5=tmp,6=ctrl,7=trace — driven from Rust by
# run_matmul8(3, instr, n, in_bo, out_bo, dummy_c, dummy_tmp, dummy_tr) (output read from the b/out
# slot; see scripts/run_npu_layernorm.py for the same ABI).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def ctx_ln(dev, sequence_length, embedding_dim, trace_size):
    n_cores = 8
    assert sequence_length % n_cores == 0, "rows must split evenly across 8 cores"
    assert embedding_dim % 16 == 0, "layer_norm_2pass_f32<16> vectorizes cols by 16"

    f32 = np.float32
    total_volume = sequence_length * embedding_dim
    dtype = np.ndarray[(total_volume,), np.dtype[f32]]

    rows_per_core = sequence_length // n_cores
    chunk_type = np.ndarray[(embedding_dim,), np.dtype[f32]]  # one row of D, f32

    of_in = [ObjectFifo(chunk_type, name=f"in_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(chunk_type, name=f"out_{i}") for i in range(n_cores)]

    # f32 two-pass centered LN, normalize-only (γ=1,β=0 implied; affine on host).
    ln_kernel = Kernel(
        "layer_norm_2pass_f32", "ln_2pass.o", [chunk_type, chunk_type, np.int32]
    )

    taps_in = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )
    taps_out = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )

    def core_body(of_in, of_out, ln):
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            ln(elem_in, elem_out, embedding_dim)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, fn_args=[of_in[i].cons(), of_out[i].prod(), ln_kernel])
        for i in range(n_cores)
    ]

    rt = Runtime()
    with rt.sequence(dtype, dtype) as (a_in, c_out):
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
print(ctx_ln(dev, int(opts.rows), int(opts.cols), int(opts.trace_size)))
