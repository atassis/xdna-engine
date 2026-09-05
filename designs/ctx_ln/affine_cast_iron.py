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
from aie.iron.device import NPU1, NPU2, from_name
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def affine_cast(dev, sequence_length, embedding_dim, trace_size, dtype="f32"):
    # dtype selects which operand narrows to bf16 -- the mixed-precision-budget-sweep candidate #2
    # arms. "bf16" narrows `x`, the ctxLN->affcast inter-op STREAM ("2 MB f32 today"). "gb_bf16"
    # narrows gamma|beta instead -- a PER-OP PARAMETER, not a stream: the same rounded value is
    # reused identically across all `rows`, a different numerics risk (systematic per-column bias,
    # not independent per-element noise) that gets its own gate, not x's inherited. out (already
    # bf16) never narrows further. The generator's --dtype and the kernel's -DAFFCAST_X_BF16 /
    # -DAFFCAST_GB_BF16 MUST move together, same ABI discipline as residual_add_iron.py/-DRESADD_BF16.
    assert dtype in ("f32", "bf16", "gb_bf16"), "dtype must be f32, bf16 or gb_bf16"
    n_cores = 8
    assert sequence_length % n_cores == 0, "rows must split evenly across 8 cores"
    assert embedding_dim % 16 == 0, "affine_cast_row<16> vectorizes cols by 16"

    f32 = np.float32
    x_elem = bfloat16 if dtype == "bf16" else f32
    gb_elem = bfloat16 if dtype == "gb_bf16" else f32
    total = sequence_length * embedding_dim
    gb_len = 2 * embedding_dim  # [gamma | beta] packed on ONE DMA input channel

    rows_per_core = sequence_length // n_cores
    in_chunk = np.ndarray[(embedding_dim,), np.dtype[x_elem]]
    gb_chunk = np.ndarray[(gb_len,), np.dtype[gb_elem]]
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

    of_in = [ObjectFifo(in_chunk, name=f"in_{i}") for i in range(n_cores)]
    of_gb = [ObjectFifo(gb_chunk, name=f"gb_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(out_chunk, name=f"out_{i}") for i in range(n_cores)]

    # The OBJECT name must follow dtype too -- both arms export affine_cast_row, so a design
    # that links the wrong object silently reads the operand at the wrong width (the exact trap
    # residual_add.cc's bf16 arm hit first: 6480 NaNs, identical xclbin size, nothing looked wrong).
    kern = Kernel(
        "affine_cast_row",
        "affine_cast.o" if dtype == "f32" else f"affine_cast_{dtype}.o",
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

    x_ty = np.ndarray[(total,), np.dtype[x_elem]]
    gb_ty = np.ndarray[(gb_len,), np.dtype[gb_elem]]
    out_ty = np.ndarray[(total,), np.dtype[bfloat16]]

    def sequence(x, gb, out, in_prods, gb_prods, out_conses):
        for i in range(n_cores):
            in_prods[i].fill(x, taps_in[i])
            gb_prods[i].fill(gb)  # full [gamma|beta] to every core (broadcast)
        for i in range(n_cores):
            out_conses[i].drain(out, taps_out[i], wait=True)

    rt = Runtime(
        sequence,
        [
            x_ty,
            gb_ty,
            out_ty,
            [of_in[i].prod() for i in range(n_cores)],
            [of_gb[i].prod() for i in range(n_cores)],
            [of_out[i].cons() for i in range(n_cores)],
        ],
    )
    return Program(dev, rt, workers=workers).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
p.add_argument("--dtype", required=False, dest="dtype", default="f32",
               choices=["f32", "bf16", "gb_bf16"],
               help="f32=shipped, bf16=narrow x, gb_bf16=narrow gamma|beta; must match the "
                    "kernel's -DAFFCAST_X_BF16 / -DAFFCAST_GB_BF16")
# Partition WIDTH in columns. Default = full device (the shipped behaviour). A brick that uses
# 2 columns but claims 8 cannot co-reside with any other design, so every xclbin-to-xclbin
# transition becomes a full array reprogram -- this flag exists to measure that.
p.add_argument("-p", "--part-cols", required=False, type=int, default=None, dest="part_cols")
opts = p.parse_args(sys.argv[1:])

dev = (from_name(opts.device, n_cols=opts.part_cols) if opts.part_cols
       else (NPU2() if opts.device == "npu2" else NPU1()))
print(affine_cast(dev, int(opts.rows), int(opts.cols), int(opts.trace_size), opts.dtype))
