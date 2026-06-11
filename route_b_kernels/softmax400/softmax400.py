#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Per-ROW softmax for GigaAM attention scores.
#
# Adapted from programming_examples/ml/softmax/softmax.py. The ONLY structural
# change is the per-tile reduction length: here a "tile" == one attention row of
# length ROW=416 (= 13*32, a multiple of SM_VEC_LEN=32 so the kernel's 32-wide
# reduce never truncates). The host pads each real length-400 row with 16 columns
# of a large negative value (-1e30) so that exp2(pad - max) underflows to 0 and the
# softmax denominator is taken over the 400 real keys only. Result in cols[:400]
# is bit-exact equal to a true length-400 softmax (verified in run_npu_softmax400.py).
#
from ml_dtypes import bfloat16
import numpy as np
import sys
import argparse

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.controlflow import range_


def vector_softmax(dev, trace_size, N, ROW):

    # Per-tile reduction length == one attention row (400 real + 16 pad).
    n = ROW
    assert n % 32 == 0, f"ROW={n} must be a multiple of SM_VEC_LEN=32"
    assert N % n == 0, f"N={N} must be a multiple of ROW={n}"
    N_div_n = N // n

    n_cores = 2
    assert N_div_n % n_cores == 0, (
        f"row count {N_div_n} must be divisible by n_cores={n_cores}"
    )
    tiles = N_div_n // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    # Type used in the memory tile which aggregates across the cores
    A_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    C_memTile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]

    # AIE Core Function declarations
    softmax_bf16_vector = Kernel(
        "softmax_bf16", "kernels.a", [tile_ty, tile_ty, np.int32]
    )

    # AIE-array data movement with object fifos
    # Input A and Output C
    inA = ObjectFifo(A_memTile_ty, name="inA")
    outC = ObjectFifo(C_memTile_ty, name="outC")

    of_a_offsets = []
    of_c_offsets = []
    if n_cores > 1:
        of_a_offsets = [n * i for i in range(n_cores)]
        of_c_offsets = [n * i for i in range(n_cores)]
    inA_fifos = inA.cons().split(
        of_a_offsets,
        obj_types=[tile_ty] * n_cores,
        names=[f"memA{i}" for i in range(n_cores)],
    )
    outC_fifos = outC.prod().join(
        of_c_offsets,
        obj_types=[tile_ty] * n_cores,
        names=[f"memC{i}" for i in range(n_cores)],
    )

    # Task for the cores to perform
    def core_fn(of_in, of_out, softmax_kernel):
        for _ in range_(tiles):
            elem_out = of_out.acquire(1)
            elem_in_a = of_in.acquire(1)
            softmax_kernel(elem_in_a, elem_out, n)
            of_in.release(1)
            of_out.release(1)

    # Set up workers to perform the task
    workers = []
    for i in range(n_cores):
        workers.append(
            Worker(
                core_fn,
                fn_args=[
                    inA_fifos[i].cons(),
                    outC_fifos[i].prod(),
                    softmax_bf16_vector,
                ],
            )
        )

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.start(*workers)
        rt.fill(inA.prod(), A)
        rt.drain(outC.cons(), C, wait=True)

    # Place components and generate an MLIR module
    return Program(dev, rt).resolve_program()


def main():
    parser = argparse.ArgumentParser(prog="softmax400")
    parser.add_argument(
        "device_name",
        choices=["npu", "npu2"],
        default="npu2",
        help="Device name (npu or npu2)",
    )
    parser.add_argument(
        "trace_size_pos",
        nargs="?",
        type=int,
        default=0,
        help="Trace size (optional positional, default: 0)",
    )
    parser.add_argument(
        "--trace_size",
        dest="trace_size_flag",
        type=int,
        default=0,
        help="Trace size (optional flag, default: 0)",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=416,
        help="Per-row softmax length incl. padding (default: 416 = 400 real + 16 pad)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=416 * 6400,  # 2,662,400 = all 6400 attention rows in one dispatch
        help="Total number of bf16 elements (default: ROW * 6400)",
    )

    args = parser.parse_args()

    trace_size = (
        args.trace_size_flag if args.trace_size_flag != 0 else args.trace_size_pos
    )

    if args.device_name == "npu":
        dev = NPU1Col1()
    elif args.device_name == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError(f"[ERROR] Device name {args.device_name} is unknown")

    module = vector_softmax(dev, trace_size, args.size, args.row)
    print(module)


if __name__ == "__main__":
    main()
