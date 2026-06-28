# conf_epi_silu_design.py -*- Python -*-
#
# IRON on-device harness for the A3 fused SiLU epilogue (conformer_epilogues.cc).
# Single-input (f32 accumulator C-tile) -> single-output (bf16) eltwise, mirroring
# the proven dwconv1d.py 3-buffer template but with only in/out fifos (no weight).
# One tile == one row of N=1024 lanes; T rows split across `columns`, each core
# looping over its share and calling conformer_silu_epilogue_f32_bf16 (EPI_M=1,
# EPI_N=1024 fixed-size symbol).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_

N = 1024  # lanes per tile (EPI_N; d_model)
T = 64    # rows (sample frames)


def my_silu(dev, num_columns):
    in_dtype = np.float32
    out_dtype = bfloat16
    if T % num_columns != 0:
        raise ValueError(f"T={T} must be divisible by columns={num_columns}")
    rpc = T // num_columns  # rows per column

    in_tile_ty = np.ndarray[(N,), np.dtype[in_dtype]]
    out_tile_ty = np.ndarray[(N,), np.dtype[out_dtype]]

    in_tensor_ty = np.ndarray[(T * N,), np.dtype[in_dtype]]
    out_tensor_ty = np.ndarray[(T * N,), np.dtype[out_dtype]]

    of_ins = [ObjectFifo(in_tile_ty, name=f"in_{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(out_tile_ty, name=f"out_{i}") for i in range(num_columns)]

    silu = Kernel(
        "conformer_silu_epilogue_f32_bf16", "kernels.a", [in_tile_ty, out_tile_ty]
    )

    def core_body(of_in, of_out, silu_fn):
        for _ in range_(rpc):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            silu_fn(ei, eo)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, [of_ins[i].cons(), of_outs[i].prod(), silu])
        for i in range(num_columns)
    ]

    in_taps = [
        TensorAccessPattern((1, T * N), rpc * N * i, [1, 1, 1, rpc * N], [0, 0, 0, 1])
        for i in range(num_columns)
    ]
    out_taps = in_taps

    rt = Runtime()
    with rt.sequence(in_tensor_ty, out_tensor_ty) as (X, Y):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(num_columns):
            rt.fill(of_ins[i].prod(), X, in_taps[i], task_group=tg)
        for i in range(num_columns):
            rt.drain(of_outs[i].cons(), Y, out_taps[i], wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="npu or npu2")
p.add_argument("-co", "--columns", required=True, dest="cols")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"[ERROR] Device name {opts.device} is unknown")

print(my_silu(dev, int(opts.cols)))
