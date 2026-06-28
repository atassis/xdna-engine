# dwconv1d/dwconv1d.py -*- Python -*-
#
# IRON design for depthwise conv1d (k=9, 'same' padding) over [C=1024, T=400] --
# the Parakeet-TDT FastConformer ConvModule depthwise conv. One ObjectFifo tile
# == one channel's time series; a second fifo streams that channel's weight tile
# (9 taps in slots [0..8] + BatchNorm-folded bias in slot [9], padded to 16). C
# channels are split across `columns`, each column-core looping over its share.
# Mirrors the proven eltwise_mul 3-buffer (in / weights / out) template; the core
# kernel (dwconv1d.cc) computes the FIR with the aie::sliding_mul COMPUTE brick.
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

C = 1024  # channels (Parakeet d_model)
T = 400   # time steps (encoder frame count baked at build)
KW = 16   # weight tile size (taps[0..8] + bias[9], rest unused)


def my_dwconv(dev, num_columns):
    dtype = bfloat16
    if C % num_columns != 0:
        raise ValueError(f"C={C} must be divisible by columns={num_columns}")
    cpc = C // num_columns  # channels per column

    in_tile_ty = np.ndarray[(T,), np.dtype[dtype]]
    w_tile_ty = np.ndarray[(KW,), np.dtype[dtype]]
    out_tile_ty = np.ndarray[(T,), np.dtype[dtype]]

    in_tensor_ty = np.ndarray[(C * T,), np.dtype[dtype]]
    w_tensor_ty = np.ndarray[(C * KW,), np.dtype[dtype]]
    out_tensor_ty = np.ndarray[(C * T,), np.dtype[dtype]]

    of_ins = [ObjectFifo(in_tile_ty, name=f"in_{i}") for i in range(num_columns)]
    of_ws = [ObjectFifo(w_tile_ty, name=f"w_{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(out_tile_ty, name=f"out_{i}") for i in range(num_columns)]

    dwconv = Kernel(
        "dwconv1d_k9_bf16", "kernels.a", [in_tile_ty, w_tile_ty, out_tile_ty]
    )

    def core_body(of_in, of_w, of_out, dwconv_fn):
        for _ in range_(cpc):
            ei = of_in.acquire(1)
            ew = of_w.acquire(1)
            eo = of_out.acquire(1)
            dwconv_fn(ei, ew, eo)
            of_in.release(1)
            of_w.release(1)
            of_out.release(1)

    workers = [
        Worker(
            core_body,
            [of_ins[i].cons(), of_ws[i].cons(), of_outs[i].prod(), dwconv],
        )
        for i in range(num_columns)
    ]

    in_taps = [
        TensorAccessPattern((1, C * T), cpc * T * i, [1, 1, 1, cpc * T], [0, 0, 0, 1])
        for i in range(num_columns)
    ]
    w_taps = [
        TensorAccessPattern((1, C * KW), cpc * KW * i, [1, 1, 1, cpc * KW], [0, 0, 0, 1])
        for i in range(num_columns)
    ]
    out_taps = in_taps

    rt = Runtime()
    with rt.sequence(in_tensor_ty, w_tensor_ty, out_tensor_ty) as (X, W, Y):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(num_columns):
            rt.fill(of_ins[i].prod(), X, in_taps[i], task_group=tg)
            rt.fill(of_ws[i].prod(), W, w_taps[i], task_group=tg)
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

print(my_dwconv(dev, int(opts.cols)))
