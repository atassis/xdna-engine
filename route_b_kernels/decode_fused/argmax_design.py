# SPDX-License-Identifier: Apache-2.0
# Per-column partial argmax design (e2e/NPU lm-head step-2). Splits a [N] bf16 vector across `cols` AIE
# columns; each column scans its contiguous N/cols slice and emits the LOCAL max index (i32) + max value
# (f32). The host does the trivial cols-way reduce (global = col*slice + local; pick the largest value).
# Modeled on iron/operators/channeled_unary_design.py (multi-column, per-column taps), but the outputs are
# tiny scalars (1 per column) instead of a same-size line.
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def my_argmax(dev, N, cols, kernel_object="argmax_slice.o", func_prefix="", verbose=False):
    assert N % cols == 0, "N must split evenly across cols"
    slice_n = N // cols
    PACK = 4  # bf16 slots per column output = 8 bytes = [val:f32 | idx:i32] (fusion is uniform-bf16)

    in_line_ty = np.ndarray[(slice_n,), np.dtype[bfloat16]]   # one column's slice
    out_line_ty = np.ndarray[(PACK,), np.dtype[bfloat16]]     # one column's packed (val,idx)
    in_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(cols * PACK,), np.dtype[bfloat16]]

    of_ins = [ObjectFifo(in_line_ty, name=f"amin{i}", depth=2) for i in range(cols)]
    of_out = [ObjectFifo(out_line_ty, name=f"amout{i}", depth=2) for i in range(cols)]

    kernel_fcn = Kernel(
        f"{func_prefix}argmax_slice_bf16",
        f"{func_prefix}{kernel_object}",
        [in_line_ty, out_line_ty, np.int32],
    )

    def core_fn(of_in, of_o, knl):
        for _ in range_(0xFFFFFFFF):
            ei = of_in.acquire(1)
            eo = of_o.acquire(1)
            knl(ei, eo, slice_n)
            of_in.release(1)
            of_o.release(1)

    workers = [
        Worker(core_fn, [of_ins[i].cons(), of_out[i].prod(), kernel_fcn])
        for i in range(cols)
    ]

    # Per-column input slice = contiguous [i*slice_n : (i+1)*slice_n].
    in_taps = [TensorAccessPattern((1, N), slice_n * i, [1, 1, 1, slice_n], [0, 0, 0, 1]) for i in range(cols)]
    # Per-column packed output gathers into out[i*PACK : (i+1)*PACK].
    out_taps = [TensorAccessPattern((1, cols * PACK), PACK * i, [1, 1, 1, PACK], [0, 0, 0, 1]) for i in range(cols)]

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (A, C):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(cols):
            rt.fill(of_ins[i].prod(), A, in_taps[i], task_group=tg)
        for i in range(cols):
            rt.drain(of_out[i].cons(), C, out_taps[i], wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
