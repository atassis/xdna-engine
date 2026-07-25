# SPDX-License-Identifier: Apache-2.0
# Single-column identity design: stream [N] bf16 through one core as tiles of `tile`, copying
# in -> out. Modeled on argmax_design.py but with the compute removed, so the only thing under
# test is the host->device->host plumbing (buffer layout, taps, fifo drain, residency/sync).
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def my_identity(dev, N, tile=None, kernel_object="identity_copy.o", func_prefix=""):
    tile = tile or N
    assert N % tile == 0, "N must split evenly into tiles"
    ntiles = N // tile

    in_line_ty = np.ndarray[(tile,), np.dtype[bfloat16]]
    out_line_ty = np.ndarray[(tile,), np.dtype[bfloat16]]
    in_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(N,), np.dtype[bfloat16]]

    of_in = ObjectFifo(in_line_ty, name="idin", depth=2)
    of_out = ObjectFifo(out_line_ty, name="idout", depth=2)

    kernel_fcn = Kernel(
        f"{func_prefix}identity_copy_bf16",
        f"{func_prefix}{kernel_object}",
        [in_line_ty, out_line_ty, np.int32],
    )

    def core_fn(of_i, of_o, knl):
        for _ in range_(0xFFFFFFFF):
            ei = of_i.acquire(1)
            eo = of_o.acquire(1)
            knl(ei, eo, tile)
            of_i.release(1)
            of_o.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod(), kernel_fcn])

    in_tap = TensorAccessPattern((1, N), 0, [1, 1, ntiles, tile], [0, 0, tile, 1])
    out_tap = TensorAccessPattern((1, N), 0, [1, 1, ntiles, tile], [0, 0, tile, 1])

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (A, C):
        rt.start(worker)
        tg = rt.task_group()
        rt.fill(of_in.prod(), A, in_tap, task_group=tg)
        rt.drain(of_out.cons(), C, out_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    # BARE resolve_program: this toolchain is the place-tiles model -- generators emit
    # aie.logical_tile and aiecc's place-tiles pass assigns physical tiles. Passing a
    # Python-side placer (SequentialPlacer) is the OLD model and is wrong here; every
    # design that actually works on device (bricklib, iron/operators/gemv) resolves bare.
    return Program(dev, rt).resolve_program()
