# relpos_full_iron.py -- STEP 5 (composition) of the Parakeet resident MHA block.
#
# The ENTIRE per-head MHA node in one dispatch: AC + BD matmuls -> rel_shift +
# softmax -> ctx matmul, all resident. Two packed bf16 inputs (2-DMA-channel budget):
#   qkv[3T,DK] = {qu, k, V}   qvp[(T+P),DK] = {qv, p}   -> ctx[T,DK].
# Single-tile, small-T only (proves the composition; the T<=172 row-tiled MemTile
# design is the scaling step). PLACE-TILES: bare resolve_program(), NO SequentialPlacer.
import sys
import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

DK = 128


def my_relpos_full(dev, T):
    P = 2 * T - 1
    qkv_ty = np.ndarray[(3 * T * DK,), np.dtype[bfloat16]]
    qvp_ty = np.ndarray[((T + P) * DK,), np.dtype[bfloat16]]
    ctx_ty = np.ndarray[(T * DK,), np.dtype[bfloat16]]

    of_qkv = ObjectFifo(qkv_ty, name="qkv", depth=1)
    of_qvp = ObjectFifo(qvp_ty, name="qvp", depth=1)
    of_ctx = ObjectFifo(ctx_ty, name="ctx", depth=1)

    relpos = Kernel("relpos_full_bake", "kernels.a", [qkv_ty, qvp_ty, ctx_ty])

    def core_body(qkv_in, qvp_in, ctx_out, relpos_fn):
        e1 = qkv_in.acquire(1)
        e2 = qvp_in.acquire(1)
        eo = ctx_out.acquire(1)
        relpos_fn(e1, e2, eo)
        qkv_in.release(1)
        qvp_in.release(1)
        ctx_out.release(1)

    worker = Worker(core_body, [of_qkv.cons(), of_qvp.cons(), of_ctx.prod(), relpos])

    rt = Runtime()
    with rt.sequence(qkv_ty, qvp_ty, ctx_ty) as (QKV, QVP, CX):
        rt.start(worker)
        rt.fill(of_qkv.prod(), QKV)
        rt.fill(of_qvp.prod(), QVP)
        rt.drain(of_ctx.cons(), CX, wait=True)

    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-T", "--frames", required=True, dest="T", type=int)
opts = p.parse_args(sys.argv[1:])
dev = NPU2() if opts.device == "npu2" else NPU1()
print(my_relpos_full(dev, opts.T))
