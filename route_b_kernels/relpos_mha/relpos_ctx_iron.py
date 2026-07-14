# relpos_ctx_iron.py -- STEP 4 of the Parakeet resident MHA block.
#
# The AV / context matmul in isolation (host-fed probs), mirroring step 1's host-
# fed scores: given probs[T,T] bf16 (attention weights) and V[T,DK] bf16 (one
# head), the core returns ctx[T,DK] bf16 = probs @ V (ctx[i]=sum_j probs[i,j]*V[j]).
# Validates the context brick before it is composed into the full resident block.
#
# 3-buffer ABI (probs in / V in / ctx out), single compute core, single tile.
# PLACE-TILES toolchain: bare Program(dev, rt).resolve_program(), NO SequentialPlacer.
import sys
import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

# head_dim baked into the kernel (RELPOS_DK, Parakeet = 128). Must match -DRELPOS_DK.
DK = 128


def my_relpos_ctx(dev, T):
    probs_ty = np.ndarray[(T * T,), np.dtype[bfloat16]]
    v_ty = np.ndarray[(T * DK,), np.dtype[bfloat16]]
    ctx_ty = np.ndarray[(T * DK,), np.dtype[bfloat16]]

    of_probs = ObjectFifo(probs_ty, name="probs", depth=1)
    of_v = ObjectFifo(v_ty, name="v", depth=1)
    of_ctx = ObjectFifo(ctx_ty, name="ctx", depth=1)

    relpos = Kernel("relpos_ctx_bake", "kernels.a", [probs_ty, v_ty, ctx_ty])

    def core_body(probs_in, v_in, ctx_out, relpos_fn):
        ep = probs_in.acquire(1)
        ev = v_in.acquire(1)
        eo = ctx_out.acquire(1)
        relpos_fn(ep, ev, eo)
        probs_in.release(1)
        v_in.release(1)
        ctx_out.release(1)

    worker = Worker(
        core_body,
        [of_probs.cons(), of_v.cons(), of_ctx.prod(), relpos],
    )

    rt = Runtime()
    with rt.sequence(probs_ty, v_ty, ctx_ty) as (PR, V, CX):
        rt.start(worker)
        rt.fill(of_probs.prod(), PR)
        rt.fill(of_v.prod(), V)
        rt.drain(of_ctx.cons(), CX, wait=True)

    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="npu or npu2")
p.add_argument("-T", "--frames", required=True, dest="T", type=int,
               help="encoder frame count T; must match -DRELPOS_T")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"unknown device {opts.device}")

print(my_relpos_ctx(dev, opts.T))
