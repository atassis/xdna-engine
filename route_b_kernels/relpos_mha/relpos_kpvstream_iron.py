# relpos_kpvstream_iron.py -- STEP 7 (MONOLITHIC reference) of the Parakeet
# resident MHA block. Drives relpos_kpvstream_bake, which is the row-tiled block
# with the k/p/V matmuls DECOMPOSED into KB-row key-blocks (column-slice AC/BD
# fill + resident-f32 ctx accumulate across V-blocks) -- the EXACT block bricks +
# accumulation order the MemTile-streaming core (relpos_rowtiled_stream_iron.py)
# executes, but with k/p/V still FULLY RESIDENT in L1 (same packed (quv, kpv) ABI
# as step-6). This gates the BLOCK-DECOMPOSED ARITHMETIC on silicon at the T where
# kpv fits L1, independent of the streaming dataflow; it does NOT reach T=172 on
# its own (kpv still whole in L1). relpos_rowtiled_stream_iron.py is the T=172
# dataflow variant; this file de-risks its compute half first.
#
# Same 2-input-DMA-channel discipline as steps 3/5/6: quv (query stream) + kpv
# (resident k/p/V) are the two packed inputs; ctx is the output. Runner:
# scripts/run_npu_relpos_rowtiled.py (identical ABI -- opcode 3, QUV/KPV/CTX).
#
# PLACE-TILES toolchain: bare Program(dev, rt).resolve_program(), NO SequentialPlacer.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

# head_dim baked into the kernel (RELPOS_DK, Parakeet = 128). Must match -DRELPOS_DK.
DK = 128


def my_relpos_kpvstream(dev, T):
    P = 2 * T - 1

    quv_ty = np.ndarray[(2 * T * DK,), np.dtype[bfloat16]]
    kpv_ty = np.ndarray[((2 * T + P) * DK,), np.dtype[bfloat16]]
    ctx_ty = np.ndarray[(T * DK,), np.dtype[bfloat16]]

    of_quv = ObjectFifo(quv_ty, name="quv", depth=1)
    # kpv routed through the MemTile: L3 -> L2 (staging) -> L1 via .forward().
    of_kpv_l3l2 = ObjectFifo(kpv_ty, name="kpv_l3l2", depth=1)
    of_kpv = of_kpv_l3l2.cons().forward(obj_type=kpv_ty, name="kpv_l2l1", depth=1)
    of_ctx = ObjectFifo(ctx_ty, name="ctx", depth=1)

    # Zero-scalar-arg kernel: T, P, Tq, DK, KB and inv_scale are baked into the .cc.
    relpos = Kernel("relpos_kpvstream_bake", "kernels.a", [quv_ty, kpv_ty, ctx_ty])

    def core_body(quv_in, kpv_in, ctx_out, relpos_fn):
        eq = quv_in.acquire(1)
        ek = kpv_in.acquire(1)
        eo = ctx_out.acquire(1)
        relpos_fn(eq, ek, eo)
        quv_in.release(1)
        kpv_in.release(1)
        ctx_out.release(1)

    worker = Worker(
        core_body,
        [of_quv.cons(), of_kpv.cons(), of_ctx.prod(), relpos],
    )

    rt = Runtime()
    with rt.sequence(quv_ty, kpv_ty, ctx_ty) as (QUV, KPV, CX):
        rt.start(worker)
        rt.fill(of_quv.prod(), QUV)
        rt.fill(of_kpv_l3l2.prod(), KPV)
        rt.drain(of_ctx.cons(), CX, wait=True)

    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="npu or npu2")
p.add_argument("-T", "--frames", required=True, dest="T", type=int,
               help="encoder frame count T (P = 2T-1); must match -DRELPOS_T")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"unknown device {opts.device}")

print(my_relpos_kpvstream(dev, opts.T))
