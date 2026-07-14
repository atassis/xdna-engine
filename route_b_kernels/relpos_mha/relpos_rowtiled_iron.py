# relpos_rowtiled_iron.py -- STEP 6 of the Parakeet resident MHA block:
# ROW-TILED, MemTile-STAGED. Handles T up to 172 (one head) by staging the
# resident k/p/V in the 512 KB MemTile (L2) and processing the T query rows in
# TILES of Tq inside the kernel, so only the per-tile [Tq,*] score/prob scratch
# lives in L1 (never the full 172 KB k/p/V).
#
# It drives ONE compute core with relpos_rowtiled_bake
# (route_b_kernels/relpos_mha/relpos_mha.cc):
#   quv[2T,DK] bf16 PACKED (qu=quv[0:T], qv=quv[T:2T])  -- input DMA ch 1
#   kpv[(2T+P),DK] bf16 PACKED (k, p, V)                 -- input DMA ch 2 (resident)
#   ctx[T,DK] bf16                                       -- output
# returns ctx = per-head rel-pos MHA, the query rows tiled by Tq with the
# GLOBAL-index rel_shift ((T-1)-(q0+il)) -- proven bit-identical to the single
# tile in scripts/parakeet_relpos_mha_golden.py (G6/G7).
#
# TWO packed inputs keep the core within the NPU2 compute tile's 2 input-DMA-
# channel budget (same discipline as steps 3/5). The resident kpv is routed
# through the MemTile (L3 -> L2 -> L1 via .forward()) -- that is the L2 STAGING.
#
# =========================== SCALING NOTE (T=172) ===========================
# The kernel's query-row loop is Tq-tiled, so its L1 SCORE scratch is
# g_ac[Tq*T] + g_bd[Tq*P] + g_probs[Tq*T] (~19 KB at Tq=8,T=172) -- fine. The
# open item is the RESIDENT kpv: this generator forwards the WHOLE kpv buffer to
# L1, so it only fits L1 up to a moderate T (kpv = (2T+P)*DK*2 bytes; p alone is
# 86 KB > L1 at T=172). To reach T=172, kpv must be STREAMED from the MemTile in
# KEY-BLOCKS (never fully resident in L1) with the L2 buffer REPLAYED once per
# query tile via ObjectFifo repeat_count (confirmed API:
# aie.iron.dataflow.ObjectFifo(..., repeat_count=n_qtiles) / .forward(...,
# repeat_count=...) "replays the MemTile buffer descriptor N times without a new
# L3 DMA"). That block-streamed variant requires decomposing the bake into
# per-block kernels (dot-matmul-into-column-slice / softmax-row / ctx-accumulate)
# driven from an acquire/release core loop -- see BUILD-AND-BENCH.md "Device gate
# / open item". This file is the single-tile-staged bring-up that gates the
# row-tiled ARITHMETIC on silicon at the T where kpv fits L1; the golden proves
# the arithmetic (incl. the T=172 index math) is already correct.
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


def my_relpos_rowtiled(dev, T):
    P = 2 * T - 1

    # quv packs qu[T,DK] then qv[T,DK]; kpv packs k[T,DK], p[P,DK], V[T,DK] (bf16).
    quv_ty = np.ndarray[(2 * T * DK,), np.dtype[bfloat16]]
    kpv_ty = np.ndarray[((2 * T + P) * DK,), np.dtype[bfloat16]]
    ctx_ty = np.ndarray[(T * DK,), np.dtype[bfloat16]]

    of_quv = ObjectFifo(quv_ty, name="quv", depth=1)
    # kpv routed through the MemTile: L3 -> L2 (staging) -> L1 via .forward().
    of_kpv_l3l2 = ObjectFifo(kpv_ty, name="kpv_l3l2", depth=1)
    of_kpv = of_kpv_l3l2.cons().forward(obj_type=kpv_ty, name="kpv_l2l1", depth=1)
    of_ctx = ObjectFifo(ctx_ty, name="ctx", depth=1)

    # Zero-scalar-arg kernel: T, P, Tq, DK and inv_scale are baked into the .cc.
    relpos = Kernel("relpos_rowtiled_bake", "kernels.a", [quv_ty, kpv_ty, ctx_ty])

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

print(my_relpos_rowtiled(dev, opts.T))
