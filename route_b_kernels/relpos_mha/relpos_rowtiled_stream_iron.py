# relpos_rowtiled_stream_iron.py -- STEP 8: the FULL-T (T up to 172) MemTile-
# STREAMED rel-pos MHA block. This is the dataflow that removes the last L1 wall:
# k/p/V are staged ONCE in the 512 KB MemTile (L2) and STREAMED to the compute
# tile's L1 in KB-row KEY-BLOCKS, REPLAYED once per query tile (ObjectFifo
# repeat_count). The compute-tile L1 then only ever holds ONE key-block + the
# [TQ,*] score/prob/ctx scratch -- never the whole ~176 KB k/p/V (p alone is
# 86 KB > 64 KB L1 at T=172). Arithmetic is the block-decomposed bricks in
# relpos_mha.cc (relpos_stream_dot / _softmax / _ctx*), device-gated by the
# monolithic STEP=7 driver (relpos_kpvstream_bake) at the T where kpv fits L1;
# this file adds ONLY the dataflow, which the golden cannot validate.
#
# ============================ TOPOLOGY (2 input DMA channels) =================
# The NPU2 CORE tile has exactly 2 input (S2MM) + 2 output (MM2S) DMA channels
# (AIE2TargetModel::getNum{Dest,Source}SwitchboxConnections -> 2 for WireBundle
# ::DMA on core tiles; MemTile -> 6). So the compute core gets AT MOST 2 input
# streams. We use both:
#
#   Channel A  of_quv : obj_type [TQ,DK] bf16. Per query tile the core acquires
#              TWO blocks in order -- qu_tile (phase K) then qv_tile (phase P).
#              2*n_qt blocks total, each read ONCE (no replay). Host packs QUV
#              TILE-INTERLEAVED: [qu[t0], qv[t0], qu[t1], qv[t1], ...].
#   Channel B  of_kpv : obj_type [KB,DK] bf16. Per query tile the core acquires
#              n_kb k-blocks, then n_pb p-blocks, then n_vb V-blocks (the L2
#              buffer laid out k | p | V, each section padded to a KB multiple).
#              REPLAYED n_qt times via repeat_count so k/p/V are fetched from L3
#              (DDR) ONCE, then re-streamed from L2 for every query tile.
#   Output     of_ctx : obj_type [TQ,DK] bf16, 1 block per query tile.
#
# Per query tile q (q0 = q*TQ, tq = min(TQ, T-q0)) the core:
#   1. acquire qu_tile; for each k-block (j0,kb): relpos_stream_dot(qu,kblk,g_ac,
#      tq,kb,j0,ncol=T)  -> fills AC[:, j0:j0+kb]; release kblk. release qu_tile.
#   2. acquire qv_tile; for each p-block (j0,pb): relpos_stream_dot(qv,pblk,g_bd,
#      tq,pb,j0,ncol=P)  -> fills BD[:, j0:j0+pb]; release pblk. release qv_tile.
#   3. relpos_stream_softmax(g_ac,g_bd,g_probs,tq,T,P,q0)  (GLOBAL-index rel_shift).
#   4. relpos_stream_ctx_zero(g_ctxf,tq); for each V-block (j0,vb): relpos_stream_
#      ctx(g_probs,vblk,g_ctxf,tq,T,vb,j0); release vblk.
#   5. relpos_stream_narrow(g_ctxf, ctx_out, tq); release ctx_out.
# g_ac/g_bd/g_probs/g_ctxf are core-local Buffers (resident L1); the STREAMED
# thing is only ever one [KB,DK] block. This is design (a) from the scoping note:
# assemble the full [TQ,*] score rows across key-blocks, then softmax -- the score
# rows fit L1, the input k/p/V do not.
#
# ============================ CORE = REAL HARDWARE LOOPS =====================
# The query-tile sweep AND the k/p/V block sweeps are aie.iron range_ loops (the
# whole_array_modal core_fn pattern: acquire/release inside range_), NOT static
# Python unrolling -- static unrolling emitted ~352 func.calls and overflowed the
# core PROGRAM memory (_XAie_LoadProgMemSection overflow) even though the 54 KB L1
# DATA budget held. Loop counts: query tiles run as range_(0, Tq_full, TQ) so the
# induction Value IS q0 (no multiply); the ragged final query tile is PEELED as
# one static iteration so tq stays a Python constant. k/V/p full-blocks loop with
# range_(0, *_full, KB) (j0 = the induction Value); the ragged final block of each
# section is peeled. Runtime i32 scalars (q0, j0) come from index_cast of the
# range_ induction Value -- the exact helper python/helpers/dialects/scf.py uses.
# PROBE 1 (was: do Python-int kernel scalars lower?) is RESOLVED: the static build
# generated valid MLIR and reached the ELF stage, so int scalar args lower fine;
# the loop version passes the derived-index scalars as index_cast'd i32 Values.
#
# ============================ ONE BUILD PROBE (cannot numpy-validate) ========
# PROBE 2 (resident-L2 replay in blocks): of_kpv_l3l2 stages the WHOLE padded kpv
# in L2 (one L3->L2 fill); of_kpv forwards it to L1 with obj_type=[KB,DK] (smaller
# than the source) and repeat_count=n_qt. Confirm this lowers to a MemTile DMA
# that keeps kpv resident and emits it as KB-blocks n_qt times (aie.memtile_dma
# with the replay), NOT n_qt fresh L3 DMAs. If forward()-with-smaller-obj_type +
# repeat_count does not lower that way, fall back to STREAM-A (documented in
# BUILD-AND-BENCH.md): runtime re-fills kpv blocks per query tile (proven
# whole_array pattern; correct, re-fetches kpv from DDR each tile -- same L1
# budget, worse data movement).
# SMALLEST PROBE: `python3 relpos_rowtiled_stream_iron.py -d npu2 -T 172 --tq 8 \
#   --kb 43 | grep -iEc 'scf.for'` should be SMALL (the loops are real, not
#   unrolled -- 4 range_ loops: 1 query + k + p + V), and
#   `... | grep -iE 'memref|memtile_dma|objectfifo|repeat'` should show (a) L1
# memref allocs of [KB,DK] + the [TQ,*] scratch, never [T|P,DK], and (b) the kpv
# path as one resident MemTile buffer with a replayed BD.
#
# PLACE-TILES toolchain: bare Program(dev, rt).resolve_program(), NO SequentialPlacer.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
# index_cast + types: the exact helpers python/helpers/dialects/scf.py uses to turn
# a range_ induction Value (index type) into an i32 kernel-scalar operand.
from aie.extras.dialects.arith import index_cast
from aie.extras import types as Ty

DK = 128
# Defaults; overridden by --tq / --kb so they always match the kernel's
# -DRELPOS_TQ / -DRELPOS_KB. T=172=4*43 -> KB=43 blocks k/V with no pad; p (343)
# is 7 full + a 42-row ragged tail.
KB = 43
TQ = 8


def ceildiv(a, b):
    return (a + b - 1) // b


def my_relpos_stream(dev, T, TQ, KB):
    P = 2 * T - 1
    n_qt = ceildiv(T, TQ)   # query tiles
    n_kb = ceildiv(T, KB)   # k-blocks (also V-blocks)
    n_pb = ceildiv(P, KB)   # p-blocks
    n_vb = n_kb
    # Padded section sizes (fixed [KB,DK] stream objects need whole-block sections).
    Tp = n_kb * KB          # padded k / V rows
    Pp = n_pb * KB          # padded p rows
    kpv_pad_rows = Tp + Pp + Tp  # L2-resident padded kpv layout: k | p | V

    # ---- tensor types ----
    quv_blk_ty = np.ndarray[(TQ * DK,), np.dtype[bfloat16]]          # qu/qv tile
    kpv_ty = np.ndarray[(kpv_pad_rows * DK,), np.dtype[bfloat16]]    # whole padded kpv (L2)
    kblk_ty = np.ndarray[(KB * DK,), np.dtype[bfloat16]]             # one streamed block
    ctx_blk_ty = np.ndarray[(TQ * DK,), np.dtype[bfloat16]]          # ctx tile out
    # runtime-argument (L3) types. Both QUV and CTX are padded to whole TQ-tiles
    # (n_qt*TQ rows) because the fixed [TQ,DK] stream/drain blocks emit whole tiles
    # even for the ragged final tile (tq<TQ); the host reads back only the first T
    # rows of CTX. QUV is tile-interleaved: [qu_t0, qv_t0, qu_t1, qv_t1, ...].
    quv_arg_ty = np.ndarray[(2 * n_qt * TQ * DK,), np.dtype[bfloat16]]
    ctx_arg_ty = np.ndarray[(n_qt * TQ * DK,), np.dtype[bfloat16]]

    # ---- core-local resident score/prob/ctx scratch (L1 Buffers) ----
    ac_ty = np.ndarray[(TQ * T,), np.dtype[np.float32]]
    bd_ty = np.ndarray[(TQ * P,), np.dtype[np.float32]]
    probs_ty = np.ndarray[(TQ * T,), np.dtype[bfloat16]]
    ctxf_ty = np.ndarray[(TQ * DK,), np.dtype[np.float32]]
    g_ac = Buffer(ac_ty, name="g_ac")
    g_bd = Buffer(bd_ty, name="g_bd")
    g_probs = Buffer(probs_ty, name="g_probs")
    g_ctxf = Buffer(ctxf_ty, name="g_ctxf")

    # ---- ObjectFifos ----
    # Channel A: quv tile stream (L3 -> L1), read once, 2 blocks/qtile.
    of_quv = ObjectFifo(quv_blk_ty, name="quv", depth=2)
    # Channel B: kpv key-block stream. Whole padded kpv staged L3 -> L2 once, then
    # forwarded L2 -> L1 in [KB,DK] blocks, REPLAYED n_qt times (repeat_count).
    of_kpv_l3l2 = ObjectFifo(kpv_ty, name="kpv_l3l2", depth=1)
    of_kpv = of_kpv_l3l2.cons().forward(
        obj_type=kblk_ty, name="kpv_l2l1", depth=2, repeat_count=n_qt
    )
    of_ctx = ObjectFifo(ctx_blk_ty, name="ctx", depth=2)

    # ---- block-brick kernels (int32-scalar ABI; see PROBE 1) ----
    dot_k = Kernel("relpos_stream_dot", "kernels.a",
                   [quv_blk_ty, kblk_ty, ac_ty, np.int32, np.int32, np.int32, np.int32])
    dot_p = Kernel("relpos_stream_dot_p", "kernels.a",
                   [quv_blk_ty, kblk_ty, bd_ty, np.int32, np.int32, np.int32, np.int32])
    softmax_k = Kernel("relpos_stream_softmax", "kernels.a",
                       [ac_ty, bd_ty, probs_ty, np.int32, np.int32, np.int32, np.int32])
    ctxzero_k = Kernel("relpos_stream_ctx_zero", "kernels.a", [ctxf_ty, np.int32])
    ctx_k = Kernel("relpos_stream_ctx", "kernels.a",
                   [probs_ty, kblk_ty, ctxf_ty, np.int32, np.int32, np.int32, np.int32])
    narrow_k = Kernel("relpos_stream_narrow", "kernels.a", [ctxf_ty, ctx_blk_ty, np.int32])

    # ---- loop-bound split constants (peel the ragged tail; loop the full body) ----
    # Query tiles: loop q0 over the n_full FULL tiles (tq == TQ), peel the ragged
    # final tile (tq < TQ) as ONE static iteration so tq stays a Python constant.
    Tq_full = (T // TQ) * TQ          # rows covered by full query tiles
    q_rag = T - Tq_full               # ragged final-tile rows (0 if TQ | T)
    # Key/pos/value sections: loop the full KB-blocks, peel the ragged final block.
    Tk_full = (T // KB) * KB          # k / V full-block rows
    k_rag = T - Tk_full               # ragged k/V block rows (0 at T=172,KB=43)
    Pp_full = (P // KB) * KB          # p full-block rows
    p_rag = P - Pp_full               # ragged p block rows (42 at P=343,KB=43)

    def core_body(quv_in, kpv_in, ctx_out, ac, bd, probs, ctxf,
                  dotk, dotp, smax, czero, ctxb, narrow):
        # ONE per-query-tile body, emitted ONCE inside a real hardware loop over
        # the query tiles (the 22x multiplier) + ONCE for the peeled ragged tile.
        # The k/p/V block sweeps are ALSO real range_ loops (nested, whole_array_
        # modal core_fn pattern), so the emitted instruction count is BOUNDED (a
        # handful of func.call sites), not the ~352 of the static unrolling that
        # overflowed core program memory. q0/j0 become runtime i32 (index_cast of
        # the range_ induction Value); tq/kb stay Python constants via peeling.
        def emit_tile(tq, q0):
            # -- phase K: qu_tile resident; k full-blocks then ragged -> AC[:, j0:] --
            equ = quv_in.acquire(1)
            for jiv in range_(0, Tk_full, KB):
                j0 = index_cast(jiv, to=Ty.i32())
                ek = kpv_in.acquire(1)
                dotk(equ, ek, ac, tq, KB, j0, T)
                kpv_in.release(1)
            if k_rag:
                ek = kpv_in.acquire(1)
                dotk(equ, ek, ac, tq, k_rag, Tk_full, T)
                kpv_in.release(1)
            quv_in.release(1)

            # -- phase P: qv_tile resident; p full-blocks then ragged -> BD[:, j0:] --
            eqv = quv_in.acquire(1)
            for jiv in range_(0, Pp_full, KB):
                j0 = index_cast(jiv, to=Ty.i32())
                ep = kpv_in.acquire(1)
                dotp(eqv, ep, bd, tq, KB, j0, P)
                kpv_in.release(1)
            if p_rag:
                ep = kpv_in.acquire(1)
                dotp(eqv, ep, bd, tq, p_rag, Pp_full, P)
                kpv_in.release(1)
            quv_in.release(1)

            # -- softmax over assembled full score rows (GLOBAL-index rel_shift q0) --
            smax(ac, bd, probs, tq, T, P, q0)

            # -- phase V: V full-blocks then ragged -> ctx (resident f32 accumulate) --
            eo = ctx_out.acquire(1)
            czero(ctxf, tq)
            for jiv in range_(0, Tk_full, KB):
                j0 = index_cast(jiv, to=Ty.i32())
                ev = kpv_in.acquire(1)
                ctxb(probs, ev, ctxf, tq, T, KB, j0)
                kpv_in.release(1)
            if k_rag:
                ev = kpv_in.acquire(1)
                ctxb(probs, ev, ctxf, tq, T, k_rag, Tk_full)
                kpv_in.release(1)
            narrow(ctxf, eo, tq)
            ctx_out.release(1)

        # range_(0, Tq_full, TQ) yields q0 = 0, TQ, 2*TQ, ... directly (NO multiply);
        # tq == TQ for every full tile.
        for q0iv in range_(0, Tq_full, TQ):
            emit_tile(TQ, index_cast(q0iv, to=Ty.i32()))
        # peeled ragged final query tile (tq < TQ), q0 a Python constant.
        if q_rag:
            emit_tile(q_rag, Tq_full)

    worker = Worker(
        core_body,
        [of_quv.cons(), of_kpv.cons(), of_ctx.prod(),
         g_ac, g_bd, g_probs, g_ctxf,
         dot_k, dot_p, softmax_k, ctxzero_k, ctx_k, narrow_k],
    )

    rt = Runtime()
    with rt.sequence(quv_arg_ty, kpv_ty, ctx_arg_ty) as (QUV, KPV, CX):
        rt.start(worker)
        rt.fill(of_quv.prod(), QUV)       # tile-interleaved qu/qv blocks stream out
        rt.fill(of_kpv_l3l2.prod(), KPV)  # one L3->L2 fill; replayed to L1 by repeat_count
        rt.drain(of_ctx.cons(), CX, wait=True)

    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="npu or npu2")
p.add_argument("-T", "--frames", required=True, dest="T", type=int,
               help="encoder frame count T (P = 2T-1); must match -DRELPOS_T")
p.add_argument("--tq", type=int, default=TQ, help="query-tile rows; must match -DRELPOS_TQ")
p.add_argument("--kb", type=int, default=KB, help="key-block rows; must match -DRELPOS_KB")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"unknown device {opts.device}")

print(my_relpos_stream(dev, opts.T, opts.tq, opts.kb))
