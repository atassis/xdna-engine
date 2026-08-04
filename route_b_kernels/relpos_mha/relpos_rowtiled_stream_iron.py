# relpos_rowtiled_stream_iron.py -- STEP 8: the FULL-T (T up to 172) MemTile-
# STREAMED rel-pos MHA block. This is the dataflow that removes the last L1 wall:
# k/p/V are staged in the 512 KB MemTile (L2) and STREAMED to the compute tile's
# L1 in KB-row KEY-BLOCKS, RE-STREAMED once per query tile (STREAM-A: ONE shim BD
# re-reads the whole kpv from DDR n_qt times via a stride-0 outer tap dim; the
# L2->L1 repeat_count replay was rejected on device -- see the STREAM-A note
# below). The compute-tile L1 then only ever holds
# ONE key-block + the [TQ,*] score/prob/ctx scratch -- never the whole ~176 KB
# k/p/V (p alone is
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
#              STREAM-A: ONE shim BD re-reads the whole kpv from DDR offset 0, n_qt
#              times (tap outer dim = n_qt at stride 0 -> BD repeat_count=n_qt-1),
#              so each tile gets k|p|V from the START. Replaces BOTH the L2->L1
#              repeat_count replay (didn't restart, corr 0.65) and the per-tile
#              fill loop (22 shim BDs > 16-BD limit).
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
# ============================ CORE = QUERY range_ + UNROLLED BLOCKS ===========
# The QUERY-tile sweep is an aie.iron range_ hardware loop (the 22x multiplier;
# fully unrolling all 22 tiles overflowed core PROGRAM memory,
# _XAie_LoadProgMemSection). Its runtime i32 q0 = index_cast(induction Value) --
# range_(0, Tq_full, TQ) so the Value IS q0 (no multiply); q0 is exercised + VERIFIED
# on device (T=32 runs 4 query tiles and passes). The ragged final query tile is
# PEELED as one static iteration so tq stays a Python constant.
# The k/p/V BLOCK loops are UNROLLED in Python (j0 a Python-int CONSTANT), NOT nested
# range_ loops: a nested range_'s index_cast'd induction j0 did NOT deliver the
# per-iteration value on device (corr 0.65 at T=172; T=32 masked it -- its k/V block
# loops are empty and p is a single j0=0 iteration). Unrolling 16 blocks/tile emits
# ~32 block calls total (query-body + peel), far under the ~352 that overflowed, and
# j0 is a proven-good compile-time constant (the ragged peels already used static j0
# and passed at T=32). The 54 KB L1 DATA budget is unchanged.
# PROBE 1 (do Python-int kernel scalars lower?) is RESOLVED: the static build reached
# ELF and T=32 passes, so Python-int scalar args lower fine. The ONLY runtime i32 is
# the query q0 (index_cast of the query range_ induction Value), verified on device.
#
# ============================ KPV REPLAY = STREAM-A, SINGLE-BD SHIM REPLAY ====
# Two mechanisms were rejected first:
#  (1) L2->L1 forward repeat_count=n_qt: BUILT but FAILED parity on device (corr
#      0.65, rel-L2 0.82) -- the MemTile replay of the STAGED L2 buffer did NOT
#      restart per tile (it does not re-read L3), so tile q saw the wrong blocks.
#  (2) a per-query-tile rt.fill loop (n_qt calls): correct in principle but emitted
#      22 static shim DMA tasks = 22 BDs > the 16-BD shim limit on tile (0,0).
# SHIPPED = STREAM-A via ONE shim BD that RE-READS DDR. A single rt.fill with a tap
# whose outer dim is n_qt at stride 0 (sizes=[n_qt,1,kpv_pad_rows,DK], strides=
# [0,0,DK,1], via TensorTiler2D.simple_tiler(pattern_repeat=n_qt)) makes
# shim_dma_single_bd_task set BD repeat_count=n_qt-1: ONE BD replayed n_qt times,
# each re-reading the whole kpv from DDR offset 0. That re-read gives every query
# tile k|p|V from the START (the restart (1) lacked), within the BD budget (1 BD,
# not 22). of_kpv obj = [KB,DK], so each replay's kpv read = 16 blocks -> 22*16 =
# 352 blocks in address order = what the core acquires. kpv streamed from DDR n_qt
# times (STREAM-A data-movement cost); L1
# budget UNCHANGED (one [KB,DK] block at a time; 54.3 KB). A future optimization can
# revisit a WORKING resident-L2 replay to cut the DDR re-fetch; correctness first.
# SMALLEST PROBE: `python3 relpos_rowtiled_stream_iron.py -d npu2 -T 172 --tq 8 \
#   --kb 43 | grep -iEc 'scf.for'` should be SMALL (1 query hardware loop; the
#   k/p/V block loops are Python-unrolled inside it), and
#   `... | grep -iE 'memref|memtile_dma|objectfifo'` should show L1 memref allocs
# of [KB,DK] + the [TQ,*] scratch, never [T|P,DK].
#
# PLACE-TILES toolchain: bare Program(dev, rt).resolve_program(), NO SequentialPlacer.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
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


# ============================ HEADS = SPATIAL PARALLELISM (H cores) ===========
# Phase-2 perf: the 8 attention heads are INDEPENDENT (distinct q/k/p/V, distinct
# ctx) and the single-Worker block used 1 of the NPU2's 32 cores, running the 8
# heads SEQUENTIALLY (device-compute-bound: ~1.1 s/clip dispatch). With --heads H
# this file emits H WORKERS (one head/core), each running the EXACT single-head
# STREAM-A relpos compute on its own of_quv/of_kpv/of_ctx fifos + its own L1 scratch
# (g_ac/g_bd/g_probs/g_ctxf) + its own t_active RTP. The H heads run in PARALLEL, so
# the dominant device compute collapses ~Hx in wall-clock and the host dispatches
# ONCE per block (all heads) instead of per head.
#
# SCATTER (heads do NOT share data -- NOT a broadcast): the host packs all H heads'
# quv/kpv/ctx CONCATENATED in one BO each. Head h's Worker reads/writes its own
# slice via a per-head OFFSET tap (TensorAccessPattern offset = h*head_len): a
# SCATTER of H distinct streams from the shim. Each head keeps the split-p kpv layout
# k|p_hi|p_lo|V on its single channel-B stream (2 input-DMA-channel budget preserved
# per core), and the STREAM-A single-BD n_qt-replay tap (unchanged, +offset). Placed
# on H columns (place-tiles), each column's shim does that head's 3 BDs (quv+kpv+ctx)
# = well under the 16-BD shim limit; the 24 total tasks NEVER land on one shim.
#
# RTP: one SHARED WorkerRuntimeBarrier + H per-core t_active RTP buffers, all written
# by ONE inline_ops (the whole-array modal-matmul pattern). All heads share the clip's
# single t_active, so the host patches every t_active word of the template insts.
# ============================================================================
def my_relpos_stream(dev, T, TQ, KB, t_active=None, splitp=False, heads=1, tskip=False, rows=1):
    # STEP-C: T is the BAKED buffer/dataflow size (the single MAX-T xclbin, e.g. 172);
    # t_active <= T is the ACTIVE key count the softmax attends (a per-insts constant on
    # the SAME xclbin, so one xclbin serves any clip padded to T). Default = T (full).
    #
    # SPLITP (split-bf16 positional operand): the decomposition probe pinned the resident
    # block's ~1% WER-gap ENTIRELY to bf16 rounding of the positional p in BD = qv.p^T
    # (bd_p owns it; qv/k/V round fine). With --splitp the host feeds p as TWO bf16 streams
    # p_hi=bf16(p), p_lo=bf16(p-p_hi); the core computes BD = qv.p_hi (overwrite) + qv.p_lo
    # (ADD), recovering ~f32 precision on p. kpv layout becomes k | p_hi | p_lo | V (one
    # extra p-section on the SAME channel-B kpv stream; 2 input DMA channels preserved).
    if t_active is None:
        t_active = T
    P = 2 * T - 1
    n_qt = ceildiv(T, TQ)   # query tiles
    # ---- ROW-TILING (--rows R): split each head's query tiles across R cores ----
    # The heads axis is exhausted at 8 of the array's 32 compute tiles, so the second axis is
    # the query tiles. Head h's R cores all need the SAME k/p/V, which is a MULTICAST: one
    # of_kpv per head with R consumer handles (ObjectFifo.cons() returns a new handle per call).
    #
    # Two constraints, both load-bearing:
    #  * n_qt % R == 0. Every consumer of a broadcast fifo must take the SAME number of objects,
    #    so all R cores must run the same tile count -- one kpv replay each. Uneven counts leave
    #    a consumer holding objects nobody releases.
    #  * tiles are dealt ROUND-ROBIN (core r takes r, r+R, r+2R, ...), NOT in contiguous chunks.
    #    With the t_active skip, live tiles are the LOW ones; contiguous chunks would give core 0
    #    every live tile and leave core R-1 idle, and the dispatch is the slowest core. Round-robin
    #    spreads the live tiles evenly. Delivery is unaffected: each core still takes exactly one
    #    replay per tile it runs, whichever tiles those are.
    if rows > 1:
        if T % TQ:
            raise ValueError(
                f"--rows {rows} needs no ragged query tile: T={T} must be a multiple of TQ={TQ}")
        if n_qt % rows:
            raise ValueError(
                f"--rows {rows} must divide n_qt={n_qt} (T={T}, TQ={TQ}); a broadcast fifo's "
                f"consumers must all take the same object count")
    n_qt_core = n_qt // rows        # query tiles per core = kpv replays the shim must send
    n_cores = heads * rows
    n_kb = ceildiv(T, KB)   # k-blocks (also V-blocks)
    n_pb = ceildiv(P, KB)   # p-blocks (p_hi; p_lo mirrors it under --splitp)
    n_vb = n_kb
    # Padded section sizes (fixed [KB,DK] stream objects need whole-block sections).
    Tp = n_kb * KB          # padded k / V rows
    Pp = n_pb * KB          # padded p rows
    # L2/DDR padded kpv layout: k | p_hi | [p_lo] | V (p_lo section only when splitp).
    kpv_pad_rows = Tp + Pp + (Pp if splitp else 0) + Tp

    # ---- tensor types ----
    quv_blk_ty = np.ndarray[(TQ * DK,), np.dtype[bfloat16]]          # qu/qv tile
    kpv_ty = np.ndarray[(kpv_pad_rows * DK,), np.dtype[bfloat16]]    # whole padded kpv (L2)
    kblk_ty = np.ndarray[(KB * DK,), np.dtype[bfloat16]]             # one streamed block
    ctx_blk_ty = np.ndarray[(TQ * DK,), np.dtype[bfloat16]]          # ctx tile out
    # runtime-argument (L3) types. Both QUV and CTX are padded to whole TQ-tiles
    # (n_qt*TQ rows) because the fixed [TQ,DK] stream/drain blocks emit whole tiles
    # even for the ragged final tile (tq<TQ); the host reads back only the first T
    # rows of CTX. QUV is tile-interleaved: [qu_t0, qv_t0, qu_t1, qv_t1, ...].
    # Per-head lengths (a single head's slice of each concatenated H-head arg).
    quv_head_len = 2 * n_qt * TQ * DK
    kpv_head_len = kpv_pad_rows * DK
    ctx_head_len = n_qt * TQ * DK
    # H-head concatenated runtime args: head h occupies [h*head_len : (h+1)*head_len].
    quv_arg_ty = np.ndarray[(heads * quv_head_len,), np.dtype[bfloat16]]
    kpv_arg_ty = np.ndarray[(heads * kpv_head_len,), np.dtype[bfloat16]]
    ctx_arg_ty = np.ndarray[(heads * ctx_head_len,), np.dtype[bfloat16]]

    # ---- core-local resident score/prob/ctx scratch (L1 Buffers), PER HEAD ----
    # Each head runs on its own core, so it needs its OWN L1 scratch (buffers are
    # tile-local; different cores cannot share one). H sets of g_ac/g_bd/g_probs/g_ctxf.
    ac_ty = np.ndarray[(TQ * T,), np.dtype[np.float32]]
    bd_ty = np.ndarray[(TQ * P,), np.dtype[np.float32]]
    probs_ty = np.ndarray[(TQ * T,), np.dtype[bfloat16]]
    ctxf_ty = np.ndarray[(TQ * DK,), np.dtype[np.float32]]
    # Indexed by the FLAT core id c = h*rows + r (buffers are tile-local, so every core needs
    # its own set -- row-tiling multiplies these by `rows`, it does not share them per head).
    g_ac = [Buffer(ac_ty, name=f"g_ac{c}") for c in range(n_cores)]
    g_bd = [Buffer(bd_ty, name=f"g_bd{c}") for c in range(n_cores)]
    g_probs = [Buffer(probs_ty, name=f"g_probs{c}") for c in range(n_cores)]
    g_ctxf = [Buffer(ctxf_ty, name=f"g_ctxf{c}") for c in range(n_cores)]

    # STEP-C: t_active RTP register (int32[16], use_write_rtp), PER HEAD. The softmax kernel
    # reads rtp[0] at runtime, so the ELF is t_active-agnostic => ONE xclbin serves any
    # t_active. The value is written into the instruction stream (inline_ops const below =>
    # per-insts on the same xclbin, the modal-matmul pattern). A SINGLE shared barrier makes
    # every worker wait for the RTP writes before reading rtp[0] (whole-array modal pattern:
    # per-core RTP buffers, one barrier). All heads share the clip's single t_active value.
    rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
    tactive_rtp = [Buffer(rtp_ty, name=f"tactive_rtp{c}", use_write_rtp=True) for c in range(n_cores)]
    rtp_barrier = WorkerRuntimeBarrier()

    # ---- ObjectFifos ----
    # BOTH input channels are DIRECT block fifos whose obj_type IS the streamed
    # block (of_quv=[TQ,DK], of_kpv=[KB,DK]); the shim streams a large fill as many
    # obj-sized blocks in address order. This is the proven pattern; the previous
    # of_kpv used of_kpv_l3l2(obj=whole kpv).forward(obj=[KB,DK]) -- a forward with
    # a SMALLER obj than its source (split a [kpv_pad_rows,DK] L2 object into
    # [KB,DK] blocks). That is NOT the standard forward (forward is 1:1 size), it
    # was the one mechanism COMMON to all three replay variants that all produced
    # the identical wrong output (corr 0.65) even though the block-decomposed
    # compute + --stream packing are numpy-proven bit-exact to the monolithic
    # model. Numpy cannot model the objectFIFO delivery; using the same direct
    # obj=block form as of_quv (which computes qu/qv correctly) removes the
    # forward-split as the delivery variable.
    # Channel A: quv tile stream. Per query tile the core acquires qu_tile (phase
    # K) then qv_tile (phase P); 2*n_qt blocks, read ONCE (no replay).
    # PER HEAD: each head's Worker gets its own quv/kpv/ctx fifos (H distinct streams,
    # scattered from the shim by a per-head OFFSET tap -- heads do NOT share data).
    # MEMTILE STAGING (rows>1) -- forced by the SHIM DMA CHANNEL budget, not by bandwidth.
    # A ShimNOC tile has 2 DMA channels per direction (AIE2TargetModel::getNumDest/
    # SourceShimMuxConnections(DMA)), so the array's 8 shim tiles give 16 MM2S total -- and the
    # rows=1 design already uses ALL 16 (8 quv fills + 8 kpv fills). Attaching per-core fifos to
    # the shim wants 4*8+8 = 40 and place-tiles rejects it outright. So each head keeps exactly
    # ONE shim MM2S for quv and ONE shim S2MM for ctx, and the fan-out to its `rows` cores happens
    # in the MemTile (6 DMA channels per direction), via split()/join(). Shim usage is then
    # IDENTICAL to rows=1 no matter how many cores a head drives.
    #
    # The L2 object is one ROUND = `rows` blocks, one per core. Because the core acquires qu and
    # qv as two separate blocks, a round carries qu for all `rows` cores, then qv for all of them;
    # the shim fill tap below re-orders L3 into that sequence, so the HOST packing is unchanged.
    if rows > 1:
        quv_round_ty = np.ndarray[(rows * TQ * DK,), np.dtype[bfloat16]]
        ctx_round_ty = np.ndarray[(rows * TQ * DK,), np.dtype[bfloat16]]
        of_quv_l2 = [ObjectFifo(quv_round_ty, name=f"quvl2{h}", depth=2) for h in range(heads)]
        of_ctx_l2 = [ObjectFifo(ctx_round_ty, name=f"ctxl2{h}", depth=2) for h in range(heads)]
        # split(): one L2 consumer -> `rows` per-core producers, core r taking block r of the round.
        of_quv_split = [
            of_quv_l2[h].cons().split(
                [r * TQ * DK for r in range(rows)],
                obj_types=[quv_blk_ty] * rows,
                names=[f"quv{h}_{r}" for r in range(rows)],
            )
            for h in range(heads)
        ]
        # join(): `rows` per-core consumers -> one L2 producer, reassembling the round in tile order.
        of_ctx_join = [
            of_ctx_l2[h].prod().join(
                [r * TQ * DK for r in range(rows)],
                obj_types=[ctx_blk_ty] * rows,
                names=[f"ctx{h}_{r}" for r in range(rows)],
            )
            for h in range(heads)
        ]
        of_quv = [of_quv_split[c // rows][c % rows] for c in range(n_cores)]
        of_ctx = [of_ctx_join[c // rows][c % rows] for c in range(n_cores)]
    else:
        of_quv = [ObjectFifo(quv_blk_ty, name=f"quv{c}", depth=2) for c in range(n_cores)]
    # Channel B: kpv key-block stream. obj = ONE [KB,DK] block; the shim re-reads
    # the whole padded kpv (16 blocks) from DDR offset 0 n_qt times via the repeat
    # tap (single BD, repeat_count=n_qt-1 -- see the rt.fill below), so each query
    # tile gets k0..k3,p0..p7,V0..V3 from the start. 16*n_qt = 352 blocks in address
    # order = exactly what the core acquires. No MemTile staging needed (kpv is
    # re-read from DDR anyway); L1 holds one [KB,DK] block at a time.
    # of_kpv is PER HEAD (not per core): head h's `rows` cores share one k/p/V stream via
    # multicast, so the shim sends n_qt_core replays instead of n_qt and every core sees them all.
    of_kpv = [ObjectFifo(kblk_ty, name=f"kpv{h}", depth=2) for h in range(heads)]
    if rows == 1:
        of_ctx = [ObjectFifo(ctx_blk_ty, name=f"ctx{c}", depth=2) for c in range(n_cores)]

    # ---- block-brick kernels (int32-scalar ABI; see PROBE 1) ----
    # TSKIP: the _ts bricks take the softmax's rtp register + the tile's global q0 and
    # drop every block whose result the softmax never loads (a clip shorter than this
    # bucket's BUILT_T otherwise pays full-BUILT_T dot compute). Delivery is identical --
    # the Worker still acquires/releases every block -- so only arithmetic goes away, and
    # the result is bit-exact against the full path. Derivation: relpos_mha.cc.
    if tskip:
        dot_k = Kernel("relpos_stream_dot_ts", "kernels.a",
                       [quv_blk_ty, kblk_ty, ac_ty, rtp_ty,
                        np.int32, np.int32, np.int32, np.int32, np.int32])
        dot_p = Kernel("relpos_stream_dot_p_ts", "kernels.a",
                       [quv_blk_ty, kblk_ty, bd_ty, rtp_ty,
                        np.int32, np.int32, np.int32, np.int32, np.int32])
        dot_p_lo = Kernel("relpos_stream_dot_p_lo_ts", "kernels.a",
                          [quv_blk_ty, kblk_ty, bd_ty, rtp_ty,
                           np.int32, np.int32, np.int32, np.int32, np.int32])
        ctx_k = Kernel("relpos_stream_ctx_ts", "kernels.a",
                       [probs_ty, kblk_ty, ctxf_ty, rtp_ty,
                        np.int32, np.int32, np.int32, np.int32, np.int32])
    else:
        dot_k = Kernel("relpos_stream_dot", "kernels.a",
                       [quv_blk_ty, kblk_ty, ac_ty, np.int32, np.int32, np.int32, np.int32])
        dot_p = Kernel("relpos_stream_dot_p", "kernels.a",
                       [quv_blk_ty, kblk_ty, bd_ty, np.int32, np.int32, np.int32, np.int32])
        # split-bf16 p_lo pass: BD += qv @ p_lo^T (ADDS into g_bd). Only wired under --splitp.
        dot_p_lo = Kernel("relpos_stream_dot_p_lo", "kernels.a",
                          [quv_blk_ty, kblk_ty, bd_ty, np.int32, np.int32, np.int32, np.int32])
        ctx_k = Kernel("relpos_stream_ctx", "kernels.a",
                       [probs_ty, kblk_ty, ctxf_ty, np.int32, np.int32, np.int32, np.int32])
    softmax_k = Kernel("relpos_stream_softmax", "kernels.a",
                       [ac_ty, bd_ty, probs_ty, rtp_ty, np.int32, np.int32, np.int32, np.int32])
    ctxzero_k = Kernel("relpos_stream_ctx_zero", "kernels.a", [ctxf_ty, np.int32])
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

    # q_start / q_step are this CORE's query-tile range (trailing params so the tracer still sees
    # a plain function; make_core_body below binds them per core). rows=1 gives (0, TQ), the
    # original whole-head sweep. The emitted code is identical in SIZE at any rows -- the sweep is
    # a hardware loop, so only its bounds differ and row-tiling does not grow .text.
    def core_body(quv_in, kpv_in, ctx_out, ac, bd, probs, ctxf, rtp, bar,
                  dotk, dotp, dotplo, smax, czero, ctxb, narrow, q_start=0, q_step=TQ):
        # TSKIP threads rtp + the tile's q0 into the block bricks so they can drop
        # never-read blocks; the plain bricks take neither. Bind the difference once
        # here so emit_tile below stays single-form.
        if tskip:
            def c_dotk(a, b, o, tq, kb, j0, ncol, q0):  dotk(a, b, o, rtp, tq, kb, j0, ncol, q0)
            def c_dotp(a, b, o, tq, pb, j0, ncol, q0):  dotp(a, b, o, rtp, tq, pb, j0, ncol, q0)
            def c_dotplo(a, b, o, tq, pb, j0, ncol, q0): dotplo(a, b, o, rtp, tq, pb, j0, ncol, q0)
            def c_ctxb(pr, v, cf, tq, T_, kb, j0, q0):  ctxb(pr, v, cf, rtp, tq, T_, kb, j0, q0)
        else:
            def c_dotk(a, b, o, tq, kb, j0, ncol, q0):  dotk(a, b, o, tq, kb, j0, ncol)
            def c_dotp(a, b, o, tq, pb, j0, ncol, q0):  dotp(a, b, o, tq, pb, j0, ncol)
            def c_dotplo(a, b, o, tq, pb, j0, ncol, q0): dotplo(a, b, o, tq, pb, j0, ncol)
            def c_ctxb(pr, v, cf, tq, T_, kb, j0, q0):  ctxb(pr, v, cf, tq, T_, kb, j0)
        # ONE per-query-tile body, emitted ONCE inside a real hardware loop over
        # the query tiles (the 22x multiplier -- range_, index_cast'd runtime q0)
        # + ONCE for the peeled ragged tile. The k/p/V BLOCK loops are UNROLLED in
        # Python (j0 a Python-int CONSTANT per block), NOT nested range_ loops:
        # a nested range_'s index_cast'd induction j0 did NOT deliver the correct
        # per-iteration value on device (T=32 passed because its k/V block loops are
        # empty and p runs a single j0=0 iteration -- multi-iteration nested j0 is
        # only exercised at T>=86; T=172 then failed corr 0.65, numpy-reproduced by
        # a stuck/iter-index j0). The OUTER query range_'s index_cast q0 DOES work
        # (T=32's 4 query tiles pass). Unrolling 16 blocks per tile emits ~32 block
        # calls total (query-body + peel) -- far under the ~352 fully-unrolled that
        # overflowed program memory, and j0 is a proven-good compile-time constant.
        def emit_tile(tq, q0):
            # -- phase K: qu_tile resident; k full-blocks then ragged -> AC[:, j0:] --
            equ = quv_in.acquire(1)
            for j0 in range(0, Tk_full, KB):           # Python-int j0 (0,KB,2KB,..)
                ek = kpv_in.acquire(1)
                c_dotk(equ, ek, ac, tq, KB, j0, T, q0)
                kpv_in.release(1)
            if k_rag:
                ek = kpv_in.acquire(1)
                c_dotk(equ, ek, ac, tq, k_rag, Tk_full, T, q0)
                kpv_in.release(1)
            quv_in.release(1)

            # -- phase P: qv_tile resident; p_hi full-blocks then ragged -> BD[:, j0:]
            #    (overwrite). Under --splitp, immediately follow with the p_lo section
            #    (same qv_tile, same block layout) ADDING qv.p_lo into BD -> ~f32 p. --
            eqv = quv_in.acquire(1)
            for j0 in range(0, Pp_full, KB):
                ep = kpv_in.acquire(1)
                c_dotp(eqv, ep, bd, tq, KB, j0, P, q0)
                kpv_in.release(1)
            if p_rag:
                ep = kpv_in.acquire(1)
                c_dotp(eqv, ep, bd, tq, p_rag, Pp_full, P, q0)
                kpv_in.release(1)
            if splitp:
                for j0 in range(0, Pp_full, KB):
                    ep = kpv_in.acquire(1)
                    c_dotplo(eqv, ep, bd, tq, KB, j0, P, q0)  # BD += qv @ p_lo^T
                    kpv_in.release(1)
                if p_rag:
                    ep = kpv_in.acquire(1)
                    c_dotplo(eqv, ep, bd, tq, p_rag, Pp_full, P, q0)
                    kpv_in.release(1)
            quv_in.release(1)

            # -- softmax over the first rtp[0]=t_active keys (GLOBAL-index rel_shift q0);
            #    buffer stride stays T so a MAX-T xclbin serves any t_active <= T --
            smax(ac, bd, probs, rtp, tq, T, P, q0)

            # -- phase V: V full-blocks then ragged -> ctx (resident f32 accumulate) --
            eo = ctx_out.acquire(1)
            czero(ctxf, tq)
            for j0 in range(0, Tk_full, KB):
                ev = kpv_in.acquire(1)
                c_ctxb(probs, ev, ctxf, tq, T, KB, j0, q0)
                kpv_in.release(1)
            if k_rag:
                ev = kpv_in.acquire(1)
                c_ctxb(probs, ev, ctxf, tq, T, k_rag, Tk_full, q0)
                kpv_in.release(1)
            narrow(ctxf, eo, tq)
            ctx_out.release(1)

        # STEP-C: wait until the runtime sequence has written rtp[0]=t_active before the
        # softmax (which reads rtp[0]) runs. Mirrors the modal-matmul RTP barrier.
        bar.wait_for_value(1)
        # range_(q_start, Tq_full, q_step) yields this core's q0 values directly (NO multiply);
        # tq == TQ for every full tile. At rows=1 this is (0, Tq_full, TQ), the original sweep;
        # at rows=R core r takes the round-robin set r, r+R, r+2R, ... (q_start=r*TQ, step=R*TQ).
        for q0iv in range_(q_start, Tq_full, q_step):
            emit_tile(TQ, index_cast(q0iv, to=Ty.i32()))
        # peeled ragged final query tile (tq < TQ), q0 a Python constant. rows>1 rejects q_rag.
        if q_rag:
            emit_tile(q_rag, Tq_full)

    def make_core_body(q_start, q_step):
        """Bind one core's query-tile range, keeping the traced callable a plain function."""
        def body(*fn_args):
            return core_body(*fn_args, q_start=q_start, q_step=q_step)
        return body

    # One Worker per (head, row) = heads*rows cores. Kernels + the shared barrier are common;
    # each CORE gets its own quv/ctx fifos + L1 scratch + t_active RTP, and shares its head's
    # of_kpv through an extra cons() handle (the multicast). place-tiles assigns them to distinct
    # compute tiles with no explicit Tile() pinning; at heads=8 rows=4 that is all 32.
    workers = [
        Worker(
            make_core_body(r * TQ, rows * TQ),
            [of_quv[h * rows + r].cons(), of_kpv[h].cons(), of_ctx[h * rows + r].prod(),
             g_ac[h * rows + r], g_bd[h * rows + r], g_probs[h * rows + r],
             g_ctxf[h * rows + r], tactive_rtp[h * rows + r], rtp_barrier,
             dot_k, dot_p, dot_p_lo, softmax_k, ctxzero_k, ctx_k, narrow_k],
        )
        for h in range(heads) for r in range(rows)
    ]

    def sequence(QUV, KPV, CX, quv_prods, kpv_prods, ctx_conses):
        # STEP-C: bake t_active into this instruction stream's RTP (per-head buffers, all the
        # same clip t_active), then release the SHARED barrier so every worker reads it. Same
        # xclbin, different t_active => different insts (the modal-matmul per-insts pattern).
        # The body is eager now, so writing the H RTP buffers is a plain loop (was inline_ops).
        for c in range(n_cores):
            tactive_rtp[c][0] = t_active
        rtp_barrier.set(1)
        # SCATTER: head h reads/writes its own slice of each concatenated H-head arg via a
        # per-head OFFSET tap (heads do NOT share data). The kpv tap keeps the STREAM-A single-
        # BD n_qt-replay (sizes=[n_qt,1,kpv_pad_rows,DK], strides=[0,0,DK,1]) -- only the base
        # OFFSET differs per head (h*kpv_head_len). place-tiles routes each head's 3 shim BDs
        # (quv fill + kpv fill + ctx drain) onto its own column's shim (24 tasks NEVER on one
        # shim -> the 16-BD limit is never approached).
        # kpv: ONE fill per head, replayed n_qt_core times (not n_qt) -- the multicast hands each
        # replay to all `rows` cores of the head, so the DDR re-read drops by `rows` as a side
        # effect. Still a single BD (repeat_count = n_qt_core-1).
        for h in range(heads):
            kpv_tap = TensorAccessPattern(
                [heads * kpv_head_len], h * kpv_head_len,
                [n_qt_core, 1, kpv_pad_rows, DK], [0, 0, DK, 1])
            kpv_prods[h].fill(KPV, tap=kpv_tap)
        if rows == 1:
            for h in range(heads):
                quv_tap = TensorAccessPattern(
                    [heads * quv_head_len], h * quv_head_len, [quv_head_len], [1])
                ctx_tap = TensorAccessPattern(
                    [heads * ctx_head_len], h * ctx_head_len, [ctx_head_len], [1])
                quv_prods[h].fill(QUV, tap=quv_tap)
                ctx_conses[h].drain(CX, tap=ctx_tap, wait=True)
        else:
            # ONE shim MM2S + ONE shim S2MM per head; the MemTile split/join fans out to the cores.
            # QUV re-order tap: L3 is tile-interleaved [qu(t),qv(t),qu(t+1),...] but a round must
            # arrive as qu for all `rows` cores, then qv for all of them (the core acquires qu and
            # qv as separate blocks). 4 dims: round, u/v selector, core, element.
            blk = TQ * DK
            for h in range(heads):
                quv_tap = TensorAccessPattern(
                    [heads * quv_head_len], h * quv_head_len,
                    [n_qt_core, 2, rows, blk],
                    [rows * 2 * blk, blk, 2 * blk, 1])
                # CTX needs no re-order: round i holds tiles i*rows .. i*rows+rows-1, which is
                # exactly their L3 order, so the head's whole ctx slice drains contiguously.
                ctx_tap = TensorAccessPattern(
                    [heads * ctx_head_len], h * ctx_head_len, [ctx_head_len], [1])
                quv_prods[h].fill(QUV, tap=quv_tap)
                ctx_conses[h].drain(CX, tap=ctx_tap, wait=True)

    rt = Runtime(
        sequence,
        [
            quv_arg_ty,
            kpv_arg_ty,
            ctx_arg_ty,
            # rows>1 attaches the runtime to the per-head L2 relay, not to the per-core fifos --
            # that is what holds shim usage at 16 MM2S / 8 S2MM regardless of core count.
            [(of_quv_l2[h] if rows > 1 else of_quv[h]).prod() for h in range(heads)],
            [of_kpv[h].prod() for h in range(heads)],
            [(of_ctx_l2[h] if rows > 1 else of_ctx[h]).cons() for h in range(heads)],
        ],
    )

    return Program(dev, rt, workers=workers).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="npu or npu2")
p.add_argument("-T", "--frames", required=True, dest="T", type=int,
               help="encoder frame count T (P = 2T-1); must match -DRELPOS_T")
p.add_argument("--tq", type=int, default=TQ, help="query-tile rows; must match -DRELPOS_TQ")
p.add_argument("--kb", type=int, default=KB, help="key-block rows; must match -DRELPOS_KB")
p.add_argument("--tactive", type=int, default=0,
               help="STEP-C active key count (<= T); 0 => T (full). One MAX-T xclbin serves "
                    "any t_active by baking it into the instruction stream (ELF is t_active-agnostic).")
p.add_argument("--splitp", action="store_true",
               help="split-bf16 positional operand: stream p as p_hi|p_lo, BD = qv.p_hi + qv.p_lo "
                    "(near-f32 p). Closes the resident-MHA WER gap (probe: p rounding owns it).")
p.add_argument("--rows", type=int, default=1,
               help="ROW-TILE each head across R cores (heads*R workers). Needs R | n_qt and "
                    "T %% TQ == 0. Head h's R cores share one k/p/V stream by MULTICAST and take "
                    "the query tiles round-robin. R=1 = the original one-core-per-head block.")
p.add_argument("--tskip", action="store_true",
               help="runtime t_active BLOCK-SKIP: drop the k/p/V blocks whose results the "
                    "softmax never loads for this clip's t_active (bit-exact; delivery "
                    "unchanged). Deletes the intra-bucket BUILT_T padding compute.")
p.add_argument("--heads", type=int, default=1,
               help="number of attention heads to run in PARALLEL (one Worker/core each). "
                    "H=8 = Parakeet's n_heads -> one dispatch/block, 8x-parallel attention. "
                    "H=1 = the original single-Worker block (per-head host dispatch).")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"unknown device {opts.device}")

print(my_relpos_stream(dev, opts.T, opts.tq, opts.kb, opts.tactive or opts.T, opts.splitp,
                       opts.heads, opts.tskip, opts.rows))
