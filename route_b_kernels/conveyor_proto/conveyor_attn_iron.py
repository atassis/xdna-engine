# REAL attention CONVEYOR, QUERY-TILED for pipeline OVERLAP (the latency-hiding win).
#   DDR{q x N_QT tiles, k, v} -> [A: scale*Q.K^T] --ac--> [B: softmax] --probs--> [C: probs.V] -> DDR{ctx}
# k and V are held RESIDENT (acquired once, reused across all N_QT query tiles) -- the conveyor's
# elegance: each stage keeps its weights, streams query tiles through. With depth-2 belts the 3 workers
# RUN CONCURRENTLY: while B softmaxes tile q, A computes q+1 and C finishes q-1 (up to 3 tiles in flight).
# VARIANT=mono builds the same math on ONE tile (all 3 stages sequential per tile) for the A/B baseline.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
import argparse
import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.controlflow import range_

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = np.float16

import os
# Dims overridable via env (real Parakeet dims: TQ=8 T=176 DK=128 N_QT=22). Defaults = validated tiny proto.
TQ = int(os.environ.get("ATTN_TQ", 8))
T = int(os.environ.get("ATTN_T", 64))
DK = int(os.environ.get("ATTN_DK", 64))
N_QT = int(os.environ.get("ATTN_NQT", 16))  # query tiles streamed through the pipeline
N_HEADS = int(os.environ.get("ATTN_HEADS", 1))  # data-parallel heads, one 3-tile conveyor per column


P = 2 * T - 1  # relative-position length (NeMo/Parakeet rel-pos)


def build(dev, mono=False, TRIVIAL=False, relpos=False, bd_onchip=False, tactive_mask=False,
          p_resident=False):
    if bd_onchip:
        relpos = True   # BD-on-chip reuses the relpos scores kernel (q||BD belt -> stage_scores_relpos_bd)
    q_ty = np.ndarray[(TQ * DK,), np.dtype[bfloat16]]      # one query tile (fifo object)
    # relpos (real-dims): query belt carries q[TQ,DK] then host-precomputed rel_shifted BD_shifted[TQ,T],
    # both bf16, in one object -> stage A does AC on-chip + adds BD_shifted (no p resident, no row_off).
    qbd_ty = np.ndarray[(TQ * DK + TQ * T,), np.dtype[bfloat16]]
    k_ty = np.ndarray[(T * DK,), np.dtype[bfloat16]]
    v_ty = np.ndarray[(T * DK,), np.dtype[bfloat16]]
    ac_ty = np.ndarray[(TQ * T,), np.dtype[np.float32]]
    probs_ty = np.ndarray[(TQ * T,), np.dtype[bfloat16]]
    ctx_ty = np.ndarray[(TQ * DK,), np.dtype[bfloat16]]    # one ctx tile (fifo object)
    # RUNTIME (L3) arg types = the WHOLE streamed buffer (N_QT tiles for q/ctx); the shim streams it
    # into the [TQ,*] fifo objects as N_QT blocks. (The fifo OBJECT is one tile; the SEQUENCE ARG is
    # the whole buffer -- mixing these up made N_QT>1 read only the first tile.)
    QELEM = TQ * DK + (TQ * T if relpos else 0)   # per-tile query-belt element count (q [+ BD_shifted])
    q_full_ty = np.ndarray[(N_QT * QELEM,), np.dtype[bfloat16]]
    ctx_full_ty = np.ndarray[(N_QT * TQ * DK,), np.dtype[bfloat16]]

    sfx = "_t" if TRIVIAL else ""   # TRIVIAL: same structure, trivial copy kernels (isolate race)
    # BISECT: TRIVIAL=2 -> only softmax trivial (scores+ctx real); TRIVIAL=3 -> only scores trivial.
    sc_sfx = "_t" if TRIVIAL in (1, 3) else ""
    sm_sfx = "_t" if TRIVIAL in (1, 2) else ""
    cx_sfx = "_t" if TRIVIAL == 1 else ""
    qbelt_ty = qbd_ty if relpos else q_ty  # query belt object: q+BD_shifted (relpos) or q (plain)
    # t_active RTP register (int32[16], use_write_rtp) -- the scores stage reads rtp[0] at runtime to mask
    # pad keys j>=t_active (BD-on-chip has no host belt-sentinel, so the mask must live in the kernel).
    # Only wired for the bd_onchip path (BD-onchip attention design). One MAX-T xclbin serves
    # any t_active<=T; t_active==T is unmasked passthrough (== stage_scores_relpos_bd byte-for-byte).
    rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
    if bd_onchip and tactive_mask:
        scores = Kernel("stage_scores_relpos_bd_mask", "kernels.a", [qbd_ty, k_ty, ac_ty, rtp_ty])
    elif relpos:
        scores = Kernel("stage_scores_relpos_bd", "kernels.a", [qbd_ty, k_ty, ac_ty])
    else:
        scores = Kernel("stage_scores" + sc_sfx, "kernels.a", [q_ty, k_ty, ac_ty])
    softmax = Kernel("stage_softmax" + sm_sfx, "kernels.a", [ac_ty, probs_ty])
    ctx_k = Kernel("stage_ctx" + cx_sfx, "kernels.a", [probs_ty, v_ty, ctx_ty])

    # relpos: the query belt carries q+BD_shifted (~2x bigger) -> depth-1 to fit stage A's L1 alongside
    # the 44 KB resident k + the f32 ac belt (the A->B ac belt keeps depth-2 for pipeline overlap).
    of_q = ObjectFifo(qbelt_ty, name="q", depth=1 if relpos else 2)
    of_ctx = ObjectFifo(ctx_ty, name="ctx", depth=2)
    # ADVANCING taps for q (in) / ctx (out): outer dim N_QT strides by the per-tile element count.
    # relpos: one flat QELEM block per tile (q||BD_shifted contiguous). plain: [TQ,DK].
    q_tap = (TensorAccessPattern([N_QT * QELEM], 0, [N_QT, 1, 1, QELEM], [QELEM, 0, 0, 1]) if relpos
             else TensorAccessPattern([N_QT * TQ * DK], 0, [N_QT, 1, TQ, DK], [TQ * DK, 0, DK, 1]))
    ctx_tap = TensorAccessPattern([N_QT * TQ * DK], 0, [N_QT, 1, TQ, DK], [TQ * DK, 0, DK, 1])
    # stride-0 replay taps (deliver the read-only weight N_QT times, one per query tile).
    replay_tap = TensorAccessPattern([T * DK], 0, [N_QT, 1, T, DK], [0, 0, DK, 1])

    rt = Runtime()
    if mono:
        # MONOLITH baseline: ONE tile, all 3 ops per query tile; q + packed kv (2 inputs = channel budget).
        kv_ty = np.ndarray[(2 * T * DK,), np.dtype[bfloat16]]
        mono_k = Kernel("stage_mono", "kernels.a", [q_ty, kv_ty, ctx_ty])
        of_kv = ObjectFifo(kv_ty, name="kv", depth=2)
        kv_replay = TensorAccessPattern([2 * T * DK], 0, [N_QT, 1, 2 * T, DK], [0, 0, DK, 1])

        def mono_fn(f_q, f_kv, f_ctx, k_mono):
            for _ in range_(N_QT):
                eq = f_q.acquire(1); ekv = f_kv.acquire(1); ec = f_ctx.acquire(1)
                k_mono(eq, ekv, ec)
                f_q.release(1); f_kv.release(1); f_ctx.release(1)

        w = Worker(mono_fn, [of_q.cons(), of_kv.cons(), of_ctx.prod(), mono_k])
        with rt.sequence(q_full_ty, kv_ty, ctx_full_ty) as (Q, KV, CTX):
            rt.start(w)
            rt.fill(of_q.prod(), Q, tap=q_tap)
            rt.fill(of_kv.prod(), KV, tap=kv_replay)
            rt.drain(of_ctx.cons(), CTX, tap=ctx_tap, wait=True)
    elif bd_onchip:
        # BD-ON-CHIP 4-stage column per head (BD -> scores -> softmax -> ctx), H data-parallel columns.
        # BD tile: g_bd = qv @ p^T (f32 accfloat) then rel_shift+split -> q_pass||BD_hi belt. q0 advances
        # via a static tile-counter in the kernel (no scalar arg). p streamed in BD_KB blocks (real T,
        # p=88KB > L1) or resident (small T). ALL-DIRECT streams per head (qpv+p+k+v+ctx = 5/head); fits
        # the shim NOC to ~H=4 (H=4x2 = the spec's 8-head shipping fallback). MemTile grouping for
        # H=8-in-1 = the >48-block kill-if follow-up. p re-read from L3 (movement opt = MemTile-resident later).
        H = N_HEADS
        qpv_ty = np.ndarray[(2 * TQ * DK,), np.dtype[bfloat16]]
        BD_KB = int(os.environ.get("BD_KB", 39))
        stream_p = (P * DK * 2) > 60000
        p_full_ty = np.ndarray[(P * DK,), np.dtype[bfloat16]]
        NBLK = (P // BD_KB) if stream_p else 1
        if stream_p:
            assert P % BD_KB == 0, f"streaming needs P({P}) %% BD_KB({BD_KB}) == 0"
            pblk_ty = np.ndarray[(BD_KB * DK,), np.dtype[bfloat16]]
            block_k = Kernel("bd_block_bake", "kernels.a", [qpv_ty, pblk_ty])
            # t_active-aware emit (bd_emit_bake_ta) when masking: rel_shift base uses t_active (rtp[0]),
            # matching relpos_mha.cc, so short clips get the correct rel-pos alignment (not the BUILT_T base).
            emit_k = (Kernel("bd_emit_bake_ta", "kernels.a", [qpv_ty, qbd_ty, rtp_ty]) if tactive_mask
                      else Kernel("bd_emit_bake", "kernels.a", [qpv_ty, qbd_ty]))
        else:
            assert not tactive_mask, "t_active mask targets the real-dims streaming p path (stream_p)"
            bd_k = Kernel("stage_bd_bake", "kernels.a", [qpv_ty, p_full_ty, qbd_ty])

        # ---- TASK 2 (p-resident-read-once) HOOK -------------------------------------------------------
        # Today (default, KNOWN-GOOD) the p fill is STREAM-A: ONE shim BD re-reads p from L3 N_QT times
        # (the ptap outer dim = N_QT at stride 0, below). That is the 22x p re-read the spec (sec 2c) wants
        # to kill by staging p ONCE in a MemTile (L2, 512 KB; p=88 KB fits) and re-forwarding on-chip.
        # WARNING (banked device evidence, relpos_rowtiled_stream_iron.py KPV-REPLAY note): the naive
        # L2->L1 `.forward(repeat_count=N_QT)` was BUILT but FAILED parity on device (corr 0.65, rel-L2
        # 0.82) -- the MemTile replay did NOT restart per query tile, so tile q saw the wrong p-blocks. The
        # per-query-tile rt.fill loop (N_QT calls) is correct-in-principle but emits N_QT shim BDs > the
        # 16-BD tile budget. So p_resident is a DEVICE-ITERATED step, not a drop-in: it needs a MemTile
        # replay that restarts its L2 read per tile (candidate: split the p MemTile fifo so the L2->L1 hop
        # re-anchors at p offset 0 each tile; verify with run_bd_onchip.py rel-L2<=5e-3 at H=1 BEFORE H=4).
        # Guarded off so the default build stays the validated STREAM-A path; flip via --p-resident once
        # the topology below is proven on device.
        assert not p_resident, (
            "p-resident-read-once is a device-iterated MemTile step (see TASK 2 HOOK + turnkey doc); "
            "the naive L2->L1 forward FAILED parity on device. Build without --p-resident (STREAM-A) "
            "until the restart-per-tile MemTile replay is validated.")
        of_qpv = [ObjectFifo(qpv_ty, name=f"qpv{h}", depth=2) for h in range(H)]
        of_p = [ObjectFifo(pblk_ty if stream_p else p_full_ty, name=f"p{h}", depth=2 if stream_p else 1) for h in range(H)]
        of_bd = [ObjectFifo(qbd_ty, name=f"bd{h}", depth=1) for h in range(H)]
        of_k = [ObjectFifo(k_ty, name=f"k{h}", depth=1) for h in range(H)]
        of_v = [ObjectFifo(v_ty, name=f"v{h}", depth=1) for h in range(H)]
        of_ac = [ObjectFifo(ac_ty, name=f"ac{h}", depth=2) for h in range(H)]
        of_pr = [ObjectFifo(probs_ty, name=f"probs{h}", depth=2) for h in range(H)]
        of_ctxh = [ObjectFifo(ctx_ty, name=f"ctx{h}", depth=2) for h in range(H)]
        # TASK 1 (t_active in-kernel key-mask): per-head RTP register + runtime barrier. The scores worker
        # reads rtp[0]=t_active and masks pad keys j>=t_active. One rtp buffer per head column (each head's
        # scores stage is its own core); all set to the same t_active per dispatch. Mirrors the proven
        # relpos_rowtiled_stream_iron.py STEP-C rtp barrier. Inert when --tactive-mask is off.
        tactive_rtp = [Buffer(rtp_ty, name=f"tactive_rtp{h}", use_write_rtp=True) for h in range(H)] if tactive_mask else None
        rtp_bar = [WorkerRuntimeBarrier() for h in range(H)] if tactive_mask else None
        # separate rtp for the BD (emit) stage core -- use_write_rtp is per-core, so the scores core and
        # the BD core each need their own t_active register (both set to the same value per dispatch).
        tactive_bd_rtp = [Buffer(rtp_ty, name=f"tactive_bd_rtp{h}", use_write_rtp=True) for h in range(H)] if tactive_mask else None
        rtp_bd_bar = [WorkerRuntimeBarrier() for h in range(H)] if tactive_mask else None

        def bd_stream_w(f_qpv, f_p, f_bd, k_block, k_emit):
            for _ in range_(N_QT):
                eqpv = f_qpv.acquire(1)
                for _ in range_(NBLK):
                    epb = f_p.acquire(1); k_block(eqpv, epb); f_p.release(1)
                ebd = f_bd.acquire(1); k_emit(eqpv, ebd)
                f_qpv.release(1); f_bd.release(1)

        # t_active-aware BD worker: waits for the runtime rtp write, then passes rtp into the emit kernel.
        def bd_stream_w_ta(f_qpv, f_p, f_bd, k_block, k_emit, rtp, bar):
            bar.wait_for_value(1)
            for _ in range_(N_QT):
                eqpv = f_qpv.acquire(1)
                for _ in range_(NBLK):
                    epb = f_p.acquire(1); k_block(eqpv, epb); f_p.release(1)
                ebd = f_bd.acquire(1); k_emit(eqpv, ebd, rtp)
                f_qpv.release(1); f_bd.release(1)

        def bd_res_w(f_qpv, f_p, f_bd, k_bd):
            ep = f_p.acquire(1)
            for _ in range_(N_QT):
                eqpv = f_qpv.acquire(1); ebd = f_bd.acquire(1); k_bd(eqpv, ep, ebd)
                f_qpv.release(1); f_bd.release(1)
            f_p.release(1)

        def stg_a(f_bd, f_k, f_ac, k_sc):
            ek = f_k.acquire(1)
            for _ in range_(N_QT):
                ebd = f_bd.acquire(1); eac = f_ac.acquire(1); k_sc(ebd, ek, eac)
                f_bd.release(1); f_ac.release(1)
            f_k.release(1)

        # t_active-masked scores worker: waits for the runtime to write rtp[0]=t_active, then passes rtp
        # into the mask kernel each tile so pad keys j>=t_active are nulled in the softmax.
        def stg_a_mask(f_bd, f_k, f_ac, k_sc, rtp, bar):
            bar.wait_for_value(1)
            ek = f_k.acquire(1)
            for _ in range_(N_QT):
                ebd = f_bd.acquire(1); eac = f_ac.acquire(1); k_sc(ebd, ek, eac, rtp)
                f_bd.release(1); f_ac.release(1)
            f_k.release(1)

        def stg_b(f_ac, f_probs, k_sm):
            for _ in range_(N_QT):
                eac = f_ac.acquire(1); ep2 = f_probs.acquire(1); k_sm(eac, ep2)
                f_ac.release(1); f_probs.release(1)

        def stg_c(f_probs, f_v, f_ctx, k_cx):
            ev = f_v.acquire(1)
            for _ in range_(N_QT):
                ep2 = f_probs.acquire(1); ec = f_ctx.acquire(1); k_cx(ep2, ev, ec)
                f_probs.release(1); f_ctx.release(1)
            f_v.release(1)

        wl = []
        for h in range(H):
            if stream_p and tactive_mask:
                wl.append(Worker(bd_stream_w_ta, [of_qpv[h].cons(), of_p[h].cons(), of_bd[h].prod(),
                                                  block_k, emit_k, tactive_bd_rtp[h], rtp_bd_bar[h]], stack_size=0x1000))
            elif stream_p:
                wl.append(Worker(bd_stream_w, [of_qpv[h].cons(), of_p[h].cons(), of_bd[h].prod(), block_k, emit_k], stack_size=0x1000))
            else:
                wl.append(Worker(bd_res_w, [of_qpv[h].cons(), of_p[h].cons(), of_bd[h].prod(), bd_k], stack_size=0x1000))
            if tactive_mask:
                wl.append(Worker(stg_a_mask, [of_bd[h].cons(), of_k[h].cons(), of_ac[h].prod(), scores,
                                              tactive_rtp[h], rtp_bar[h]]))
            else:
                wl.append(Worker(stg_a, [of_bd[h].cons(), of_k[h].cons(), of_ac[h].prod(), scores]))
            wl.append(Worker(stg_b, [of_ac[h].cons(), of_pr[h].prod(), softmax], stack_size=0x1000))
            wl.append(Worker(stg_c, [of_pr[h].cons(), of_v[h].cons(), of_ctxh[h].prod(), ctx_k]))

        QPVE = N_QT * 2 * TQ * DK
        qpv_all_ty = np.ndarray[(H * QPVE,), np.dtype[bfloat16]]
        p_all_ty = np.ndarray[(H * P * DK,), np.dtype[bfloat16]]
        k_all_ty = np.ndarray[(H * T * DK,), np.dtype[bfloat16]]
        v_all_ty = np.ndarray[(H * T * DK,), np.dtype[bfloat16]]
        c_all_ty = np.ndarray[(H * N_QT * TQ * DK,), np.dtype[bfloat16]]
        with rt.sequence(qpv_all_ty, p_all_ty, k_all_ty, v_all_ty, c_all_ty) as (QPV, PP, K, V, CTX):
            rt.start(*wl)
            if tactive_mask:
                # Bake t_active = T (full length) as the DEFAULT immediate; the host patches this word in
                # insts.bin per dispatch for shorter clips (mirrors RELPOS_TACTIVE_WORD). Unpatched = T =
                # unmasked passthrough (correct for T==BUILT_T + the run_bd_onchip.py standalone gate).
                def _mk_set(val):
                    def _set(p):
                        p[0] = val
                    return _set
                for h in range(H):
                    rt.inline_ops(_mk_set(T), [tactive_rtp[h]])
                    rt.set_barrier(rtp_bar[h], 1)
                    rt.inline_ops(_mk_set(T), [tactive_bd_rtp[h]])   # BD-stage core: same t_active
                    rt.set_barrier(rtp_bd_bar[h], 1)
            for h in range(H):
                rt.fill(of_qpv[h].prod(), QPV, tap=TensorAccessPattern([H * QPVE], h * QPVE, [N_QT, 1, 1, 2 * TQ * DK], [2 * TQ * DK, 0, 0, 1]))
                if stream_p:
                    ptap = TensorAccessPattern([H * P * DK], h * P * DK, [N_QT, NBLK, BD_KB, DK], [0, BD_KB * DK, DK, 1])
                else:
                    ptap = TensorAccessPattern([H * P * DK], h * P * DK, [N_QT, 1, P, DK], [0, 0, DK, 1])
                rt.fill(of_p[h].prod(), PP, tap=ptap)
                kvtap = TensorAccessPattern([H * T * DK], h * T * DK, [N_QT, 1, T, DK], [0, 0, DK, 1])
                rt.fill(of_k[h].prod(), K, tap=kvtap)
                rt.fill(of_v[h].prod(), V, tap=kvtap)
                rt.drain(of_ctxh[h].cons(), CTX, tap=TensorAccessPattern([H * N_QT * TQ * DK], h * N_QT * TQ * DK, [N_QT, 1, TQ, DK], [TQ * DK, 0, DK, 1]), wait=True)
    else:
        # k, V are read-only weights. At real dims (T*DK bf16 = 44 KB) a depth-2 weight fifo blows the
        # 64 KB L1, so depth-1. Structure = the validated per-tile-acquire + stride-0 replay tap (the
        # tiny-dims proven path), only single-buffered. (True acquire-once residency deadlocked at
        # depth-1 -- deferred; this re-streams weights on-chip but is the fair, known-good dataflow.)
        # MULTI-HEAD: replicate the 3-tile conveyor per head; place-tiles assigns each head its own
        # column (data-parallel heads, no cross-head data). Per-head fifos (unique names).
        H = N_HEADS
        qd = 1 if relpos else 2
        of_vh = [ObjectFifo(v_ty, name=f"v{h}", depth=1) for h in range(H)]
        of_ach = [ObjectFifo(ac_ty, name=f"ac{h}", depth=2) for h in range(H)]
        of_ph = [ObjectFifo(probs_ty, name=f"probs{h}", depth=2) for h in range(H)]
        # 8-head fit: split q AND k (v stays per-head DIRECT) + ctx JOIN = 3 MemTile ops/group. Shim math
        # at H=8: q-split 2 + k-split 2 + v-direct 8 + ctx-join 2 = 14 streams (under budget). 4 MemTile
        # ops (q+k+v split + join) DEADLOCKED at runtime; 2 (q-split+join) worked -> 3 is the target.
        GJ = 4
        groups = [list(range(g, min(g + GJ, H))) for g in range(0, H, GJ)]
        q_subs = [None] * H; k_subs = [None] * H; ctx_subs = [None] * H
        of_q_bigs, of_k_bigs, of_ctx_bigs = [], [], []
        for gi, hs in enumerate(groups):
            gsz = len(hs)
            ofq = ObjectFifo(np.ndarray[(gsz * QELEM,), np.dtype[bfloat16]], name=f"qbig{gi}", depth=2)
            qs = ofq.cons().split([i * QELEM for i in range(gsz)], obj_types=[qbelt_ty] * gsz,
                                  depths=[qd] * gsz, names=[f"qs{gi}_{i}" for i in range(gsz)])
            ofk = ObjectFifo(np.ndarray[(gsz * T * DK,), np.dtype[bfloat16]], name=f"kbig{gi}", depth=2)
            ks = ofk.cons().split([i * T * DK for i in range(gsz)], obj_types=[k_ty] * gsz,
                                  depths=[1] * gsz, names=[f"ks{gi}_{i}" for i in range(gsz)])
            ofb = ObjectFifo(np.ndarray[(gsz * TQ * DK,), np.dtype[bfloat16]], name=f"ctxbig{gi}", depth=2)
            cs = ofb.prod().join([i * TQ * DK for i in range(gsz)],
                                 obj_types=[ctx_ty] * gsz, names=[f"ctxj{gi}_{i}" for i in range(gsz)])
            of_q_bigs.append((ofq, hs)); of_k_bigs.append((ofk, hs)); of_ctx_bigs.append((ofb, hs))
            for i, h in enumerate(hs):
                q_subs[h] = qs[i]; k_subs[h] = ks[i]; ctx_subs[h] = cs[i]

        def stage_a(f_q, f_k, f_ac, k_sc):
            ek = f_k.acquire(1)          # DIAGNOSTIC: acquire-once k (test acquire-once in isolation, no split)
            for _ in range_(N_QT):
                eq = f_q.acquire(1); eac = f_ac.acquire(1)
                k_sc(eq, ek, eac)
                f_q.release(1); f_ac.release(1)
            f_k.release(1)

        def stage_b(f_ac, f_probs, k_sm):
            for _ in range_(N_QT):
                eac = f_ac.acquire(1); ep = f_probs.acquire(1)
                k_sm(eac, ep)
                f_ac.release(1); f_probs.release(1)

        def stage_c(f_probs, f_v, f_ctx, k_cx):
            ev = f_v.acquire(1)          # DIAGNOSTIC: acquire-once v
            for _ in range_(N_QT):
                ep = f_probs.acquire(1); ec = f_ctx.acquire(1)
                k_cx(ep, ev, ec)
                f_probs.release(1); f_ctx.release(1)
            f_v.release(1)

        # stage_b (softmax) holds `float srow[T]` on the AIE stack -> bump its stack_size (real T
        # overflows the 1 KB default -> silent hang). stage A (relpos BD-in-belt) / C need no bump.
        wl = []
        for h in range(H):
            wl += [Worker(stage_a, [q_subs[h].cons(), k_subs[h].cons(), of_ach[h].prod(), scores]),
                   Worker(stage_b, [of_ach[h].cons(), of_ph[h].prod(), softmax], stack_size=0x1000),
                   Worker(stage_c, [of_ph[h].cons(), of_vh[h].cons(), ctx_subs[h].prod(), ctx_k])]

        # q GROUP-MAJOR (split): per group [N_QT, gsz, QELEM]. k/v head-major, filled ONCE (acquire-once).
        KT, VT = T * DK, T * DK
        q_all_ty = np.ndarray[(N_QT * H * QELEM,), np.dtype[bfloat16]]
        k_all_ty = np.ndarray[(H * KT,), np.dtype[bfloat16]]
        v_all_ty = np.ndarray[(H * VT,), np.dtype[bfloat16]]
        c_all_ty = np.ndarray[(N_QT * H * TQ * DK,), np.dtype[bfloat16]]
        with rt.sequence(q_all_ty, k_all_ty, v_all_ty, c_all_ty) as (Q, K, V, CTX):
            rt.start(*wl)
            for h in range(H):   # v stays per-head direct, filled ONCE (acquire-once)
                vh = TensorAccessPattern([H * VT], h * VT, [1, 1, T, DK], [0, 0, DK, 1])
                rt.fill(of_vh[h].prod(), V, tap=vh)
            qoff = koff = 0
            for (ofq, hs), (ofk, _) in zip(of_q_bigs, of_k_bigs):
                gsz = len(hs)
                qtap = TensorAccessPattern([N_QT * H * QELEM], qoff, [N_QT, 1, 1, gsz * QELEM], [gsz * QELEM, 0, 0, 1])
                ktap = TensorAccessPattern([H * KT], koff, [1, 1, 1, gsz * T * DK], [0, 0, 0, 1])  # k once, group slice
                rt.fill(ofq.prod(), Q, tap=qtap)
                rt.fill(ofk.prod(), K, tap=ktap)
                qoff += N_QT * gsz * QELEM; koff += gsz * T * DK
            coff = 0   # per-group ctx drain (each group -> its own shim drain, contiguous slice of C_all)
            for ofb, hs in of_ctx_bigs:
                gsz = len(hs)
                gtap = TensorAccessPattern([N_QT * H * TQ * DK], coff,
                                           [N_QT, 1, 1, gsz * TQ * DK], [gsz * TQ * DK, 0, 0, 1])
                rt.drain(ofb.cons(), CTX, tap=gtap, wait=True)
                coff += N_QT * gsz * TQ * DK

    return Program(dev, rt).resolve_program()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dev", required=True, dest="device")
ap.add_argument("--mono", action="store_true", help="single-tile baseline (all 3 stages on 1 tile)")
ap.add_argument("--trivial", type=int, default=0, help="0=real; 1=all trivial; 2=softmax trivial; 3=scores trivial")
ap.add_argument("--relpos", action="store_true", help="fused relpos scores stage (on-chip AC+BD+rel_shift)")
ap.add_argument("--relpos-bd-onchip", dest="bd_onchip", action="store_true",
                help="BD on-chip as a 4th stage (BD->scores->softmax->ctx); H=1 N_QT=1 arithmetic gate")
ap.add_argument("--tactive-mask", dest="tactive_mask", action="store_true",
                help="in-kernel t_active RTP key-mask on the bd_onchip scores stage (variable-length clips)")
ap.add_argument("--p-resident", dest="p_resident", action="store_true",
                help="[device-iterated] stage p once in a MemTile, re-stream on-chip (kills the 22x L3 re-read)")
opts = ap.parse_args(sys.argv[1:])
dev = NPU2() if opts.device == "npu2" else NPU1()
print(build(dev, mono=opts.mono, TRIVIAL=opts.trivial, relpos=opts.relpos, bd_onchip=opts.bd_onchip,
            tactive_mask=opts.tactive_mask, p_resident=opts.p_resident))
