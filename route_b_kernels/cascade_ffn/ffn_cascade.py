# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cascade-FFN Phase 0 (Task 4): the Whisper decode FFN as ONE on-chip air.launch.
#
#   x[768] -> LayerNorm -> fc1(768->3072) -> +bias_fc1 -> GELU(tanh)
#          -> fc2(3072->768) K-reduce + b_fc2 -> out[768]
#
# This computes gen_ffn.py's EXACT function (LN -> fc1 -> +bias -> GELU -> fc2 ->
# +b_fc2; NO decode-block +x -- that is a separate op outside the FFN span, see
# STRUCTURE.md B.4 correction) so the rel-L2 gate is apples-to-apples.
#
# Structure (STRUCTURE.md B.5): ONE air.launch -> ONE air.segment -> ONE herd
# sizes=[8, 1] (8 cores in one row across 8 columns). The 8-way K-reduction
# rides a W->E npu_cascade ACROSS the columns (axis tx) -- an AIE2P column has
# only 4 rows so an 8-deep vertical cascade does not place; this is the int4
# [N,1] horizontal idiom. HEAD = tx==0 (seeds b_fc2), TAIL = tx==7 (writes out).
#
# DATA MOVEMENT (int4 o_gemv_ffn_int4_fused idiom): a per-core [384,768] weight
# slab is 576KB >> the 64KB AIE2P L1 AND the 8 slabs together exceed the 512KB
# L2, so weights cannot be made resident -- they STREAM L3->L2->L1 in [8,*]
# tiles. The L3-side read is a launch-level ChannelPut (one consolidated shim BD
# per col, NOT a per-core direct-from-L3 dma which over-subscribes the shim DMA);
# the segment relays L2->L1 tile-by-tile; the herd ChannelGets one tile per iter.
# Output is assembled by TAIL into L2 then drained L2->L3 by the segment.
#
# Per core tx in 0..7, all intermediates on-chip:
#   1. x_norm = LayerNorm(x)   (non-affine; affine folded host-side into
#      Wfc1/bias_fc1 per gen_ffn 75-78; x broadcast to all cols)
#   2. fc1: for t in 0..47: get [8,768] Wfc1 tile -> matvec_fc1_tile_bf16_store ->
#           scatter 8 rows into h_ty[384]   (this core's 384-row fc1 slab)
#   3. h_ty += bias_fc1[tx*384 : tx*384+384] (inline, BEFORE GELU)
#   4. gelu_tile_bf16(384, h_ty)            (ONCE over the full 384 slab)
#   5. fc2: for t in 0..95: get [8,384] Wfc2 col-block tile ->
#           matvec_fc2_tile_bf16_store -> scatter 8 rows into partial_ty[768]
#      (h_ty[384] is exactly this core's fc2 K-chunk -> stays in L1, never L3)
#   6. cascade-reduce partial_ty[768] W->E: HEAD seeds acc = partial + b_fc2
#      (static head-inject R), MIDDLE get/add/forward, TAIL get/add -> out[768].
#
# Links route_b_kernels/cascade_ffn/mv_bf16_gelu.o (Task 3): matvec_fc1_tile_bf16_store
# (<8,768>), matvec_fc2_tile_bf16_store (<8,384>), gelu_tile_bf16, partial_plus_r_bf16.

import argparse
import os

from air.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineMapAttr,
    AffineSymbolExpr,
    BF16Type,
    BoolAttr,
    F32Type,
    InsertionPoint,
    IntegerAttr,
    MemRefType,
    StringAttr,
    UnitAttr,
    VectorType,
)
from air.dialects.affine import apply as affine_apply
from air.dialects.air import (
    Channel,
    ChannelGet,
    ChannelPut,
    MemorySpace,
    T,
    dma_memcpy_nd,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects.air import channel as channel_decl
from air.dialects.func import FuncOp, CallOp
from air.dialects.memref import (
    AllocOp,
    DeallocOp,
    subview,
    load as memref_load,
    store as memref_store,
)
from air.dialects import arith, scf, math as math_dialect
from air.dialects.scf import for_, yield_
from air.dialects.vector import (
    transfer_read,
    transfer_write,
    BroadcastOp,
    reduction as vector_reduction,
)
from air.backend.xrt import XRTBackend

KERNEL_OBJ = "mv_bf16_gelu.o"


def _map_mul(coeff):
    return AffineMap.get(
        0, 1, [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(coeff))]
    )


def _map_mul_add(c0, c1):
    return AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(c0)),
                AffineExpr.get_mul(AffineSymbolExpr.get(1), AffineConstantExpr.get(c1)),
            )
        ],
    )


def _map_mul_plus(coeff, k):
    return AffineMap.get(
        0,
        1,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(coeff)),
                AffineConstantExpr.get(k),
            )
        ],
    )


@module_builder
def build_module(D, FF, NCORES, M_INPUT, EPS, M_TILE):
    # M_TILE = activation rows (sequence positions) processed per dispatch. The
    # FFN is GEMM now (weight tile reused across all M_TILE rows), not the M=1
    # GEMV. The host re-dispatches this ELF ceil(T_enc/M_TILE) times per layer
    # (re-entrant via the mlir-air load_pdi cascade reset, ELF path). M_TILE is
    # L1-bound: the full-tile cascade payload partial+recv = 2*M_TILE*D*2 bytes
    # must fit the 64KB L1 (M_TILE=16 -> 48KB peak). Larger M_TILE needs cascade
    # row-sub-tiling (Phase 1 scaling). See internal notes fused-ffn-phase0-gate.
    M_SLAB = FF // NCORES            # 384: fc1 out rows/core == fc2 K-chunk/core
    assert FF % NCORES == 0 and M_SLAB % M_INPUT == 0 and D % M_INPUT == 0
    assert M_SLAB % 16 == 0 and D % 16 == 0 and D % 32 == 0  # bf16 legalize + cascade payload
    assert M_TILE >= 1
    FC1_TILES = M_SLAB // M_INPUT   # 48
    FC2_TILES = D // M_INPUT        # 96
    # fc1 weight tiles are K-AUGMENTED: D=768 reduction cols + a 32-wide bias block
    # (col D = bias_fc1[oc], D+1..D+32 = 0). The fc1 GEMM kernel folds the bias in
    # (vector mac) so bias is NOT a separate inL2L1 transfer -- that multiplexing
    # inflated the w1 objectFIFO lock and overran the weight buffer at M_TILE>1
    # (see internal notes mtile-ffn-phase1-ln-fix). FC1_WROW must match the kernel.
    FC1_WROW = D + 32                # 800
    N_CASCADE = NCORES

    # --- Per-stage latency-attribution stubs (measure-first; latency only, NOT
    # correct). FFN_STUB_GEMM=1 keeps the inL2L1 weight stream (no relay deadlock)
    # but skips the fc1/fc2/gelu compute -> slope_full - slope_stubgemm = GEMM+gelu
    # per-row cost. FFN_STUB_CASCADE=1 drops the 8-hop W->E serial reduction (TAIL
    # writes its own wbuf) -> slope_full - slope_stubcascade = cascade per-row cost.
    # See internal notes (the 150 us/row attribution).
    STUB_GEMM = os.environ.get("FFN_STUB_GEMM") == "1"
    STUB_CASCADE = os.environ.get("FFN_STUB_CASCADE") == "1"

    bf16_ty = BF16Type.get()
    f32_ty = F32Type.get()
    l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l2_ms = IntegerAttr.get(T.i32(), MemorySpace.L2)

    # L3 / DDR func args (host buffer layout -- see module docstring + report).
    # 5 host BOs (the aiecc kernel-ABI JSON hardcaps host BOs at 5): the two
    # static bias vectors are folded into ONE `biases` buffer of length FF+D:
    #   biases[0:FF]      = bias_fc1 = bf@Wfc1 + b_fc1   (per-core +bias before GELU)
    #   biases[FF:FF+D]   = b_fc2                         (cascade-HEAD residual)
    # x/out are now [M_TILE, D] (the M-tile activation/output batch); weights and
    # biases are M-independent (reused across all rows).
    x_l3_ty = MemRefType.get([M_TILE, D], bf16_ty)
    Wfc1_l3_ty = MemRefType.get([FF, FC1_WROW], bf16_ty)  # K-aug: (gf*Wfc1).T ++ bias col
    biases_l3_ty = MemRefType.get([FF + D], bf16_ty)     # bias_fc1 ++ b_fc2  (3840)
    Wfc2_l3_ty = MemRefType.get([D, FF], bf16_ty)        # Wfc2.T (= mat_fc2)
    out_l3_ty = MemRefType.get([M_TILE, D], bf16_ty)

    # L1 buffer types.
    actMT_l1 = MemRefType.get([M_TILE, D], bf16_ty, memory_space=l1_ms)      # x/xnorm/partial/recv
    hMT_l1 = MemRefType.get([M_TILE, M_SLAB], bf16_ty, memory_space=l1_ms)   # fc1 out slab / fc2 in
    vecD_l1 = MemRefType.get([D], bf16_ty, memory_space=l1_ms)               # b_fc2 broadcast vec
    vecSlab_l1 = MemRefType.get([M_SLAB], bf16_ty, memory_space=l1_ms)       # bias_fc1 slab
    w1tile_l1 = MemRefType.get([M_INPUT, FC1_WROW], bf16_ty, memory_space=l1_ms)  # [8,800] K-aug
    w2tile_l1 = MemRefType.get([M_INPUT, M_SLAB], bf16_ty, memory_space=l1_ms)  # [8,384]
    # L2 staging buffers (one tile each, ping-ponged by the relay).
    w1tile_l2 = MemRefType.get([M_INPUT, FC1_WROW], bf16_ty, memory_space=l2_ms)
    w2tile_l2 = MemRefType.get([M_INPUT, M_SLAB], bf16_ty, memory_space=l2_ms)
    bias_l2_ty = MemRefType.get([M_SLAB], bf16_ty, memory_space=l2_ms)
    out_l2_ty = MemRefType.get([M_TILE, D], bf16_ty, memory_space=l2_ms)

    # --- Channels ---
    # Broadcast x to all cols. Each col's weights ride ONE multiplexed L3->L2 shim
    # channel colL3L2[c] (w1 tiles, then bias, then w2 tiles -- in herd-consume
    # order; matvec_cascade_add multiplexes R+A on one channel, so mixed shapes on
    # one channel is fine). The segment relay fans them out to the cheap memtile
    # L2->L1 channels (w1/bias/w2). Keeping ONE L3 channel per col (not 3) is what
    # fits the shim NOC DMA budget. b_fc2 -> col0; npu_cascade W->E along columns.
    # x: its own broadcast DMA channel into every core. ALL other per-core inputs
    # (w1 tiles, bias, w2 tiles, and col0's b_fc2) are consumed SEQUENTIALLY, so
    # they share ONE reused inL2L1[c] core DMA channel -- AIE2P cores have only ~2
    # input DMA channels, so x + inL2L1 = the 2-channel budget (mixed shapes on one
    # channel is fine; matvec_cascade_add reuses one channel for R + A).
    Channel("inX", size=[1, 1], broadcast_shape=[NCORES, 1])
    channel_decl("colL3L2", size=[NCORES])
    channel_decl("inL2L1", size=[NCORES])
    channel_decl("chan_cascade", size=[N_CASCADE - 1], channel_type="npu_cascade")

    # --- Private kernel decls (link mv_bf16_gelu.o) ---
    def _kdecl(name, in_tys):
        f = FuncOp(name, (in_tys, []), visibility="private")
        f.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)
        f.attributes["llvm.emit_c_interface"] = UnitAttr.get()
        return f

    # LayerNorm over M_TILE rows: (i32 m_act, i32 n, f32 eps, x[M_TILE,D], xnorm[M_TILE,D])
    ln_func = _kdecl("layernorm_rows_bf16", [T.i32(), T.i32(), f32_ty, actMT_l1, actMT_l1])
    # GEMM tiles: (i32 m_act, i32 col_off, w_tile, act[M_TILE,*], out[M_TILE,*]).
    # fc1: w[8,768] @ xnorm[M_TILE,768] -> h_slab[M_TILE,384] at col_off.
    fc1_func = _kdecl("gemm_fc1_tile_bf16", [T.i32(), T.i32(), w1tile_l1, actMT_l1, hMT_l1])
    # fc2: w[8,384] @ h_slab[M_TILE,384] -> partial[M_TILE,768] at col_off.
    # fc2 reads the SHARED [M_INPUT,FC1_WROW] weight buffer (w2 lands in its first
    # M_SLAB cols, row stride FC1_WROW) -> decl uses w1tile_l1, kernel uses WROW=FC1_WROW.
    fc2_func = _kdecl("gemm_fc2_tile_bf16", [T.i32(), T.i32(), w1tile_l1, hMT_l1, actMT_l1])
    # broadcast-bias adds (single memref type each): +bias_fc1 on h, +b_fc2 on partial.
    bias_slab_func = _kdecl("add_bias_bcast_slab_bf16", [T.i32(), T.i32(), hMT_l1, vecSlab_l1])
    bias_d_func = _kdecl("add_bias_bcast_d_bf16", [T.i32(), T.i32(), actMT_l1, vecD_l1])
    # GELU once over the full [M_TILE,M_SLAB] flat slab (runtime n = M_TILE*M_SLAB).
    gelu_func = _kdecl("gelu_tile_bf16", [T.i32(), hMT_l1])
    # partial_plus_r_bf16(n, partial, r_full, offset, d): cascade middle/tail add,
    # elementwise over the flat [M_TILE,D] payload (n = M_TILE*D, offset 0).
    ppr_func = _kdecl(
        "partial_plus_r_bf16", [T.i32(), actMT_l1, actMT_l1, T.i32(), actMT_l1]
    )

    # Host BO order (group_id = arg index + 3): bo0/x=gid3, bo1/Wfc1=gid4,
    # bo2/biases=gid5, bo3/Wfc2=gid6, bo4/out=gid7. (5 BOs, fits the aiecc cap.)
    @FuncOp.from_py_func(
        x_l3_ty, Wfc1_l3_ty, biases_l3_ty, Wfc2_l3_ty, out_l3_ty
    )
    def ffn_cascade(x_a, w1_a, biases_a, w2_a, out_a):
        @launch(sizes=[1, 1], operands=[x_a, w1_a, biases_a, w2_a, out_a])
        def launch_body(lx, ly, lsx, lsy, x_l3, w1_l3, biases_l3, w2_l3, out_l3):
            # L3-side reads (consolidated shim BDs). The inX broadcast carries TWO
            # transfers to every core: x[M_TILE,D] (the activation batch, for LN +
            # fc1) then b_fc2[D] (the cascade-HEAD residual). b_fc2 rides inX -- the
            # proven-reliable channel every core already reads -- NOT the weight
            # stream: multiplexing the odd b_fc2 onto core0's colL3L2/inL2L1
            # corrupted core0's weights. inX is already one of each core's 2 input
            # DMA channels, so this adds NO channel (wall #4 stays clear). The weight
            # stream stays uniform [w1x48, bias, w2x96] across all 8 cores.
            ChannelPut("inX", x_l3, offsets=[0, 0], sizes=[M_TILE, D], strides=[D, 1])
            # b_fc2[D] read as [M_TILE,D] via stride-0 row replication, so this
            # transfer matches x's [M_TILE,D] size on the shared inX broadcast
            # (unequal transfers on one broadcast channel corrupt -- the M_TILE>1 bug).
            ChannelPut("inX", biases_l3, offsets=[FF], sizes=[M_TILE, D], strides=[0, 1])
            for c in range(NCORES):
                ci = arith.ConstantOp.create_index(c)
                # ONE multiplexed shim stream per col, in herd-consume order:
                # Wfc1 col slab rows [c*384:+384] of [FF,FC1_WROW] as 48 x [8,800].
                # The bias is baked into col D of each weight row (K-aug), so there
                # is NO separate bias transfer on this stream anymore.
                ChannelPut(
                    "colL3L2", w1_l3, indices=[ci],
                    offsets=[c * FC1_TILES, 0, 0],
                    sizes=[FC1_TILES, M_INPUT, FC1_WROW],
                    strides=[M_INPUT * FC1_WROW, FC1_WROW, 1],
                )
                # ... then Wfc2 col-block cols [c*384:+384] of [D,FF] as 96 x [8,384].
                ChannelPut(
                    "colL3L2", w2_l3, indices=[ci],
                    offsets=[0, 0, c * M_SLAB],
                    sizes=[FC2_TILES, M_INPUT, M_SLAB],
                    strides=[M_INPUT * FF, FF, 1],
                )

            @segment(name="ffn_seg", operands=[out_l3])
            def segment_body(out_s):
                out_l2 = AllocOp(out_l2_ty, [], [])

                # L2 relay: drain the multiplexed colL3L2 stream in herd-consume
                # order (w1 tiles, then w2 tiles) and fan out to the memtile L2->L1
                # channels. bias is K-aug'd into the w1 tiles (no separate transfer).
                # FIFO order on colL3L2 == consume order -> no cross-channel deadlock.
                for c in range(NCORES):
                    ci = arith.ConstantOp.create_index(c)
                    for _ in for_(FC1_TILES):
                        t2 = AllocOp(w1tile_l2, [], [])
                        ChannelGet("colL3L2", t2.result, indices=[ci])
                        ChannelPut("inL2L1", t2.result, indices=[ci])
                        DeallocOp(t2)
                        yield_([])
                    for _ in for_(FC2_TILES):
                        t2 = AllocOp(w2tile_l2, [], [])
                        ChannelGet("colL3L2", t2.result, indices=[ci])
                        ChannelPut("inL2L1", t2.result, indices=[ci])
                        DeallocOp(t2)
                        yield_([])

                @herd(name="ffn_herd", sizes=[N_CASCADE, 1], operands=[out_l2.result])
                def herd_body(tx, ty, sx, sy, _out_l2):
                    c0 = arith.ConstantOp.create_index(0)
                    c1 = arith.ConstantOp.create_index(1)
                    c8 = arith.ConstantOp.create_index(M_INPUT)
                    last_ty = arith.ConstantOp.create_index(N_CASCADE - 1)

                    m_tile_i32 = arith.constant(T.i32(), M_TILE)
                    n_d_i32 = arith.constant(T.i32(), D)
                    n_slab_i32 = arith.constant(T.i32(), M_SLAB)
                    gelu_n_i32 = arith.constant(T.i32(), M_TILE * M_SLAB)   # GELU flat len
                    ppr_n_i32 = arith.constant(T.i32(), M_TILE * D)         # cascade flat len
                    off0_i32 = arith.constant(T.i32(), 0)
                    eps_f32 = arith.ConstantOp(f32_ty, float(EPS)).result

                    # L1-CAPACITY: the AIE static allocator does NOT reuse a
                    # DeallocOp'd buffer for a later AllocOp -- it sums every live
                    # AllocOp -- and weight tiles double-buffer (~36KB). So we ALIAS
                    # one working buffer `wbuf` through the disjoint lifetimes
                    # x -> xnorm (LN in-place) -> partial (fc2 out / cascade), and
                    # keep one `rbuf` for cascade recv. Two [M_TILE,D] buffers, not
                    # four -- the L1 fit that sets the all-L1 M_TILE ceiling.
                    wbuf = AllocOp(actMT_l1, [], [])       # x -> xnorm -> partial
                    wbuf.attributes["air.shrinkage"] = BoolAttr.get(False)
                    rbuf = AllocOp(actMT_l1, [], [])       # cascade recv
                    rbuf.attributes["air.shrinkage"] = BoolAttr.get(False)
                    h_l1 = AllocOp(hMT_l1, [], [])
                    bfc2_l1 = AllocOp(actMT_l1, [], [])  # b_fc2 replicated to [M_TILE,D]
                    # ONE shared weight buffer for BOTH the fc1 and fc2 streams. Using a
                    # single AllocOp (not a per-loop one) makes aircc give the inL2L1
                    # weight objectFIFO exactly ONE buffer -> free-lock init=1 -> strict
                    # producer/consumer alternation, so the L2->L1 producer can NEVER
                    # overrun the in-flight tile (the M_TILE>1 bug was aircc merging the
                    # per-loop w1+w2 buffers into one over-credited FIFO; see
                    # mtile-ffn-phase1-ln-fix). [8,FC1_WROW] holds the fc1 [8,800] tile;
                    # fc2 reuses its first [8,M_SLAB]. Trade-off: depth-1 = no DMA/compute
                    # overlap (the M-tile LPDDR-reuse win is unaffected -- weights still
                    # stream once per tile). Recovering overlap = a Phase-2 aircc lever.
                    wtile = AllocOp(w1tile_l1, [], [])
                    wtile.attributes["air.shrinkage"] = BoolAttr.get(False)

                    # ---- 1. broadcast x[M_TILE,D] -> wbuf + b_fc2(rep [M_TILE,D]) ->
                    # bfc2_l1; ALL cores consume both (broadcast FIFO), only HEAD uses
                    # bfc2_l1. Both transfers are [M_TILE,D] (equal size on inX). ----
                    ChannelGet("inX", wbuf.result, indices=[tx, ty])
                    ChannelGet("inX", bfc2_l1.result, indices=[tx, ty])

                    # ---- 2. LayerNorm per row, IN-PLACE in wbuf (reads a full row
                    # before writing it, so in-place is safe). ----
                    CallOp(ln_func, [m_tile_i32, n_d_i32, eps_f32, wbuf.result, wbuf.result])

                    # ---- 3. fc1 GEMM (+ bias K-aug'd into the weight tile): each
                    # [8,FC1_WROW] tile reused across all M_TILE rows -> kernel writes
                    # [M_TILE,8] = dot(xnorm, w[:768]) + bias into h_l1[:,col_off].
                    # Reads wbuf (now xnorm). ----
                    for t in for_(0, FC1_TILES):
                        ChannelGet("inL2L1", wtile.result, indices=[tx])
                        if not STUB_GEMM:  # keep the get (drain inL2L1), skip compute
                            col_off = arith.IndexCastOp(T.i32(), arith.MulIOp(t, c8)).result
                            CallOp(fc1_func, [m_tile_i32, col_off, wtile.result, wbuf.result, h_l1.result])
                        yield_([])

                    # ---- 4. GELU once over the full [M_TILE,M_SLAB] slab (bias already
                    # folded into fc1). ----
                    if not STUB_GEMM:
                        CallOp(gelu_func, [gelu_n_i32, h_l1.result])

                    # ---- 5. fc2 GEMM: each [8,M_SLAB] weight tile reused across rows ->
                    # kernel writes [M_TILE,8] into wbuf (now the partial) at col_off.
                    # The w2 [8,M_SLAB] tile lands in the first M_SLAB cols of wtile
                    # (row stride FC1_WROW); the fc2 kernel reads it at WROW=FC1_WROW.
                    # wbuf's xnorm role is dead after fc1 -> safe reuse. ----
                    for t in for_(0, FC2_TILES):
                        ChannelGet("inL2L1", wtile.result, offsets=[0, 0],
                                   sizes=[M_INPUT, M_SLAB], strides=[FC1_WROW, 1], indices=[tx])
                        if not STUB_GEMM:  # keep the get (drain inL2L1), skip compute
                            col_off = arith.IndexCastOp(T.i32(), arith.MulIOp(t, c8)).result
                            CallOp(fc2_func, [m_tile_i32, col_off, wtile.result, h_l1.result, wbuf.result])
                        yield_([])
                    DeallocOp(h_l1)
                    DeallocOp(wtile)

                    # ---- 6. cascade K-reduction (+ b_fc2 head-inject), W->E along tx.
                    # Payload is the full [M_TILE,D] partial in wbuf (multi-beat). HEAD
                    # (tx==0): wbuf += b_fc2 (broadcast) -> put. MIDDLE (1..6): get prev
                    # -> rbuf, wbuf += rbuf in-place, put. TAIL (tx==7): get -> rbuf,
                    # wbuf += rbuf in-place, drain wbuf -> L2. ----
                    if STUB_CASCADE:
                        # latency-only: drop the 8-hop W->E serial reduce; TAIL writes
                        # its own wbuf (no inter-core get/put/ppr). Cores run parallel.
                        cmp_tail = arith.CmpIOp(arith.CmpIPredicate.eq, tx, last_ty)
                        if_t = scf.IfOp(cmp_tail, has_else=True)
                        with InsertionPoint(if_t.then_block):
                            dma_memcpy_nd(
                                _out_l2, wbuf.result,
                                dst_offsets=[0, 0], dst_sizes=[M_TILE, D], dst_strides=[D, 1],
                                src_offsets=[0, 0], src_sizes=[M_TILE, D], src_strides=[D, 1],
                            )
                            yield_([])
                        with InsertionPoint(if_t.else_block):
                            yield_([])
                        DeallocOp(wbuf)
                        DeallocOp(rbuf)
                        DeallocOp(bfc2_l1)
                        return
                    cmp_head = arith.CmpIOp(arith.CmpIPredicate.eq, tx, c0)
                    if_head = scf.IfOp(cmp_head, has_else=True)
                    with InsertionPoint(if_head.then_block):
                        # wbuf += b_fc2 (bfc2_l1 is [M_TILE,D] replicated -> elementwise ppr).
                        CallOp(ppr_func, [ppr_n_i32, wbuf.result, bfc2_l1.result, off0_i32, wbuf.result])
                        ChannelPut("chan_cascade", wbuf.result, indices=[tx])
                        yield_([])
                    with InsertionPoint(if_head.else_block):
                        cmp_tail = arith.CmpIOp(arith.CmpIPredicate.eq, tx, last_ty)
                        if_tail = scf.IfOp(cmp_tail, has_else=True)
                        with InsertionPoint(if_tail.then_block):
                            prev_t = arith.SubIOp(tx, c1)
                            ChannelGet("chan_cascade", rbuf.result, indices=[prev_t])
                            CallOp(ppr_func, [ppr_n_i32, wbuf.result, rbuf.result, off0_i32, wbuf.result])
                            dma_memcpy_nd(
                                _out_l2, wbuf.result,
                                dst_offsets=[0, 0], dst_sizes=[M_TILE, D], dst_strides=[D, 1],
                                src_offsets=[0, 0], src_sizes=[M_TILE, D], src_strides=[D, 1],
                            )
                            yield_([])
                        with InsertionPoint(if_tail.else_block):
                            prev_m = arith.SubIOp(tx, c1)
                            ChannelGet("chan_cascade", rbuf.result, indices=[prev_m])
                            CallOp(ppr_func, [ppr_n_i32, wbuf.result, rbuf.result, off0_i32, wbuf.result])
                            ChannelPut("chan_cascade", wbuf.result, indices=[tx])
                            yield_([])
                        yield_([])

                    DeallocOp(wbuf)
                    DeallocOp(rbuf)
                    DeallocOp(bfc2_l1)

                herd_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)

                # Drain the assembled out[M_TILE,D] L2 -> L3 (after the herd).
                dma_memcpy_nd(
                    out_s, out_l2.result,
                    dst_offsets=[0, 0], dst_sizes=[M_TILE, D], dst_strides=[D, 1],
                    src_offsets=[0, 0], src_sizes=[M_TILE, D], src_strides=[D, 1],
                )
                DeallocOp(out_l2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="ffn_cascade.py", description="Single-launch bf16 Whisper-FFN cascade")
    ap.add_argument("--d", type=int, default=768, dest="D")
    ap.add_argument("--ff", type=int, default=3072, dest="FF")
    ap.add_argument("--cores", type=int, default=8, dest="NCORES")
    ap.add_argument("--m-input", type=int, default=8, dest="M_INPUT")
    ap.add_argument("--m-tile", type=int, default=4, dest="M_TILE",
                    help="activation rows per dispatch (M-tile). All-L1 ceiling is small "
                         "(~4): two [M_TILE,D] act buffers + h + double-buffered weight "
                         "tiles (~36KB) must fit 64KB L1. Large M_TILE needs the "
                         "L2-resident intermediate (Phase 1).")
    ap.add_argument("--eps", type=float, default=1.0e-5)
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin", dest="output_format")
    ap.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-xclbin", "print"],
        default="compile-only",
        dest="compile_mode",
    )
    ap.add_argument("--out", default=".", dest="out")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    module = build_module(args.D, args.FF, args.NCORES, args.M_INPUT, args.eps, args.M_TILE)

    if args.print_module_only or args.compile_mode == "print":
        print(module)
        raise SystemExit(0)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "air.mlir"), "w") as f:
        f.write(str(module))

    # XRTBackend writes air.xclbin / air.insts.bin + air_project/ to CWD; run in out_dir.
    os.chdir(out_dir)
    # Backend recipe = the int4 SINGLE-TRIP cascade (o_gemv_ffn_int4_fused), NOT the
    # matvec_cascade_add MULTI-TRIP recipe the original copied. matvec re-arms its
    # cascade lock state via a 128-trip launch loop; our launch is sizes=[1,1] (one
    # trip), so use_lock_race_condition_fix=True + runtime_loop_tiling_sizes=[2,2]
    # (both tied to the shim-DMA-BD pass over the runtime loop) mis-configure a
    # non-existent loop -> 2nd dispatch "qds_device::wait() unexpected command state".
    # The only re-entrant SINGLE-TRIP npu_cascade reference uses the defaults +
    # stack_size=4096. See internal notes.
    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        use_lock_race_condition_fix=False,
        stack_size=4096,
    )
    backend.compile(module)
    backend.unload()
