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
def build_module(D, FF, NCORES, M_INPUT, EPS):
    M_SLAB = FF // NCORES            # 384: fc1 out rows/core == fc2 K-chunk/core
    assert FF % NCORES == 0 and M_SLAB % M_INPUT == 0 and D % M_INPUT == 0
    assert M_SLAB % 16 == 0 and D % 16 == 0 and D % 32 == 0  # bf16 legalize + cascade payload
    FC1_TILES = M_SLAB // M_INPUT   # 48
    FC2_TILES = D // M_INPUT        # 96
    N_CASCADE = NCORES

    bf16_ty = BF16Type.get()
    f32_ty = F32Type.get()
    l1_ms = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l2_ms = IntegerAttr.get(T.i32(), MemorySpace.L2)

    # L3 / DDR func args (host buffer layout -- see module docstring + report).
    # 5 host BOs (the aiecc kernel-ABI JSON hardcaps host BOs at 5): the two
    # static bias vectors are folded into ONE `biases` buffer of length FF+D:
    #   biases[0:FF]      = bias_fc1 = bf@Wfc1 + b_fc1   (per-core +bias before GELU)
    #   biases[FF:FF+D]   = b_fc2                         (cascade-HEAD residual)
    x_l3_ty = MemRefType.get([D], bf16_ty)
    Wfc1_l3_ty = MemRefType.get([FF, D], bf16_ty)        # (gf[:,None]*Wfc1).T
    biases_l3_ty = MemRefType.get([FF + D], bf16_ty)     # bias_fc1 ++ b_fc2  (3840)
    Wfc2_l3_ty = MemRefType.get([D, FF], bf16_ty)        # Wfc2.T (= mat_fc2)
    out_l3_ty = MemRefType.get([D], bf16_ty)

    # L1 buffer types.
    vecD_l1 = MemRefType.get([D], bf16_ty, memory_space=l1_ms)          # 768
    vecSlab_l1 = MemRefType.get([M_SLAB], bf16_ty, memory_space=l1_ms)  # 384
    w1tile_l1 = MemRefType.get([M_INPUT, D], bf16_ty, memory_space=l1_ms)       # [8,768]
    w2tile_l1 = MemRefType.get([M_INPUT, M_SLAB], bf16_ty, memory_space=l1_ms)  # [8,384]
    tile_l1 = MemRefType.get([M_INPUT], bf16_ty, memory_space=l1_ms)    # [8]
    acc_l1 = MemRefType.get([16], f32_ty, memory_space=l1_ms)
    # L2 staging buffers (one tile each, ping-ponged by the relay).
    w1tile_l2 = MemRefType.get([M_INPUT, D], bf16_ty, memory_space=l2_ms)
    w2tile_l2 = MemRefType.get([M_INPUT, M_SLAB], bf16_ty, memory_space=l2_ms)
    bias_l2_ty = MemRefType.get([M_SLAB], bf16_ty, memory_space=l2_ms)
    out_l2_ty = MemRefType.get([D], bf16_ty, memory_space=l2_ms)

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

    fc1_func = _kdecl("matvec_fc1_tile_bf16_store", [w1tile_l1, vecD_l1, tile_l1])
    fc2_func = _kdecl("matvec_fc2_tile_bf16_store", [w2tile_l1, vecSlab_l1, tile_l1])
    gelu_func = _kdecl("gelu_tile_bf16", [T.i32(), vecSlab_l1])
    # partial_plus_r_bf16(uint32 n, bf16* partial, bf16* r_full, int offset, bf16* d);
    # used only at length D=768 (cascade) -> one static shape, no dynamic memref.
    ppr_func = _kdecl(
        "partial_plus_r_bf16", [T.i32(), vecD_l1, vecD_l1, T.i32(), vecD_l1]
    )

    slab_map = _map_mul(M_SLAB)         # i -> i*384
    mtile_map = _map_mul(M_INPUT)       # t -> t*8
    scatter_maps = [_map_mul_plus(M_INPUT, i) for i in range(M_INPUT)]  # t -> t*8 + i

    # Host BO order (group_id = arg index + 3): bo0/x=gid3, bo1/Wfc1=gid4,
    # bo2/biases=gid5, bo3/Wfc2=gid6, bo4/out=gid7. (5 BOs, fits the aiecc cap.)
    @FuncOp.from_py_func(
        x_l3_ty, Wfc1_l3_ty, biases_l3_ty, Wfc2_l3_ty, out_l3_ty
    )
    def ffn_cascade(x_a, w1_a, biases_a, w2_a, out_a):
        @launch(sizes=[1, 1], operands=[x_a, w1_a, biases_a, w2_a, out_a])
        def launch_body(lx, ly, lsx, lsy, x_l3, w1_l3, biases_l3, w2_l3, out_l3):
            # L3-side reads (consolidated shim BDs). The inX broadcast carries TWO
            # same-shape [768] transfers to every core: x (for LN) then b_fc2 (the
            # cascade-HEAD residual). b_fc2 rides inX -- the proven-reliable channel
            # core0 already reads -- NOT the weight stream: multiplexing the odd
            # [768] b_fc2 onto core0's colL3L2/inL2L1 corrupted core0's weights
            # (only core0's partial was wrong). inX is already one of core0's 2 input
            # DMA channels, so this adds NO channel (wall #4 stays clear), and both
            # transfers are uniform [768] (no odd-shape fragility). The weight stream
            # is now uniform [w1x48, bias, w2x96] -- identical to the 7 clean cores.
            ChannelPut("inX", x_l3, offsets=[0], sizes=[D], strides=[1])
            ChannelPut("inX", biases_l3, offsets=[FF], sizes=[D], strides=[1])
            for c in range(NCORES):
                ci = arith.ConstantOp.create_index(c)
                # ONE multiplexed shim stream per col, in herd-consume order:
                # Wfc1 col slab rows [c*384:+384] of [FF,D] as 48 x [8,768] ...
                ChannelPut(
                    "colL3L2", w1_l3, indices=[ci],
                    offsets=[c * FC1_TILES, 0, 0],
                    sizes=[FC1_TILES, M_INPUT, D],
                    strides=[M_INPUT * D, D, 1],
                )
                # ... then the bias_fc1 slot biases[c*384:+384] (biases[0:FF]) ...
                ChannelPut(
                    "colL3L2", biases_l3, indices=[ci],
                    offsets=[c * M_SLAB], sizes=[M_SLAB], strides=[1],
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
                # order (w1 tiles, bias, w2 tiles) and fan out to the memtile
                # L2->L1 channels. FIFO order on colL3L2 == consume order -> no
                # cross-channel deadlock.
                for c in range(NCORES):
                    ci = arith.ConstantOp.create_index(c)
                    # Uniform weight stream for every col: w1 tiles, bias, w2 tiles.
                    # (b_fc2 no longer rides this channel -- it is on inX broadcast.)
                    for _ in for_(FC1_TILES):
                        t2 = AllocOp(w1tile_l2, [], [])
                        ChannelGet("colL3L2", t2.result, indices=[ci])
                        ChannelPut("inL2L1", t2.result, indices=[ci])
                        DeallocOp(t2)
                        yield_([])
                    b2 = AllocOp(bias_l2_ty, [], [])
                    ChannelGet("colL3L2", b2.result, indices=[ci])
                    ChannelPut("inL2L1", b2.result, indices=[ci])
                    DeallocOp(b2)
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
                    cD_idx = arith.ConstantOp.create_index(D)
                    cSlab_idx = arith.ConstantOp.create_index(M_SLAB)
                    c16_idx = arith.ConstantOp.create_index(16)
                    last_ty = arith.ConstantOp.create_index(N_CASCADE - 1)

                    n_slab_i32 = arith.constant(T.i32(), M_SLAB)
                    n_d_i32 = arith.constant(T.i32(), D)
                    off0_i32 = arith.constant(T.i32(), 0)

                    vbf = VectorType.get([16], bf16_ty)
                    vf32 = VectorType.get([16], f32_ty)
                    idmap = AffineMapAttr.get(AffineMap.get_identity(1))
                    cst0b = arith.ConstantOp(bf16_ty, 0.0)
                    cst0f = arith.ConstantOp(f32_ty, 0.0)

                    x_l1 = AllocOp(vecD_l1, [], [])
                    xnorm_l1 = AllocOp(vecD_l1, [], [])
                    h_l1 = AllocOp(vecSlab_l1, [], [])
                    bias_l1 = AllocOp(vecSlab_l1, [], [])
                    partial_op = AllocOp(vecD_l1, [], [])
                    partial_op.attributes["air.shrinkage"] = BoolAttr.get(False)
                    recv_op = AllocOp(vecD_l1, [], [])
                    recv_op.attributes["air.shrinkage"] = BoolAttr.get(False)
                    bfc2_l1 = AllocOp(vecD_l1, [], [])
                    out_l1 = AllocOp(vecD_l1, [], [])

                    # ---- 1. broadcast x + b_fc2 (two same-shape [768] gets on the
                    # inX broadcast; ALL cores consume both for broadcast FIFO
                    # consistency, but only the HEAD core uses bfc2_l1). x feeds LN;
                    # bfc2_l1 is held until the cascade HEAD injects it. ----
                    ChannelGet("inX", x_l1.result, indices=[tx, ty])
                    ChannelGet("inX", bfc2_l1.result, indices=[tx, ty])

                    acc_sum = AllocOp(acc_l1, [], [])
                    acc_sq = AllocOp(acc_l1, [], [])
                    zf = BroadcastOp(vf32, cst0f)
                    transfer_write(None, zf, acc_sum.result, [c0], idmap, [True])
                    transfer_write(None, zf, acc_sq.result, [c0], idmap, [True])
                    for j in for_(c0, cD_idx, c16_idx):
                        sub_x = subview(x_l1.result, [j], [16], [1])
                        vx = transfer_read(vbf, sub_x, [c0], idmap, cst0b, [True])
                        vxf = arith.extf(vf32, vx)
                        vs = transfer_read(vf32, acc_sum.result, [c0], idmap, cst0f, [True])
                        transfer_write(None, arith.addf(vs, vxf), acc_sum.result, [c0], idmap, [True])
                        # sum-of-squares: square in BF16 (f32 vector mul/fma are illegal
                        # on AIE2P -- mul_elem unsupported), accumulate in f32. (int4 RMS.)
                        vsq_b = arith.mulf(vx, vx)
                        vq = transfer_read(vf32, acc_sq.result, [c0], idmap, cst0f, [True])
                        transfer_write(None, arith.addf(vq, arith.extf(vf32, vsq_b)), acc_sq.result, [c0], idmap, [True])
                        yield_([])
                    v_sum_f = transfer_read(vf32, acc_sum.result, [c0], idmap, cst0f, [True])
                    total_sum = vector_reduction(f32_ty, "add", v_sum_f)
                    v_sq_f = transfer_read(vf32, acc_sq.result, [c0], idmap, cst0f, [True])
                    total_sq = vector_reduction(f32_ty, "add", v_sq_f)
                    cDf = arith.ConstantOp(f32_ty, float(D))
                    epsf = arith.ConstantOp(f32_ty, float(EPS))
                    mean = arith.divf(total_sum, cDf)
                    mean_of_sq = arith.divf(total_sq, cDf)
                    var = arith.subf(mean_of_sq, arith.mulf(mean, mean))
                    rstd = math_dialect.rsqrt(arith.addf(var, epsf))
                    # Normalize in BF16 (x-mean)*rstd: f32 vector mul is illegal on AIE2P,
                    # and bf16 elementwise mul is the supported path (int4 RMS normalize).
                    v_mean = BroadcastOp(vbf, arith.truncf(bf16_ty, mean))
                    v_rstd = BroadcastOp(vbf, arith.truncf(bf16_ty, rstd))
                    for j in for_(c0, cD_idx, c16_idx):
                        sub_x = subview(x_l1.result, [j], [16], [1])
                        vx = transfer_read(vbf, sub_x, [c0], idmap, cst0b, [True])
                        v_cen = arith.subf(vx, v_mean.result)
                        v_nrm = arith.mulf(v_cen, v_rstd.result)
                        sub_o = subview(xnorm_l1.result, [j], [16], [1])
                        transfer_write(None, v_nrm, sub_o, [c0], idmap, [True])
                        yield_([])
                    DeallocOp(acc_sum)
                    DeallocOp(acc_sq)
                    DeallocOp(x_l1)

                    # ---- 2. fc1 tiling: get [8,768] tiles, scatter into h_ty[384] ----
                    for t in for_(0, FC1_TILES):
                        w1t = AllocOp(w1tile_l1, [], [])
                        ChannelGet("inL2L1", w1t.result, indices=[tx])
                        ht = AllocOp(tile_l1, [], [])
                        CallOp(fc1_func, [w1t.result, xnorm_l1.result, ht.result])
                        for i in range(M_INPUT):
                            ci = arith.ConstantOp.create_index(i)
                            v = memref_load(ht.result, [ci])
                            dpos = affine_apply(scatter_maps[i], [t])
                            memref_store(v, h_l1.result, [dpos])
                        DeallocOp(w1t)
                        DeallocOp(ht)
                        yield_([])
                    DeallocOp(xnorm_l1)

                    # ---- 3. + bias_fc1 slab (inline, BEFORE GELU) ----
                    ChannelGet("inL2L1", bias_l1.result, indices=[tx])
                    for j in for_(c0, cSlab_idx, c16_idx):
                        sub_h = subview(h_l1.result, [j], [16], [1])
                        sub_b = subview(bias_l1.result, [j], [16], [1])
                        vh = transfer_read(vbf, sub_h, [c0], idmap, cst0b, [True])
                        vb = transfer_read(vbf, sub_b, [c0], idmap, cst0b, [True])
                        vsum = arith.addf(arith.extf(vf32, vh), arith.extf(vf32, vb))
                        transfer_write(None, arith.truncf(vbf, vsum), sub_h, [c0], idmap, [True])
                        yield_([])
                    DeallocOp(bias_l1)

                    # ---- 4. GELU(tanh) once over the full 384 slab ----
                    CallOp(gelu_func, [n_slab_i32, h_l1.result])

                    # ---- 5. fc2 tiling: get [8,384] tiles, scatter into partial[768] ----
                    for t in for_(0, FC2_TILES):
                        w2t = AllocOp(w2tile_l1, [], [])
                        ChannelGet("inL2L1", w2t.result, indices=[tx])
                        pt = AllocOp(tile_l1, [], [])
                        CallOp(fc2_func, [w2t.result, h_l1.result, pt.result])
                        for i in range(M_INPUT):
                            ci = arith.ConstantOp.create_index(i)
                            v = memref_load(pt.result, [ci])
                            dpos = affine_apply(scatter_maps[i], [t])
                            memref_store(v, partial_op.result, [dpos])
                        DeallocOp(w2t)
                        DeallocOp(pt)
                        yield_([])
                    DeallocOp(h_l1)

                    # ---- 6. cascade K-reduction (+ b_fc2 head-inject), W->E along tx ----
                    # HEAD (tx==0): acc = partial + b_fc2 -> put slot[0]. MIDDLE
                    # (tx 1..6): get slot[tx-1], add own partial, put slot[tx]. TAIL
                    # (tx==7): get slot[tx-1], add -> out_l1 -> L2. (int4 W->E idiom.)
                    cmp_head = arith.CmpIOp(arith.CmpIPredicate.eq, tx, c0)
                    if_head = scf.IfOp(cmp_head, has_else=True)
                    with InsertionPoint(if_head.then_block):
                        # b_fc2 already in bfc2_l1 (from the inX broadcast); inject it.
                        CallOp(ppr_func, [n_d_i32, partial_op.result, bfc2_l1.result, off0_i32, partial_op.result])
                        ChannelPut("chan_cascade", partial_op.result, indices=[tx])
                        yield_([])
                    with InsertionPoint(if_head.else_block):
                        cmp_tail = arith.CmpIOp(arith.CmpIPredicate.eq, tx, last_ty)
                        if_tail = scf.IfOp(cmp_tail, has_else=True)
                        with InsertionPoint(if_tail.then_block):
                            prev_t = arith.SubIOp(tx, c1)
                            ChannelGet("chan_cascade", recv_op.result, indices=[prev_t])
                            CallOp(ppr_func, [n_d_i32, partial_op.result, recv_op.result, off0_i32, out_l1.result])
                            dma_memcpy_nd(
                                _out_l2, out_l1.result,
                                dst_offsets=[0], dst_sizes=[D], dst_strides=[1],
                                src_offsets=[0], src_sizes=[D], src_strides=[1],
                            )
                            yield_([])
                        with InsertionPoint(if_tail.else_block):
                            prev_m = arith.SubIOp(tx, c1)
                            ChannelGet("chan_cascade", recv_op.result, indices=[prev_m])
                            CallOp(ppr_func, [n_d_i32, partial_op.result, recv_op.result, off0_i32, partial_op.result])
                            ChannelPut("chan_cascade", partial_op.result, indices=[tx])
                            yield_([])
                        yield_([])

                    DeallocOp(partial_op)
                    DeallocOp(recv_op)
                    DeallocOp(bfc2_l1)
                    DeallocOp(out_l1)

                herd_body.attributes["link_with"] = StringAttr.get(KERNEL_OBJ)

                # Drain the assembled out[768] L2 -> L3 (after the herd).
                dma_memcpy_nd(
                    out_s, out_l2.result,
                    dst_offsets=[0], dst_sizes=[D], dst_strides=[1],
                    src_offsets=[0], src_sizes=[D], src_strides=[1],
                )
                DeallocOp(out_l2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="ffn_cascade.py", description="Single-launch bf16 Whisper-FFN cascade")
    ap.add_argument("--d", type=int, default=768, dest="D")
    ap.add_argument("--ff", type=int, default=3072, dest="FF")
    ap.add_argument("--cores", type=int, default=8, dest="NCORES")
    ap.add_argument("--m-input", type=int, default=8, dest="M_INPUT")
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

    module = build_module(args.D, args.FF, args.NCORES, args.M_INPUT, args.eps)

    if args.print_module_only or args.compile_mode == "print":
        print(module)
        raise SystemExit(0)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "air.mlir"), "w") as f:
        f.write(str(module))

    # XRTBackend writes air.xclbin / air.insts.bin + air_project/ to CWD; run in out_dir.
    os.chdir(out_dir)
    backend = XRTBackend(
        verbose=args.verbose,
        omit_while_true_loop=False,
        runtime_loop_tiling_sizes=[2, 2],
        output_format=args.output_format,
        use_lock_race_condition_fix=True,
    )
    backend.compile(module)
    backend.unload()
