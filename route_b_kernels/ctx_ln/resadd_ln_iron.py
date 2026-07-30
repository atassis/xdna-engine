#
# FUSED residual-add -> LN+affine+cast in ONE xclbin (the first one-xclbin-per-block collapse).
#
#   out_bf16 = LN(a + scale*b) * gamma + beta        and        sum_f32 = a + scale*b
#
# WHY THIS PAIR. Every one of the fused block's four `residual_add_dev` calls is immediately followed
# by a device-in LN (ff1-residual -> satt LN; mhsa-residual -> conv LN; conv-residual -> ff2 LN;
# ff2-residual -> block-exit LN). 4 sites x 24 blocks = 96 command submissions per clip whose ONLY job
# is to hand a [PAD_M,KRES] f32 tensor to the next command through DDR.
#
# WHAT IT SAVES, and it is not mainly the command. Per fusion the intermediate sum no longer makes a
# DDR round-trip into the LN: that is 2 MB of read deleted per site, 192 MB/clip, against a measured
# ~6 GB/s effective rate. The command submission goes away too. See the tightened frontier invariant
# in AGENTS.md ("inside a block, the stream never leaves L2") -- keeping the intermediate ON-CHIP,
# not merely off the host, is the actual saving.
#
# TWO CORES PER COLUMN, joined by an on-chip ObjectFifo -- the dwconv_silu_iron.py pattern, whose own
# header records the separate-xclbin version costing ~1 ms/block. Two cores rather than one fused loop
# is also what keeps each core inside the AIE2 compute-tile 2-input-DMA budget:
#
#   host --a(f32)--> [resadd core] --sum(f32) ON-CHIP--> [ln core] --out(bf16)--> host
#   host --b(f32)--^        |                                ^
#                           +--sum(f32)--> host              +--gb(f32)-- host
#
#   resadd core: a + b            = 2 in, 1 out (broadcast)
#   ln core:     sum + gb         = 2 in, 1 out
#
# `sum` is broadcast to BOTH the shim and the LN core because three of the four sites reuse the
# residual downstream (x_bo feeds the next residual). The fourth (block-exit) does not, but building
# the general shape keeps one xclbin for all four.
#
# ABI: (a, b, gb, sum, out) -> group_ids g3,g4,g5,g6,g7 driven from Rust.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2, AnyComputeTile, Tile
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
from aie.iron.controlflow import range_

# Shim MM2S (host->device) DMA channels available across the whole array. AIE2 gives every ShimNOC
# tile 2 dest + 2 source DMA connections (AIETargetModel.cpp, AIE2TargetModel::getNumDest/
# SourceShimMuxConnections -> 2 for WireBundle::DMA) and NPU2TargetModel::columns() == 8, so the
# budget is 16 each way. This is SILICON, not a placer limit -- checked in the target model before
# working around it.
SHIM_MM2S_BUDGET = 16


def resadd_ln(dev, sequence_length, embedding_dim, scale, n_cores=8, spread=False):
    assert sequence_length % n_cores == 0, "rows must split evenly across the cores"
    assert embedding_dim % 16 == 0, "both kernels vectorize cols by 16"

    f32 = np.float32
    total = sequence_length * embedding_dim
    gb_len = 2 * embedding_dim
    rows_per_core = sequence_length // n_cores

    # SHIM INPUT BUDGET, and why `a` is staged through a MemTile at n_cores=8.
    #
    # Naively the design wants a(n) + b(n) + gb(1) shim MM2S channels. At n_cores=8 that is 17
    # against a budget of 16 and aiecc fails with "no ShimNOCTile has sufficient DMA capacity" --
    # which is what forced the first build down to 4 columns, i.e. HALF the parallelism, and made
    # the whole collapse a measured wash (1.76 ms/command against the 1.86 ms pair it replaces).
    #
    # The fix is to spend ONE shim channel on TWO cores: a core PAIR's `a` rows arrive on a single
    # shim channel as [2,embedding_dim] objects and a MemTile ObjectFifoLink splits them to the two
    # cores. Each shared pair frees exactly one channel, so `share_pairs` = the overshoot.
    #
    # NOT the alternative recorded in the worklog (gamma/beta on a host-written L1 RTP buffer). An
    # RTP is indeed an L1 address rather than a DMA channel, but aie/dialects/aie.py emits ONE
    # npu_write_rtp per INDEX and rejects slicing, so a [2*embedding_dim] gb costs 2048 write32 per
    # LN core -- ~200 KB of instruction stream per command submission on a brick whose entire point
    # is deleting bytes. mm_silu_epilogue.cc's RTP carries one i32 mode word; that precedent does
    # not extend to 8 KB. Staging through L2 costs no extra bytes at all, and moving fan-out into
    # the MemTile is the direction the tightened frontier invariant wants anyway.
    share_pairs = max(0, (2 * n_cores + 1) - SHIM_MM2S_BUDGET)
    assert 2 * share_pairs <= n_cores, (
        f"n_cores={n_cores} needs {share_pairs} shared pairs, more than the {n_cores // 2} available"
    )
    n_solo_a = n_cores - 2 * share_pairs      # cores whose `a` still has its own shim channel

    a_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]
    b_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]
    sum_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]     # resadd -> LN, ON-CHIP
    gb_chunk = np.ndarray[(gb_len,), np.dtype[f32]]
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

    a_pair_chunk = np.ndarray[(2 * embedding_dim,), np.dtype[f32]]

    # Cores [0, n_solo_a) keep a private shim channel; the trailing cores come in pairs off one
    # shared channel via a MemTile split (see the budget note above).
    of_a = [ObjectFifo(a_chunk, name=f"a_{i}") for i in range(n_solo_a)]
    of_a_pair = []
    for p in range(share_pairs):
        lo = n_solo_a + 2 * p
        pair = ObjectFifo(a_pair_chunk, name=f"a_pair_{p}", depth=2)
        of_a_pair.append(pair)
        # offsets [0, embedding_dim]: the first row of each [2,embedding_dim] object goes to the
        # even core of the pair, the second to the odd one. The fill tap below is what makes each
        # core's rows land CONTIGUOUS and in order, so every output tap stays untouched.
        of_a.extend(pair.cons().split(
            [0, embedding_dim],
            obj_types=[a_chunk, a_chunk],
            names=[f"a_{lo}", f"a_{lo + 1}"],
            depths=[2, 2],
        ))
    assert len(of_a) == n_cores

    of_b = [ObjectFifo(b_chunk, name=f"b_{i}") for i in range(n_cores)]
    # ONE producer, TWO consumers: the LN core (on-chip) and the shim (the reused residual).
    of_sum = [ObjectFifo(sum_chunk, name=f"sum_{i}", depth=2) for i in range(n_cores)]
    # ONE gb fifo BROADCAST to every LN core, not one per column. gamma/beta are identical for all
    # cores, and a per-column fill costs a shim input channel per column -- which is exactly what
    # blew the shim DMA budget on the first build (aiecc: "no ShimNOCTile has sufficient DMA
    # capacity"). Broadcast spends ONE shim input for all 8.
    of_gb = ObjectFifo(gb_chunk, name="gb")
    of_out = [ObjectFifo(out_chunk, name=f"out_{i}") for i in range(n_cores)]

    add_kern = Kernel("residual_add_row", "residual_add.o",
                      [a_chunk, b_chunk, sum_chunk, np.float32, np.int32])
    ln_kern = Kernel("ln_affine_cast_row", "ln_affine_cast.o",
                     [sum_chunk, gb_chunk, out_chunk, np.int32])

    taps = TensorTiler2D.simple_tiler((sequence_length, embedding_dim), (rows_per_core, embedding_dim))

    def add_body(of_a, of_b, of_sum, add):
        for _ in range_(rows_per_core):
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            es = of_sum.acquire(1)
            add(ea, eb, es, scale, embedding_dim)   # scale baked, as in residual_add_iron.py
            of_a.release(1)
            of_b.release(1)
            of_sum.release(1)

    def ln_body(of_sum, of_gb, of_out, ln):
        egb = of_gb.acquire(1)      # [gamma|beta] acquired ONCE, reused across this core's rows
        for _ in range_(rows_per_core):
            es = of_sum.acquire(1)
            eo = of_out.acquire(1)
            ln(es, egb, eo, embedding_dim)
            of_sum.release(1)
            of_out.release(1)
        of_gb.release(1)

    # COLUMN SPREAD (opt-in, --spread). The sequential placer is column-MAJOR, so it fills columns
    # 0..3 rows 2..5 with all 16 workers and leaves compute columns 4..7 empty while the shim DMAs
    # for those columns still route across the array. Pinning each resadd/LN pair to its own column
    # puts every pair next to the shim channels that feed and drain it. Column only -- the row is
    # left to the placer, as in lpddr_bw_microbench.py's _pinned().
    #
    # Kept as a FLAG rather than the default because it is a separate variable from the shim-budget
    # fix, and the note that made this session's A/Bs readable is that one change at a time is what
    # keeps a result attributable.
    def _tile(i):
        return Tile(col=i % 8) if spread else AnyComputeTile

    workers = []
    for i in range(n_cores):
        workers.append(Worker(add_body, tile=_tile(i),
                              fn_args=[of_a[i].cons(), of_b[i].cons(), of_sum[i].prod(), add_kern]))
        workers.append(Worker(ln_body, tile=_tile(i),
                              fn_args=[of_sum[i].cons(), of_gb.cons(), of_out[i].prod(), ln_kern]))

    a_ty = np.ndarray[(total,), np.dtype[f32]]
    b_ty = np.ndarray[(total,), np.dtype[f32]]
    gb_ty = np.ndarray[(gb_len,), np.dtype[f32]]
    sum_ty = np.ndarray[(total,), np.dtype[f32]]
    out_ty = np.ndarray[(total,), np.dtype[bfloat16]]
    def sequence(a, b, gb, s_out, out, a_prods, a_pair_prods, b_prods, gb_prod, sum_conses, out_conses):
        for i in range(n_solo_a):
            a_prods[i].fill(a, taps[i])
        for p in range(share_pairs):
            lo = n_solo_a + 2 * p
            # Interleave the two cores' row ranges so each [2,embedding_dim] object is
            # (row lo_base+i, row hi_base+i). After the split each core sees its own 64 rows in
            # natural order -- which is why `sum`/`out` keep the plain simple_tiler taps.
            #
            # The row is emitted as `inner_n` runs of `inner` elements rather than one run of
            # embedding_dim: a shim BD's D0_Wrap field is 10 bits (AIEXDialect.cpp wrap_bits = 10,
            # XAIEMLGBL_NOC_MODULE_DMA_BD0_3_D0_WRAP_WIDTH), so a non-collapsible innermost
            # dimension is capped at 1023 elements and embedding_dim=1024 overflows it by one. The
            # stock per-core taps never hit this because they are fully contiguous and collapse to a
            # linear transfer. Factoring costs nothing -- the bytes and their order are identical.
            inner, inner_n = embedding_dim, 1
            while inner > 1023:
                assert inner % 2 == 0, f"cannot factor embedding_dim={embedding_dim} under the 1023 D0 wrap"
                inner //= 2
                inner_n *= 2
            a_pair_prods[p].fill(a, TensorAccessPattern(
                (sequence_length, embedding_dim),
                offset=lo * rows_per_core * embedding_dim,
                sizes=[rows_per_core, 2, inner_n, inner],
                strides=[embedding_dim, rows_per_core * embedding_dim, inner, 1],
            ))
        for i in range(n_cores):
            b_prods[i].fill(b, taps[i])
        gb_prod.fill(gb)                      # ONE broadcast fill for all LN cores
        for i in range(n_cores):
            sum_conses[i].drain(s_out, taps[i], wait=True)
            out_conses[i].drain(out, taps[i], wait=True)

    rt = Runtime(
        sequence,
        [
            a_ty,
            b_ty,
            gb_ty,
            sum_ty,
            out_ty,
            [of_a[i].prod() for i in range(n_solo_a)],
            [of_a_pair[p].prod() for p in range(share_pairs)],
            [of_b[i].prod() for i in range(n_cores)],
            of_gb.prod(),
            [of_sum[i].cons() for i in range(n_cores)],
            [of_out[i].cons() for i in range(n_cores)],
        ],
    )
    return Program(dev, rt, workers=workers).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-s", "--scale", required=True, dest="scale")
p.add_argument("-n", "--cores", required=False, dest="cores", default=8)
p.add_argument("--spread", action="store_true", dest="spread",
               help="pin each resadd/LN pair to its own column (0..7) instead of letting the "
                    "column-major placer pack all 16 workers into columns 0..3")
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(resadd_ln(dev, int(opts.rows), int(opts.cols), float(opts.scale), int(opts.cores), opts.spread))
