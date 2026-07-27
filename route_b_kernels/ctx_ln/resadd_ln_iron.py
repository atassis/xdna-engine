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
# in AGENTS.md ("inside a block, the stream never leaves L2") and
# [[l2-resident-stream-is-the-real-prize]].
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
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def resadd_ln(dev, sequence_length, embedding_dim, scale, n_cores=8):
    assert sequence_length % n_cores == 0, "rows must split evenly across the cores"
    assert embedding_dim % 16 == 0, "both kernels vectorize cols by 16"

    f32 = np.float32
    total = sequence_length * embedding_dim
    gb_len = 2 * embedding_dim
    rows_per_core = sequence_length // n_cores

    a_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]
    b_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]
    sum_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]     # resadd -> LN, ON-CHIP
    gb_chunk = np.ndarray[(gb_len,), np.dtype[f32]]
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]

    of_a = [ObjectFifo(a_chunk, name=f"a_{i}") for i in range(n_cores)]
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

    workers = []
    for i in range(n_cores):
        workers.append(Worker(add_body,
                              fn_args=[of_a[i].cons(), of_b[i].cons(), of_sum[i].prod(), add_kern]))
        workers.append(Worker(ln_body,
                              fn_args=[of_sum[i].cons(), of_gb.cons(), of_out[i].prod(), ln_kern]))

    rt = Runtime()
    a_ty = np.ndarray[(total,), np.dtype[f32]]
    b_ty = np.ndarray[(total,), np.dtype[f32]]
    gb_ty = np.ndarray[(gb_len,), np.dtype[f32]]
    sum_ty = np.ndarray[(total,), np.dtype[f32]]
    out_ty = np.ndarray[(total,), np.dtype[bfloat16]]
    with rt.sequence(a_ty, b_ty, gb_ty, sum_ty, out_ty) as (a, b, gb, s_out, out):
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_a[i].prod(), a, taps[i])
            rt.fill(of_b[i].prod(), b, taps[i])
        rt.fill(of_gb.prod(), gb)             # ONE broadcast fill for all LN cores
        for i in range(n_cores):
            rt.drain(of_sum[i].cons(), s_out, taps[i], wait=True)
            rt.drain(of_out[i].cons(), out, taps[i], wait=True)
    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-s", "--scale", required=True, dest="scale")
p.add_argument("-n", "--cores", required=False, dest="cores", default=8)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(resadd_ln(dev, int(opts.rows), int(opts.cols), float(opts.scale), int(opts.cores)))
