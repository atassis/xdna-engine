#
# mha_decode — on-chip SINGLE-QUERY (M=1) multi-head attention for the Whisper
# decoder (M1 Task 0). Standalone parity design; validated vs the host reference
# `attend_one` by rust/npu-asr/src/bin/mha_decode_probe.rs.
#
# whisper-small: D=768, n_heads=12, head_dim hd=64.
#
# STREAMING / FLASH design (forced by L1 capacity — see mha_decode.cc header):
# the full per-head K+V does not fit one core's ~64 KB, so K/V are streamed in
# tiles of TKV keys and the softmax is ONLINE (flash). ONE compute core, ONE
# worker; per head it sweeps ceil(S/TKV) K/V tiles accumulating online state,
# then emits ctx_h.
#
# DMA-channel budget (a compute tile has 2 input + 2 output channels):
#   q  fifo  : 12 tiles of [HD] bf16  (one per head)        -> 1 input  DMA
#   kv fifo  : 12*n_tiles tiles of [TKV*HD | TKV*HD] bf16    -> 1 input  DMA
#   ctx fifo : 12 tiles of [HD] f32                          -> 1 output DMA
#  => 2 in + 1 out. OK.
#
# HOST BUFFER LAYOUT (what the probe packs / the IRON tilers slice):
#   q   : [12, HD]                              bf16  (head-major)   -> ABI slot 3 (A)
#   kv  : [12, n_tiles, 2*TKV*HD + 2]           bf16  (per head, per tile:
#         K-tile (TKV*HD) | V-tile (TKV*HD) | int32 s_in_tile (2 bf16 lanes))
#                                                                    -> ABI slot 4 (B)
#   ctx : [12, HD]                              f32                  -> ABI slot 5 (C), read back
#  (3 runtime args -> run_matmul8 slots A=g3, B=g4, C=g5; ctx read from the C/g5 slot.)
#
# RUNTIME S (one xclbin for all S<=448). The self-KV cache grows per token, so S varies
# 1..448. Zero-pad-and-round-up POISONS softmax (a zero K row scores 0, not -inf, so it
# steals weight). Fix: build ONE xclbin with a FIXED tile count n_tiles=ceil(S_MAX/TKV)=7
# (S_MAX=448), and pass the real per-tile key count at RUNTIME. The host writes, into the
# 4 bytes after each tile's V-tile, an int32 s_in_tile: >0 normal, <0 last non-empty
# (finalize), 0 empty (skipped). The kernel reads that (bit-exact via 2 bf16 lanes) and
# ignores the baked int arg. seq still names the xclbin (mha_decode_${seq}); build seq=448.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

D = 768
NHEADS = 12
HD = 64
TKV = 64  # keys per K/V tile; MUST equal -DMHA_TKV in the kernel build.
S_MAX = 448  # whisper-small max decode positions; fixes the unrolled tile count.


def ceildiv(a, b):
    return (a + b - 1) // b


def mha_decode(dev, S, trace_size):
    assert HD % 16 == 0
    from ml_dtypes import bfloat16 as _bf16

    # FIXED tile count for the one resident xclbin (independent of the runtime S).
    n_tiles = ceildiv(S_MAX, TKV)
    KV_TILE = 2 * TKV * HD + 2  # K-tile | V-tile | 2-bf16 (int32) runtime-S header

    # element types
    q_tile_ty = np.ndarray[(HD,), np.dtype[_bf16]]
    kv_tile_ty = np.ndarray[(KV_TILE,), np.dtype[_bf16]]  # K | V | hdr
    ctx_tile_ty = np.ndarray[(HD,), np.dtype[np.float32]]

    # full host buffers
    q_all_ty = np.ndarray[(NHEADS * HD,), np.dtype[_bf16]]
    kv_all_ty = np.ndarray[(NHEADS * n_tiles * KV_TILE,), np.dtype[_bf16]]
    ctx_all_ty = np.ndarray[(NHEADS * HD,), np.dtype[np.float32]]

    mha_kernel = Kernel(
        "mha_tile",
        "mha_decode.o",
        [q_tile_ty, kv_tile_ty, ctx_tile_ty, np.int32, np.int32],
    )

    of_q = ObjectFifo(q_tile_ty, name="q_in", depth=2)
    of_kv = ObjectFifo(kv_tile_ty, name="kv_in", depth=2)
    of_ctx = ObjectFifo(ctx_tile_ty, name="ctx_out", depth=2)

    def core_body(q_cons, kv_cons, ctx_prod, kern):
        # 12 heads; inner tile loop unrolled over the FIXED n_tiles. The baked int arg is
        # a placeholder (0); the real per-tile count is read at runtime from the kv header.
        for _h in range(NHEADS):
            eq = q_cons.acquire(1)
            ec = ctx_prod.acquire(1)
            for t in range(n_tiles):
                ekv = kv_cons.acquire(1)
                kern(eq, ekv, ec, t, 0)
                kv_cons.release(1)
            q_cons.release(1)
            ctx_prod.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_q.cons(), of_kv.cons(), of_ctx.prod(), mha_kernel],
    )

    # ONE DMA per fifo over the whole flat buffer (separate per-tile fills blow the
    # shim BD pool). The contiguous host layout is already in tile order, so the
    # consumer's repeated acquire(1) of one element walks it tile by tile:
    #   q  : 12 * [HD]          (head order)
    #   kv : 12 * n_tiles * [2*TKV*HD]  (head-major, tile order: K-tile|V-tile)
    #   ctx: 12 * [HD]          (head order)
    rt = Runtime()
    with rt.sequence(q_all_ty, kv_all_ty, ctx_all_ty) as (q, kv, ctx):
        rt.start(worker)
        rt.fill(of_q.prod(), q)
        rt.fill(of_kv.prod(), kv)
        rt.drain(of_ctx.cons(), ctx, wait=True)
    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-s", "--seq", required=True, dest="seq", type=int)
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(mha_decode(dev, int(opts.seq), int(opts.trace_size)))
