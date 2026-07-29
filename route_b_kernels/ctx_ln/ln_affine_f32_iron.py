#
# Fused ctxLN + affine dataflow, f32 OUT (one kernel / one hw-context). The f32-out twin of
# ln_affine_cast_iron.py -- the BLOCK-EXIT LN, whose output the next block consumes directly, so it must
# stay f32: out[t,D] = LN(x[t,D]) * gamma[D] + beta[D], all in one dispatch. Same 8-core
# per-row structure + (x, gb, out) DMA topology as affine_cast_iron.py; only the kernel differs (it does
# the two-pass LN reduction before the affine+cast). gamma/beta packed [gamma|beta] on ONE channel and
# broadcast to all cores (x + gb = 2 input DMAs, the compute-tile limit).
#
# Runtime sequence (x, gb, out) -> group_ids g3,g4,g5; driven from Rust by
#   run_matmul8(3, instr, n, bo_x, bo_gb, bo_out, dummy_tmp, dummy_tr).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def ln_affine_f32(dev, sequence_length, embedding_dim, trace_size, n_cores=8):
    assert sequence_length % n_cores == 0, "rows must split evenly across the cores"
    assert embedding_dim % 8 == 0, "ln_affine_f32_row<8> vectorizes cols by 8 (256-bit f32 stores)"

    f32 = np.float32
    total = sequence_length * embedding_dim
    gb_len = 2 * embedding_dim  # [gamma | beta] packed on ONE DMA input channel

    rows_per_core = sequence_length // n_cores
    in_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]
    gb_chunk = np.ndarray[(gb_len,), np.dtype[f32]]
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]   # f32 OUT (block-boundary dtype)

    of_in = [ObjectFifo(in_chunk, name=f"in_{i}") for i in range(n_cores)]
    of_gb = [ObjectFifo(gb_chunk, name=f"gb_{i}") for i in range(n_cores)]
    of_out = [ObjectFifo(out_chunk, name=f"out_{i}") for i in range(n_cores)]

    kern = Kernel(
        "ln_affine_f32_row", "ln_affine_f32.o",
        [in_chunk, gb_chunk, out_chunk, np.int32],
    )

    # One tile per core on both paths, exactly like the bf16 sibling. The tile-splitting that used to
    # sit here was a workaround for a "128 KiB transfer cap" that does not exist -- see out_ty below.
    taps_in = TensorTiler2D.simple_tiler((sequence_length, embedding_dim), (rows_per_core, embedding_dim))
    taps_out = TensorTiler2D.simple_tiler((sequence_length, embedding_dim), (rows_per_core, embedding_dim))

    def core_body(of_in, of_gb, of_out, fn):
        egb = of_gb.acquire(1)  # [gamma|beta] acquired ONCE, reused across this core's rows
        for _ in range_(rows_per_core):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            fn(ei, egb, eo, embedding_dim)
            of_in.release(1)
            of_out.release(1)
        of_gb.release(1)

    workers = [
        Worker(core_body, fn_args=[of_in[i].cons(), of_gb[i].cons(), of_out[i].prod(), kern])
        for i in range(n_cores)
    ]

    rt = Runtime()
    x_ty = np.ndarray[(total,), np.dtype[f32]]
    gb_ty = np.ndarray[(gb_len,), np.dtype[f32]]
    # f32 OUT. This was `bfloat16` -- copied verbatim from the bf16 sibling and never changed when the
    # kernel's store was widened. The runtime-sequence tensor's dtype is what the shim drain BDs are
    # sized and strided in, so a 2-byte element size moved exactly HALF the bytes at HALF the offsets:
    # core 0 wrote f32 rows 0-31 bit-exact and core 1's (zero-padded, hence exactly `beta`) rows landed
    # on top of f32 rows 32-63. That is the whole "128 KiB cap" -- 64*1024*2 = 131072 is not a limit,
    # it is the wrong element size. See the retraction in the worklog.
    out_ty = np.ndarray[(total,), np.dtype[f32]]
    with rt.sequence(x_ty, gb_ty, out_ty) as (x, gb, out):
        rt.start(*workers)
        for i in range(n_cores):
            rt.fill(of_in[i].prod(), x, taps_in[i])
            rt.fill(of_gb[i].prod(), gb)  # full [gamma|beta] to every core (broadcast)
        for i in range(n_cores):
            rt.drain(of_out[i].cons(), out, taps_out[i], wait=True)
    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
p.add_argument("-n", "--cores", required=False, dest="cores", default=8)
opts = p.parse_args(sys.argv[1:])

dev = NPU2() if opts.device == "npu2" else NPU1()
print(ln_affine_f32(dev, int(opts.rows), int(opts.cols), int(opts.trace_size), int(opts.cores)))
