#
# Device-side f32 -> bf16 cast dataflow (resident-rails seam primitive).
#
# Mirrors ctx_ln_iron.py's 8-core per-row structure EXACTLY (rows split across 8
# cores, each core processes its rows over `cols`), but the kernel is the
# elementwise cast (cast_f32_bf16.cc) and the OUTPUT ObjectFifo is bf16 (half the
# bytes). Input f32, output bf16. Used to bridge an f32 producer (ctxLN) to the
# bf16-in whole_array matmul with the activation staying device-side.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import argparse
import sys

from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2, from_name
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_


def cast_ffn(dev, sequence_length, embedding_dim, trace_size, l2_groups=0):
    n_cores = 8
    assert sequence_length % n_cores == 0, "rows must split evenly across 8 cores"
    assert embedding_dim % 16 == 0, "cast_f32_bf16_row<16> vectorizes cols by 16"
    # l2_groups=G stages the DDR<->core traffic through G MemTiles instead of giving every
    # core its own DDR-facing fifo. Baseline is one shim channel PER CORE per direction (16
    # for 8 cores), which spreads the design over 4 shim columns though its compute fits in
    # 2; with G groups it is 2*G. Cores in a group take strided rows instead of a contiguous
    # block -- fine here because the kernel is elementwise and the join applies the SAME
    # permutation to the output. G=0 (default) keeps the shipped per-core path untouched.
    assert l2_groups == 0 or n_cores % l2_groups == 0, "cores must split evenly into L2 groups"

    f32 = np.float32
    total = sequence_length * embedding_dim
    in_dtype = np.ndarray[(total,), np.dtype[f32]]
    out_dtype = np.ndarray[(total,), np.dtype[bfloat16]]

    rows_per_core = sequence_length // n_cores
    in_chunk = np.ndarray[(embedding_dim,), np.dtype[f32]]       # one f32 row of D
    out_chunk = np.ndarray[(embedding_dim,), np.dtype[bfloat16]]  # one bf16 row of D

    if l2_groups:
        grp = n_cores // l2_groups                       # cores fed by one MemTile
        rows_per_group = sequence_length // l2_groups
        in_l2_ty = np.ndarray[(grp * embedding_dim,), np.dtype[f32]]
        out_l2_ty = np.ndarray[(grp * embedding_dim,), np.dtype[bfloat16]]
        of_in_l2 = [ObjectFifo(in_l2_ty, name=f"inl2{g}", depth=2) for g in range(l2_groups)]
        of_out_l2 = [ObjectFifo(out_l2_ty, name=f"outl2{g}", depth=2) for g in range(l2_groups)]
        offs = [r * embedding_dim for r in range(grp)]
        in_split = [
            of_in_l2[g].cons().split(
                offs, obj_types=[in_chunk] * grp, depths=[2] * grp,
                names=[f"in{g}_{r}" for r in range(grp)],
            )
            for g in range(l2_groups)
        ]
        out_join = [
            of_out_l2[g].prod().join(
                offs, obj_types=[out_chunk] * grp, depths=[2] * grp,
                names=[f"out{g}_{r}" for r in range(grp)],
            )
            for g in range(l2_groups)
        ]
        # split()/join() return ObjectFifos, not handles -- take the handle at the worker.
        core_in = [in_split[i // grp][i % grp].cons() for i in range(n_cores)]
        core_out = [out_join[i // grp][i % grp].prod() for i in range(n_cores)]
    else:
        of_in = [ObjectFifo(in_chunk, name=f"in_{i}") for i in range(n_cores)]
        of_out = [ObjectFifo(out_chunk, name=f"out_{i}") for i in range(n_cores)]
        core_in = [of_in[i].cons() for i in range(n_cores)]
        core_out = [of_out[i].prod() for i in range(n_cores)]

    cast_kernel = Kernel(
        "cast_f32_bf16_row", "cast_f32_bf16.o", [in_chunk, out_chunk, np.int32]
    )

    taps_in = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )
    taps_out = TensorTiler2D.simple_tiler(
        (sequence_length, embedding_dim), (rows_per_core, embedding_dim)
    )

    def core_body(of_in, of_out, cast):
        for _ in range_(rows_per_core):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            cast(elem_in, elem_out, embedding_dim)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, fn_args=[core_in[i], core_out[i], cast_kernel])
        for i in range(n_cores)
    ]

    if l2_groups:
        # One contiguous DDR slice per MemTile; the split hands round r's block r to core r.
        g_taps_in = TensorTiler2D.simple_tiler(
            (sequence_length, embedding_dim), (rows_per_group, embedding_dim)
        )
        g_taps_out = TensorTiler2D.simple_tiler(
            (sequence_length, embedding_dim), (rows_per_group, embedding_dim)
        )
        n_streams = l2_groups
        taps_in, taps_out = g_taps_in, g_taps_out
        rt_in = [of_in_l2[g].prod() for g in range(l2_groups)]
        rt_out = [of_out_l2[g].cons() for g in range(l2_groups)]
    else:
        n_streams = n_cores
        rt_in = [of_in[i].prod() for i in range(n_cores)]
        rt_out = [of_out[i].cons() for i in range(n_cores)]

    def sequence(a_in, c_out, in_prods, out_conses):
        for i in range(n_streams):
            in_prods[i].fill(a_in, taps_in[i])
        for i in range(n_streams):
            out_conses[i].drain(c_out, taps_out[i], wait=True)

    rt = Runtime(sequence, [in_dtype, out_dtype, rt_in, rt_out])
    return Program(dev, rt, workers=workers).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-r", "--rows", required=True, dest="rows")
p.add_argument("-c", "--cols", required=True, dest="cols")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0)
p.add_argument("-p", "--part-cols", required=False, type=int, default=None, dest="part_cols")
p.add_argument("--l2-groups", required=False, type=int, default=0, dest="l2_groups",
               help="stage DDR<->core traffic through N MemTiles (0 = per-core shim fifos)")
opts = p.parse_args(sys.argv[1:])

dev = (from_name(opts.device, n_cols=opts.part_cols) if opts.part_cols
       else (NPU2() if opts.device == "npu2" else NPU1()))
print(cast_ffn(dev, int(opts.rows), int(opts.cols), int(opts.trace_size), opts.l2_groups))
