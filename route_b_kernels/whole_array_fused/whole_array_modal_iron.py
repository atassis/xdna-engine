#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
#
# WHOLE-ARRAY (8-column) matmul with a FUSED epilogue. ONE xclbin computes
#     out = silu(A @ B + bias)          (silu mode,  bf16 out)
#     out =      A @ B + bias           (bias mode,  bf16 out, no activation)
# across all n_aie_cols columns (4 rows x 8 cols = 32 compute cores on NPU2),
# bf16 inputs, f32 in-core accumulate, bf16 output written to DDR.
#
# This merges two already-proven pieces in this repo:
#   1. whole_array_iron.py  -- the 8-column matmul that builds N=3072 in one shot
#      (n*n_aie_cols=256, 3072/256=12 row-blocks, well under the 64-BD limit, so
#       NO N-splitting is needed unlike the single-core design).
#   2. mm_silu_fused_iron.py (single core) -- the f32 acc_buf + mm_silu_epilogue +
#      f32->bf16 output ObjectFifo pattern, with BIAS folded by K-augmentation.
#
# WHAT CHANGED vs whole_array_iron.py:
#   * The C ObjectFifo chain (C_l1l2 / C_l2l3) carries the OUTPUT dtype (bf16),
#     not the f32 accumulator. The shim de-shuffle dims (c_dims) are pure
#     stride tuples and are dtype-independent, so they are unchanged.
#   * Each of the n_aie_rows*n_aie_cols workers gets its OWN core-local f32
#     accumulator Buffer (cacc_ty). The matmul reduces into that f32 buffer over
#     K; after the K reduction the epilogue kernel reads the f32 buffer, applies
#     silu (or a plain narrow in bias mode) and writes the bf16 C tile out.
#   * core_fn acquires the bf16 out tile, zeroes the f32 acc, runs the matmul
#     loop into the acc, then epilogue(acc, out_tile), then releases.
#
# BIAS via K-AUGMENTATION (host side, no third DMA input). The NPU2 compute tile
# has only 2 input DMA channels (A and B), so a separate bias stream does not
# fit. The host appends ONE extra k-block:
#     A_aug = [A | ones_col]   (extra k-block: col 0 = 1, rest 0)
#     B_aug = [B ; bias_row]   (extra k-block: row 0 = bias, rest 0)
#   A_aug @ B_aug = A@B + ones@bias = A@B + bias   (bias added to every row)
# So the device just runs a plain matmul with K = K_aug = K + k, then the
# epilogue. The runner script (run_npu_mm_silu_wa.py) builds A_aug/B_aug.
import argparse
import numpy as np

from aie.iron import (
    Buffer,
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    WorkerRuntimeBarrier,
    str_to_dtype,
)
from aie.iron.device import NPU1Col1, NPU1Col2, NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
        "i8": (4, 8, 8),
        "i16": (4, 4, 4),
    },
    "npu2": {
        "bf16": {
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
        "i8": (8, 8, 8),
        "i16": (4, 4, 8),
    },
}


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    b_col_maj,
    do_silu,
    emulate_bf16_mmul_with_bfp16,
    trace_size,
    generate_taps=False,
):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    # MODAL design: bf16 in / f32 accumulate / f32 OUT (no host re-expand; see internal notes).
    dtype_in_str = "bf16"
    dtype_acc_str = "f32"
    dtype_out_str = "f32"

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_acc = str_to_dtype(dtype_acc_str)
    dtype_out = str_to_dtype(dtype_out_str)

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map[dev]["bf16"]
    if dev == "npu2":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    if dev == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""
    assert K % k == 0
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0
    assert (m * n) % 16 == 0, "epilogue walks the tile in 16-wide chunks"

    fifo_depth = 2
    # Single-buffer the C path on wide-N fast tiles (WA_C_DEPTH=1): the f32 acc_buf + a double-buffered
    # C overflows L1 at 64x32x96. Matches whole_array_iron.py's WA_C_DEPTH knob.
    import os as _os

    c_fifo_depth = int(_os.environ.get("WA_C_DEPTH", str(fifo_depth)))

    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    if dev == "npu":
        if n_aie_cols == 1:
            dev_ty = NPU1Col1()
        elif n_aie_cols == 2:
            dev_ty = NPU1Col2()
        elif n_aie_cols == 4:
            dev_ty = NPU1()
    else:
        dev_ty = NPU2()

    A_taps = []
    B_taps = []
    C_taps = []

    # Define tensor types. NOTE: C / C_l2 / C_l1 are now OUTPUT dtype (bf16).
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
    # Core-local f32 accumulator tile (one per worker).
    cacc_ty = np.ndarray[(m, n), np.dtype[dtype_acc]]

    # AIE Core Function declarations. The zero+matmul kernels operate on the f32
    # accumulator (cacc_ty); the epilogue reads f32 acc, writes bf16 out tile.
    zero_kernel = Kernel(f"zero_{dtype_acc_str}", f"mm_{m}x{k}x{n}.o", [cacc_ty])
    matmul_kernel = Kernel(
        f"matmul_{dtype_in_str}_{dtype_acc_str}",
        f"mm_{m}x{k}x{n}.o",
        [A_l1_ty, B_l1_ty, cacc_ty],
    )
    # MODAL f32-out epilogue: rtp[0] selects silu (1) vs identity (0) at runtime, so ONE xclbin
    # serves both via the inst-stream-baked rtp value (the do_silu build flag sets that value below).
    rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
    epilogue_kernel = Kernel(
        "mm_modal_epilogue_f32_f32",
        f"mm_silu_epilogue_{m}x{k}x{n}.o",
        [cacc_ty, C_l1_ty, rtp_ty],
    )

    # Tile declarations as tile[row][col]
    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    # AIE-array data movement with object fifos
    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows

    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols

    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # Input A
    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims_to_stream = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
            )
        )

        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    # Input B
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        if b_col_maj:
            dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=dims_to_stream,
            )
        )

        # Output C (bf16). The shim de-shuffle dims are dtype-independent stride
        # tuples (same as the plain matmul) -- they reorder the mmul-blocked tile
        # to row-major on the way to DDR.
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=c_fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]

        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[c_fifo_depth] * n_aie_rows,
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    # Tasks for each worker to perform. Because the modal output is f32 (same dtype as the f32
    # accumulator), the matmul reduces DIRECTLY into the C output tile and the epilogue runs IN-PLACE
    # on it (silu/identity, f32->f32) — no separate acc_buf. That saves an m*n*4 L1 buffer per core,
    # which is what lets the wide-N fast tile (64x32x96) fit in L1.
    def core_fn(in_a, in_b, out_c, rtp_buff, barrier, zero, matmul, epilogue):
        barrier.wait_for_value(1)  # wait for the host to write the epilogue mode into rtp
        loop = range(1)  # Workaround for issue #1547
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)

            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            epilogue(elem_out, elem_out, rtp_buff)  # in-place; rtp[0]: 1=silu, 0=identity
            out_c.release(1)

    # Per-core RTP (epilogue mode) + a shared runtime barrier. Each worker reads rtp[0].
    rtp_barrier = WorkerRuntimeBarrier()
    rtp_bufs = [
        [
            Buffer(rtp_ty, name=f"rtp_{row}_{col}", use_write_rtp=True)
            for col in range(n_aie_cols)
        ]
        for row in range(n_aie_rows)
    ]

    # Set up compute tiles. In-place modal epilogue -> no per-core acc_buf (matmul reduces into C).
    workers = []
    for row in range(n_aie_rows):
        for col in range(n_aie_cols):
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[row][col].prod(),
                        rtp_bufs[row][col],
                        rtp_barrier,
                        zero_kernel,
                        matmul_kernel,
                        epilogue_kernel,
                    ],
                    stack_size=0xD00,
                )
            )

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    if b_col_maj:
        B_tiles = TensorTiler2D.step_tiler(
            (N, K),
            (n, k),
            tile_group_repeats=(N // n // n_aie_cols, K // k),
            tile_group_steps=(n_aie_cols, 1),
            prune_step=False,
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),
            (k, n),
            tile_group_repeats=(K // k, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            tile_group_col_major=True,
            prune_step=False,
        )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N),
        (m * n_aie_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        prune_step=False,
    )
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        # bake the epilogue mode into this instruction stream's RTP (1=silu, 0=identity), then
        # release the barrier so the cores read it. A silu-built stream and an identity-built stream
        # are two .txt insts on the SAME xclbin -> the host picks mode by choosing the stream.
        mode_val = 1 if do_silu else 0
        flat_rtps = [rtp_bufs[r][c] for r in range(n_aie_rows) for c in range(n_aie_cols)]

        def set_modes(*ps):
            for p in ps:
                p[0] = mode_val

        rt.inline_ops(set_modes, flat_rtps)
        rt.set_barrier(rtp_barrier, 1)
        rt.start(*workers)

        tg = rt.task_group()
        for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break

                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )

                for col in range(n_aie_cols):
                    C_taps.append(C_tiles[c_index])
                    rt.drain(
                        C_l2l3_fifos[col].cons(),
                        C,
                        tap=C_tiles[c_index],
                        wait=True,
                        task_group=tg,
                    )
                    c_index += 1

                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)

                        if col < n_aie_rows:
                            rt.fill(
                                A_l3l2_fifos[col].prod(),
                                A,
                                tap=A_tiles[tile_offset],
                                task_group=tg,
                            )

                        rt.fill(
                            B_l3l2_fifos[col].prod(),
                            B,
                            tap=B_tiles[col],
                            task_group=tg,
                        )

                        A_taps.append(A_tiles[tile_offset])
                        B_taps.append(B_tiles[col])
                if tb > 0 or (tb == 0 and pingpong > 0):
                    rt.finish_task_group(tg)
                    tg = rt.task_group()
        rt.finish_task_group(tg)

    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )

    my_program = Program(dev_ty, rt)
    module = my_program.resolve_program()
    return module


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE whole-array fused matmul + bias (+ optional SiLU)",
        description="silu(A@B+bias) or A@B+bias across all columns, bf16 in/out, "
        "f32 accumulate, single xclbin",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu2")
    argparser.add_argument("-M", type=int, default=512)
    argparser.add_argument("-K", type=int, default=768)
    argparser.add_argument("-N", type=int, default=3072)
    argparser.add_argument("-m", type=int, default=32)
    argparser.add_argument("-k", type=int, default=32)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=8)
    argparser.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    argparser.add_argument(
        "--no-silu",
        action="store_true",
        help="bias mode: out = A@B+bias (plain f32->bf16 narrow, no activation). "
        "Default is silu mode: out = silu(A@B+bias).",
    )
    argparser.add_argument(
        "--emulate-bf16-mmul-with-bfp16", type=bool, default=False
    )
    # Accepted for makefile-common compatibility; this design is fixed to
    # bf16 in / f32 accumulate / bf16 out.
    argparser.add_argument("--dtype_in", type=str, default="bf16", choices=["bf16"])
    argparser.add_argument("--dtype_out", type=str, default="f32", choices=["f32"])
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument("--generate-taps", action="store_true")
    args = argparser.parse_args()
    maybe_module = my_matmul(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.n_aie_cols,
        args.b_col_maj,
        not args.no_silu,
        args.emulate_bf16_mmul_with_bfp16,
        args.trace_size,
        args.generate_taps,
    )
    if args.generate_taps:
        return maybe_module
    else:
        print(maybe_module)


if __name__ == "__main__":
    main()
