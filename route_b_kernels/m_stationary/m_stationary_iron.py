#
# M-STATIONARY (row/token-stationary) whole-array GEMM — the probe for internal notes CLUE 1.
#
# This INVERTS the parallel axis of whole_array_iron.py. The shipped design is N-STATIONARY:
# the 8 columns split the OUTPUT N across themselves (B distributed by column), so each core only
# ever sees n (e.g. 96) of the N feature columns of a given output row — a per-row (feature-dim)
# reduction (LayerNorm / softmax) is therefore NOT intra-core and cannot fuse as an epilogue
# (internal notes, 09: the "N-stationary reduction-axis" wall).
#
# Here the 8 columns split M (rows/tokens) instead, and B is BROADCAST (each column streams the
# FULL N of B). Consequence: each compute core owns a contiguous m-tile of rows and sweeps the
# FULL N, so a per-row reduction is INTRA-CORE — LN/softmax become fusable epilogues (the lever).
#
# LAYOUT (the simplification that makes this tractable):
#   column `col` owns the contiguous M-band rows [col*M_band : (col+1)*M_band],
#   M_band = M // n_aie_cols, split across the 4 rows of that column into 4 m-tiles.
#   => each column is an INDEPENDENT single-column M-stationary unit on its own M-band, all
#      reading the same full B. (Naive B replication: each column re-reads B from DDR. A MemTile
#      read-once fan-out is a later optimization, gated on the measured bandwidth — see docs/10.)
#
# This is the N-stationary dataflow with the M and N roles (and A and B roles) SWAPPED:
#   - A[col] is a fixed per-column tap (its M-band, all K), reused across all N   (was B[col]).
#   - B walks the N-blocks, broadcast to all columns                              (was A).
#   - C[col] = the column's M-band × the current N-block.
#
# Plain GEMM first (no epilogue) to de-risk placement/correctness; the fused per-row reduction
# epilogue (the gate's question 2) is a separate build on top of this dataflow.
import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, str_to_dtype
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D

microkernel_mac_dim_map = {
    "npu2": {
        "bf16": {True: (8, 8, 8), False: (4, 8, 8)},
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
    emulate_bf16_mmul_with_bfp16,
    dtype_in_str,
    dtype_out_str,
    trace_size,
    generate_taps=False,
):
    assert dev == "npu2"
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    mac_dims = microkernel_mac_dim_map[dev][dtype_in_str]
    if dtype_in_str == "bf16":
        r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]
    else:
        r, s, t = mac_dims

    M_band = M // n_aie_cols
    # M-STATIONARY invariants: every core owns exactly one (m, K) row-tile and sweeps the full N.
    assert M % (m * n_aie_rows * n_aie_cols) == 0, "M must split into n_aie_cores m-row-tiles"
    assert M_band == m * n_aie_rows
    assert K % k == 0
    assert N % n == 0
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    N_div_n = N // n
    K_div_k = K // k
    # Each core produces N_div_n output tiles (the full N for its m-rows).
    n_tiles_per_core = N_div_n

    fifo_depth = 2
    import os as _os
    c_fifo_depth = int(_os.environ.get("WA_C_DEPTH", str(fifo_depth)))

    dev_ty = NPU2()

    A_taps = []
    B_taps = []
    C_taps = []

    # Tensor types
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    # one column's A reader holds the 4 row m-tiles for one k-block, split to the 4 rows
    A_l2_ty = np.ndarray[(m * k * n_aie_rows,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    # Kernels
    zero_kernel = Kernel(f"zero_{dtype_out_str}", f"mm_{m}x{k}x{n}.o", [C_l1_ty])
    matmul_kernel = Kernel(
        f"matmul_{dtype_in_str}_{dtype_out_str}",
        f"mm_{m}x{k}x{n}.o",
        [A_l1_ty, B_l1_ty, C_l1_ty],
    )

    # Fifos
    A_l3l2_fifos = [None] * n_aie_cols
    A_l2l1_fifos = [[None] * n_aie_rows for _ in range(n_aie_cols)]
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_rows for _ in range(n_aie_cols)]
    C_l2l3_fifos = [None] * n_aie_cols

    for col in range(n_aie_cols):
        # --- A: column col's M-band, distributed across its 4 rows ---
        A_l3l2_fifos[col] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=fifo_depth)
        of_offsets = [m * k * row for row in range(n_aie_rows)]
        a_dims = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]] * n_aie_rows
        a_tmp = (
            A_l3l2_fifos[col]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * n_aie_rows,
                names=[f"A_L2L1_{col}_{row}" for row in range(n_aie_rows)],
                dims_to_stream=a_dims,
            )
        )
        for row in range(n_aie_rows):
            A_l2l1_fifos[col][row] = a_tmp[row]

        # --- B: FULL B, broadcast to the 4 rows of this column ---
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        if b_col_maj:
            b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims)
        )

        # --- C: join the 4 row m-tiles of this column ---
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=c_fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        of_offsets = [m * n * row for row in range(n_aie_rows)]
        c_tmp = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[c_fifo_depth] * n_aie_rows,
            )
        )
        for row in range(n_aie_rows):
            C_l1l2_fifos[col][row] = c_tmp[row]

    # Generic output-stationary core: sweep the full N (n_tiles_per_core output tiles), each a
    # K-reduction into the [m,n] C tile. Identical to whole_array_iron's core_fn — the M-stationary
    # behaviour comes entirely from the dataflow (which A/B/C each core sees), not the core program.
    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1)  # issue #1547
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_div_k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = []
    for col in range(n_aie_cols):
        for row in range(n_aie_rows):
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[col][row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[col][row].prod(),
                        zero_kernel,
                        matmul_kernel,
                    ],
                    stack_size=0xD00,
                )
            )

    # A single DMA descriptor dimension is limited to a wrap count of 64. The full N has N_div_n
    # n-blocks (96 at the FFN-mm1 shape) > 64, so we cannot stream all of N (nor reuse A across all
    # of N via pattern_repeat) in one BD. Chunk N into groups of CH n-blocks (CH <= 64, dividing
    # N_div_n) and loop the runtime over the few chunks (2 at 96) — keeps the insts compact.
    CH = N_div_n
    while CH > 64 or N_div_n % CH != 0:
        CH -= 1
    n_chunks = N_div_n // CH

    # A (tall tile = whole_array_iron "distribute" idiom): one L2 buffer carries the 4 row m-tiles of
    # ONE k-block contiguously (row-major); the split of_offsets=[m*k*row] cut them into 4 rows.
    # pattern_repeat=CH reuses the band across a chunk's n-blocks. Same tap every chunk.
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_aie_rows, k), (1, K_div_k),
        pattern_repeat=CH, prune_step=False,
    )
    # B: per-chunk tap = B[:, chunk] tiled (k,n) col-major (all K of an n-block before the next).
    if b_col_maj:
        B_tiles = TensorTiler2D.group_tiler(
            (N, K), (n, k), (CH, K_div_k), prune_step=False
        )
    else:
        B_tiles = TensorTiler2D.group_tiler(
            (K, N), (k, n), (K_div_k, CH), tile_group_col_major=True, prune_step=False
        )
    # C: per (column, chunk) tap = column's M-band × the chunk's n-blocks. group index = col*n_chunks+c.
    C_tiles = TensorTiler2D.group_tiler(
        (M, N), (m * n_aie_rows, n), (1, CH), prune_step=False
    )

    # --- Runtime: per chunk, fill all columns' A bands + this chunk of B (kick off all 32 cores),
    # then drain all columns' C. Each column is an independent M-stationary unit. ---
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)
        for c in range(n_chunks):
            tg = rt.task_group()
            for col in range(n_aie_cols):
                rt.fill(A_l3l2_fifos[col].prod(), A, tap=A_tiles[col], task_group=tg)
                rt.fill(B_l3l2_fifos[col].prod(), B, tap=B_tiles[c], task_group=tg)
                A_taps.append(A_tiles[col])
                B_taps.append(B_tiles[c])
            for col in range(n_aie_cols):
                rt.drain(
                    C_l2l3_fifos[col].cons(), C,
                    tap=C_tiles[col * n_chunks + c], wait=True, task_group=tg,
                )
                C_taps.append(C_tiles[col * n_chunks + c])
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
    p = argparse.ArgumentParser(prog="M-stationary whole-array GEMM probe")
    p.add_argument("--dev", type=str, choices=["npu2"], default="npu2")
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=768)
    p.add_argument("-N", type=int, default=3072)
    p.add_argument("-m", type=int, default=16)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    p.add_argument("--dtype_in", type=str, default="bf16", choices=["bf16", "i8", "i16"])
    p.add_argument("--dtype_out", type=str, default="f32", choices=["f32", "i32", "bf16", "i16", "i8"])
    p.add_argument("--trace_size", type=int, default=0)
    p.add_argument("--generate-taps", action="store_true")
    args = p.parse_args()
    maybe = my_matmul(
        args.dev, args.M, args.K, args.N, args.m, args.k, args.n,
        args.n_aie_cols, args.b_col_maj, args.emulate_bf16_mmul_with_bfp16,
        args.dtype_in, args.dtype_out, args.trace_size, args.generate_taps,
    )
    if args.generate_taps:
        return maybe
    print(maybe)


if __name__ == "__main__":
    main()
