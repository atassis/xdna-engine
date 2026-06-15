#
# M-STATIONARY GEMM with a FUSED LayerNorm epilogue (Phase 1.2 spike).
#
# Builds on m_stationary_iron.py (the M-stationary dataflow: 8 columns split M, B broadcast, so each
# core owns the FULL N of its m output rows) + the fused-epilogue idiom from whole_array_silu_iron.py
# (a per-core f32 accumulator Buffer; matmul reduces into it over K; then epilogue(acc, out) writes bf16).
#
# The epilogue here is a per-row TWO-PASS LayerNorm over the full row (mm_ln_epilogue.cc) — the row
# reduction that ONLY M-stationary can fuse intra-core (N-stationary splits N across columns, no core
# owns a full row). NORMALIZE-ONLY for now (gamma=1/beta=0); affine is a follow-on.
#
# SCOPE: this first version handles the SINGLE-BLOCK case n == N (N_div_n == 1) — each core's full
# row is one contiguous [m, N] matmul tile, so the resident f32 acc is contiguous and the LN epilogue
# reads it directly. The multi-block case (N > n, e.g. the 512x768x768 primary shape) needs a blocked
# [N_div_n, m, n] acc + a gather in the epilogue — flagged NotImplementedError below; build it after the
# single-block mechanism is proven on device (narrow-N is also where M-stationary already wins the GEMM).
import argparse
import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker, str_to_dtype
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D

microkernel_mac_dim_map = {
    "npu2": {"bf16": {True: (8, 8, 8), False: (4, 8, 8)}},
}


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    dev, M, K, N, m, k, n, n_aie_cols, b_col_maj,
    emulate_bf16_mmul_with_bfp16, dtype_in_str, trace_size, generate_taps=False,
):
    assert dev == "npu2"
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_acc = str_to_dtype("f32")   # resident GEMM accumulator (the LN reduces over this, in f32)
    dtype_out = str_to_dtype("bf16")  # fused output is bf16

    r, s, t = microkernel_mac_dim_map[dev][dtype_in_str][emulate_bf16_mmul_with_bfp16]

    M_band = M // n_aie_cols
    assert M % (m * n_aie_rows * n_aie_cols) == 0, "M must split into n_aie_cores m-row-tiles"
    assert M_band == m * n_aie_rows
    assert K % k == 0
    assert N % n == 0
    assert m % r == 0 and k % s == 0 and n % t == 0
    assert (m * n) % 16 == 0, "epilogue walks the row in 16-wide chunks"

    N_div_n = N // n
    K_div_k = K // k
    if N_div_n != 1:
        raise NotImplementedError(
            f"multi-block N (N_div_n={N_div_n}) needs a blocked [N_div_n,m,n] acc + gather epilogue; "
            "this build handles only the single-block case n==N (see header)."
        )
    n_tiles_per_core = N_div_n  # == 1

    fifo_depth = 2
    dev_ty = NPU2()
    A_taps, B_taps, C_taps = [], [], []

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_aie_rows,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]   # bf16 output tile
    cacc_ty = np.ndarray[(m, n), np.dtype[dtype_acc]]   # resident f32 accumulator (== full row since n==N)

    # Kernels: zero+matmul reduce into the f32 acc; the LN epilogue reads f32 acc, writes bf16 out.
    zero_kernel = Kernel("zero_f32", f"mm_{m}x{k}x{n}.o", [cacc_ty])
    matmul_kernel = Kernel("matmul_bf16_f32", f"mm_{m}x{k}x{n}.o", [A_l1_ty, B_l1_ty, cacc_ty])
    ln_kernel = Kernel("mm_ln_epilogue_f32_bf16", f"mm_ln_epilogue_{m}x{k}x{n}.o", [cacc_ty, C_l1_ty])

    A_l3l2_fifos = [None] * n_aie_cols
    A_l2l1_fifos = [[None] * n_aie_rows for _ in range(n_aie_cols)]
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_rows for _ in range(n_aie_cols)]
    C_l2l3_fifos = [None] * n_aie_cols

    for col in range(n_aie_cols):
        A_l3l2_fifos[col] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=fifo_depth)
        of_offsets = [m * k * row for row in range(n_aie_rows)]
        a_dims = [[(m // r, r * k), (k // s, s), (r, k), (s, 1)]] * n_aie_rows
        a_tmp = A_l3l2_fifos[col].cons().split(
            of_offsets, obj_types=[A_l1_ty] * n_aie_rows,
            names=[f"A_L2L1_{col}_{row}" for row in range(n_aie_rows)], dims_to_stream=a_dims,
        )
        for row in range(n_aie_rows):
            A_l2l1_fifos[col][row] = a_tmp[row]

        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        if b_col_maj:
            b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
            obj_type=B_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims,
        )

        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        of_offsets = [m * n * row for row in range(n_aie_rows)]
        c_tmp = C_l2l3_fifos[col].prod().join(
            of_offsets, obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)], depths=[fifo_depth] * n_aie_rows,
        )
        for row in range(n_aie_rows):
            C_l1l2_fifos[col][row] = c_tmp[row]

    # Fused core: zero the f32 acc, K-reduce the matmul into it, then LN(acc) -> bf16 out tile.
    def core_fn(in_a, in_b, out_c, acc, zero, matmul, layernorm):
        loop = range(1)
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(acc)
            for _ in range_(K_div_k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, acc)
                in_a.release(1)
                in_b.release(1)
            layernorm(acc, elem_out)
            out_c.release(1)

    workers = []
    for col in range(n_aie_cols):
        for row in range(n_aie_rows):
            acc_buf = Buffer(cacc_ty, name=f"acc_buf_{row}_{col}")
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[col][row].cons(),
                        B_l2l1_fifos[col].cons(),
                        C_l1l2_fifos[col][row].prod(),
                        acc_buf, zero_kernel, matmul_kernel, ln_kernel,
                    ],
                    stack_size=0xD00,
                )
            )

    CH = N_div_n
    while CH > 64 or N_div_n % CH != 0:
        CH -= 1
    n_chunks = N_div_n // CH

    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_aie_rows, k), (1, K_div_k), pattern_repeat=CH, prune_step=False,
    )
    if b_col_maj:
        B_tiles = TensorTiler2D.group_tiler((N, K), (n, k), (CH, K_div_k), prune_step=False)
    else:
        B_tiles = TensorTiler2D.group_tiler((K, N), (k, n), (K_div_k, CH), tile_group_col_major=True, prune_step=False)
    C_tiles = TensorTiler2D.group_tiler((M, N), (m * n_aie_rows, n), (1, CH), prune_step=False)

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)
        for c in range(n_chunks):
            tg = rt.task_group()
            for col in range(n_aie_cols):
                rt.fill(A_l3l2_fifos[col].prod(), A, tap=A_tiles[col], task_group=tg)
                rt.fill(B_l3l2_fifos[col].prod(), B, tap=B_tiles[c], task_group=tg)
                A_taps.append(A_tiles[col]); B_taps.append(B_tiles[c])
            for col in range(n_aie_cols):
                rt.drain(C_l2l3_fifos[col].cons(), C, tap=C_tiles[col * n_chunks + c], wait=True, task_group=tg)
                C_taps.append(C_tiles[col * n_chunks + c])
            rt.finish_task_group(tg)

    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return Program(dev_ty, rt).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="M-stationary GEMM + fused LayerNorm epilogue")
    p.add_argument("--dev", type=str, choices=["npu2"], default="npu2")
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=768)
    p.add_argument("-N", type=int, default=64)
    p.add_argument("-m", type=int, default=16)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-n", type=int, default=64)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    p.add_argument("--dtype_in", type=str, default="bf16", choices=["bf16"])
    # output is always bf16 for the fused path; accept --dtype_out from makefile-common and ignore it.
    p.add_argument("--dtype_out", type=str, default="bf16")
    p.add_argument("--trace_size", type=int, default=0)
    p.add_argument("--generate-taps", action="store_true")
    args = p.parse_args()
    maybe = my_matmul(
        args.dev, args.M, args.K, args.N, args.m, args.k, args.n, args.n_aie_cols,
        args.b_col_maj, args.emulate_bf16_mmul_with_bfp16, args.dtype_in, args.trace_size, args.generate_taps,
    )
    if args.generate_taps:
        return maybe
    print(maybe)


if __name__ == "__main__":
    main()
