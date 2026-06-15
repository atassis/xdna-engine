#
# Fused Norm+GEMV decode primitive — IRON design (whole_array M=64, f32 out).
#
# Computes the decode projection  out[64, N] = A[64, K] @ W''[K, N]  on the array (bf16 in,
# f32 in-core accumulate, f32 out), where W'' is the FOLDED weight (W'' = diag(gamma)*W) the
# host precomputes in CtxDecode::register_fused / norm_gemv_probe. This is the heavy half of
# the fused norm+GEMV primitive:
#
#     RMS:  out = inv_rms*(x @ W'')               (+ bias, host-added)
#     LN :  out = inv_std*((x - mean) @ W'') + bias'
#
# The SEPARABILITY of the fold (proven to machine-eps by `norm_gemv_probe selftest`) means the
# norm reduces to (a) two scalars over the input row x (mean/inv_std or inv_rms) and (b) a
# uniform input-scale  x_norm = inv_*( x [- mean] ). For DECODE (M=1) those scalars are a single
# K-vector reduction — free on the host, NOT a dispatch — so CtxDecode::fused_norm_gemv does the
# input-scale on host and dispatches THIS xclbin for x_norm @ W'', then host-adds bias'. One
# dispatch; the same dispatch count as a plain GEMV. The xclbin is f32-out so the host consumer
# needs nothing back-converted.
#
# --- On-device prologue (norm_prologue.cc): DEFERRED, with a concrete structural blocker. ---
# The companion `aie_kernels/norm_gemv_prologue.cc` normalizes a FULL-K-resident [m,K] A tile in
# place before the GEMV. That requires the core to hold all of K of row 0 at once. The whole_array
# dataflow this design is built on STREAMS A as mmul-blocked [m,k=32] sub-tiles (A dims_to_stream
# shuffles each [m,k] block into the layout matmul_bf16_f32 expects); the microkernel's pointer
# math is baked to that [m,k] blocked layout, so a contiguous full-[m,K] resident A cannot be
# sub-tiled into it without a NEW matmul microkernel (or a prologue core that re-shuffles [m,K] ->
# [m,k] blocks). That is real kernel authoring + on-device validation, not a drop-in of the
# existing decode_gemv matmul. Since the M=1 decode norm is free on host, the on-device prologue
# buys nothing here and is left as a future optimization (it pays off only for M>1 / when the norm
# scalars must be computed array-side). See internal notes.
#
# So this file emits the SAME f32-out GEMV MLIR as the plain decode_gemv (build_decode_kernels.sh);
# the norm kind is carried in the xclbin filename (_ln / _rms) for ABI / CtxDecode-loader symmetry.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, str_to_dtype
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessSequence, TensorTiler2D

# npu2 native bf16 (no bfp16 emul): r,s,t = 4,8,8
R, S, T = 4, 8, 8


def ceildiv(a, b):
    return (a + b - 1) // b


def my_norm_gemv(M, K, N, m, k, n, n_aie_cols, generate_taps=False):
    """f32-out whole_array GEMV (the heavy half of fused norm+GEMV). `norm` is carried in the
    filename only — the separable fold runs host-side (see header)."""
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype("bf16")
    dtype_out = str_to_dtype("f32")
    r, s, t = R, S, T

    assert M % (m * n_aie_rows) == 0, "A must tile into (m*n_aie_rows, k) blocks"
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0, "B must tile into (k, n*n_aie_cols) blocks"
    assert m % r == 0 and k % s == 0 and n % t == 0

    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1
    dev_ty = NPU2()

    A_taps, B_taps, C_taps = [], [], []

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    zero_kernel = Kernel("zero_f32", f"mm_{m}x{k}x{n}.o", [C_l1_ty])
    matmul_kernel = Kernel(
        "matmul_bf16_f32", f"mm_{m}x{k}x{n}.o", [A_l1_ty, B_l1_ty, C_l1_ty]
    )

    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims_to_stream = [
            [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
        ] * (stop_row - start_row)
        a_tmp = A_l3l2_fifos[i].cons().split(
            of_offsets,
            obj_types=[A_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=dims_to_stream,
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp[j]

    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = B_l3l2_fifos[col].cons().forward(
            obj_type=B_l1_ty, name=f"B_L2L1_{col}", dims_to_stream=b_dims
        )
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]
        c_tmp = C_l2l3_fifos[col].prod().join(
            of_offsets, obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp[j]

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1)
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                ea = in_a.acquire(1)
                eb = in_b.acquire(1)
                matmul(ea, eb, elem_out)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

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
                        zero_kernel,
                        matmul_kernel,
                    ],
                    stack_size=0xD00,
                )
            )

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m * n_A_tiles_per_shim, k), (1, K // k),
        pattern_repeat=N // n // n_aie_cols, prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N), (k, n),
        tile_group_repeats=(K // k, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        tile_group_col_major=True, prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N), (m * n_aie_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols), prune_step=False,
    )
    c_index = 0

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)
        tg = rt.task_group()
        for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break
                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min([tb_max_n_rows // 2, M // m // n_aie_rows - row_base])
                for col in range(n_aie_cols):
                    C_taps.append(C_tiles[c_index])
                    rt.drain(C_l2l3_fifos[col].cons(), C, tap=C_tiles[c_index], wait=True, task_group=tg)
                    c_index += 1
                    for tile_row in range(current_tb_n_rows):
                        tile_offset = ((row_base + tile_row) * n_shim_mem_A + col) % len(A_tiles)
                        if col < n_aie_rows:
                            rt.fill(A_l3l2_fifos[col].prod(), A, tap=A_tiles[tile_offset], task_group=tg)
                        rt.fill(B_l3l2_fifos[col].prod(), B, tap=B_tiles[col], task_group=tg)
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
    return Program(dev_ty, rt).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Fused Norm+GEMV decode primitive (whole_array M=64, f32 out)")
    p.add_argument("--dev", type=str, choices=["npu2"], default="npu2")
    p.add_argument("-M", type=int, default=64)
    p.add_argument("-K", type=int, default=768)
    p.add_argument("-N", type=int, default=768)
    p.add_argument("-m", type=int, default=8)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=8)
    p.add_argument("--norm", type=str, choices=["none", "rms", "ln"], default="ln",
                   help="folded host-side; carried in the xclbin filename for ABI symmetry")
    # makefile-common compatibility (this design is fixed bf16-in / f32-out)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    p.add_argument("--dtype_in", type=str, default="bf16", choices=["bf16"])
    p.add_argument("--dtype_out", type=str, default="f32", choices=["f32"])
    p.add_argument("--trace_size", type=int, default=0)
    p.add_argument("--generate-taps", action="store_true")
    args = p.parse_args()
    maybe = my_norm_gemv(args.M, args.K, args.N, args.m, args.k, args.n,
                         args.n_aie_cols, args.generate_taps)
    if args.generate_taps:
        return maybe
    print(maybe)


if __name__ == "__main__":
    main()
