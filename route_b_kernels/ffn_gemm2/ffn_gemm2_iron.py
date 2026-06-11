#
# Fused two-matmul chain  C = (A @ W1) @ W2   (bf16 in, f32 out)
# with the intermediate  H = A @ W1  materialized ON-CHIP (MemTile/L2)
# and consumed by the second matmul in a SINGLE host dispatch.
#
# Derived from single_core_iron.py (one-matmul template).  Two compute
# cores form a pipeline:
#   core1 = mm1:  A[M,K] @ W1[K,P] -> H[M,P]   (accumulate over K)
#   core2 = mm2:  H[M,P] @ W2[P,N] -> C[M,N]   (accumulate over P)
# H never leaves the device: it flows core1 -> MemTile -> core2 through an
# ObjectFifo, with the layout relaid out in the MemTile DMA.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, str_to_dtype
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


# r, s, t microkernel MAC dims for AIE2P bf16 (no bfp16 emulation).
RST_BF16_NPU2 = (4, 8, 8)


def ceildiv(a, b):
    return (a + b - 1) // b


def my_ffn(
    dev,
    M,
    K,
    P,
    N,
    m,
    k,
    p,
    n,
    dtype_in_str,
    dtype_out_str,
):
    # ----- shapes / tiling -----
    assert M % m == 0
    assert K % k == 0
    assert P % p == 0
    assert N % n == 0
    # H is mm1's output (m x p) and mm2's A operand (m x p): one tile shape.
    assert dtype_in_str == "bf16" and dtype_out_str == "f32", "prototype is bf16->f32"

    r, s, t = RST_BF16_NPU2
    # mm1 produces H in r x t micro-tiles; mm2 consumes H as A in r x s
    # micro-tiles.  We relayout H to plain row-major in the MemTile in
    # between, so each core only ever sees its own native tiling.
    for nm, dim, lo in [("m", m, r), ("k", k, s), ("p", p, max(s, t)), ("n", n, t)]:
        assert dim % lo == 0, f"{nm}={dim} must be divisible by {lo}"

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    M_div_m = M // m
    K_div_k = K // k
    P_div_p = P // p
    N_div_n = N // n

    # ----- tensor types (flat host buffers + tile types) -----
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    W1_ty = np.ndarray[(K * P,), np.dtype[dtype_in]]
    W2_ty = np.ndarray[(P * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]

    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]      # mm1 A tile
    w1_ty = np.ndarray[(k, p), np.dtype[dtype_in]]     # mm1 B tile
    h_in_ty = np.ndarray[(m, p), np.dtype[dtype_in]]   # H tile, mm2 A operand (bf16!)
    h_out_ty = np.ndarray[(m, p), np.dtype[dtype_out]]  # H tile, mm1 C operand (f32)
    w2_ty = np.ndarray[(p, n), np.dtype[dtype_in]]     # mm2 B tile
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]     # mm2 C tile

    # (kernels are constructed in build_program() once H's dtype is known)

    # ----- data movement: inputs (same pattern as single_core) -----
    # A -> mm1 (relaid to r x s micro-tiles)
    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    # W1 -> mm1 B (row-major k x p, relaid s x t)
    inW1 = ObjectFifo(w1_ty, name="inW1")
    w1_dims = [(k // s, s * p), (p // t, t), (s, p), (t, 1)]
    memW1 = inW1.cons().forward(name="memW1", dims_to_stream=w1_dims)

    # W2 -> mm2 B (row-major p x n, relaid s x t)
    inW2 = ObjectFifo(w2_ty, name="inW2")
    w2_dims = [(p // s, s * n), (n // t, t), (s, n), (t, 1)]
    memW2 = inW2.cons().forward(name="memW2", dims_to_stream=w2_dims)

    # ----- the ON-CHIP INTERMEDIATE H : core1 -> MemTile -> core2 -----
    # mm2's A operand must be bf16 (matmul_bf16_f32 takes bf16 inputs).  A
    # pure-IRON DMA cannot cast f32->bf16, so the fused path keeps H in bf16:
    # mm1 runs bf16->bf16, mm2 consumes the bf16 H directly.  The H fifo and
    # workers are assembled in build_program() (needs H dtype).

    # mm1 worker task: zero H tile, accumulate over K, release H.
    def core1_fn(of_a, of_w1, of_h_prod, zero, matmul):
        for _ in range_(M_div_m * P_div_p) if (M_div_m * P_div_p) > 1 else range(1):
            elem_h = of_h_prod.acquire(1)
            zero(elem_h)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                ea = of_a.acquire(1)
                ew = of_w1.acquire(1)
                matmul(ea, ew, elem_h)
                of_a.release(1)
                of_w1.release(1)
            of_h_prod.release(1)

    # mm2 worker task: zero C tile, accumulate over P (full H inner dim), release C.
    def core2_fn(of_h_cons, of_w2, of_c, zero, matmul):
        for _ in range_(M_div_m * N_div_n) if (M_div_m * N_div_n) > 1 else range(1):
            elem_c = of_c.acquire(1)
            zero(elem_c)
            for _ in range_(P_div_p) if P_div_p > 1 else range(1):
                eh = of_h_cons.acquire(1)
                ew = of_w2.acquire(1)
                matmul(eh, ew, elem_c)
                of_h_cons.release(1)
                of_w2.release(1)
            of_c.release(1)

    return (
        locals()
    )  # assembled in build_program()


def build_program(cfg, h_dtype_str):
    """Assemble fifos + workers + runtime using either a bf16 or f32 on-chip H.

    h_dtype_str: 'bf16'  -> mm1 emits bf16 H, mm2 consumes bf16 H directly (FUSED, works).
                 'f32'   -> mm1 emits f32 H; demonstrates the cast obstacle (see report).
    """
    L = cfg
    (
        M, K, P, N, m, k, p, n,
        dtype_in, dtype_out, dtype_in_str, dtype_out_str,
        r, s, t,
        M_div_m, K_div_k, P_div_p, N_div_n,
    ) = (
        L["M"], L["K"], L["P"], L["N"], L["m"], L["k"], L["p"], L["n"],
        L["dtype_in"], L["dtype_out"], L["dtype_in_str"], L["dtype_out_str"],
        L["r"], L["s"], L["t"],
        L["M_div_m"], L["K_div_k"], L["P_div_p"], L["N_div_n"],
    )
    A_ty, W1_ty, W2_ty, C_ty = L["A_ty"], L["W1_ty"], L["W2_ty"], L["C_ty"]
    a_ty, w1_ty, w2_ty, c_ty = L["a_ty"], L["w1_ty"], L["w2_ty"], L["c_ty"]
    inA, inW1, inW2 = L["inA"], L["inW1"], L["inW2"]
    memA, memW1, memW2 = L["memA"], L["memW1"], L["memW2"]
    core1_fn, core2_fn = L["core1_fn"], L["core2_fn"]
    dev = L["dev"]

    h_dtype = str_to_dtype(h_dtype_str)
    h_prod_ty = np.ndarray[(m, p), np.dtype[h_dtype]]  # mm1 C operand / mm2 A operand
    h_cons_ty = h_prod_ty

    # mm1 kernels emit h_dtype; mm2 A operand is h_dtype.
    zero_h = Kernel(f"zero_{h_dtype_str}", f"mm_{m}x{k}x{p}.o", [h_prod_ty])
    if h_dtype_str == "bf16":
        mm1 = Kernel("matmul_bf16_bf16", f"mm_{m}x{k}x{p}.o", [a_ty, w1_ty, h_prod_ty])
    else:
        mm1 = Kernel("matmul_bf16_f32", f"mm_{m}x{k}x{p}.o", [a_ty, w1_ty, h_prod_ty])
    zero_c = Kernel(f"zero_{dtype_out_str}", f"mm_{m}x{p}x{n}.o", [c_ty])
    mm2 = Kernel("matmul_bf16_f32", f"mm_{m}x{p}x{n}.o", [h_cons_ty, w2_ty, c_ty])

    # ---- H on-chip relayout through MemTile ----
    # The CORE tile DMA only supports <=3 transform dims; the 4-dim mmul
    # relayouts must run on the MemTile.  So core1 writes H in its native
    # mmul C-tile layout (NO core-side transform) and core2 reads H in its
    # native mmul A-tile layout (NO core-side transform); the two 4-dim
    # relayouts both run on MemTile DMAs:
    #   ofH : core1 --(plain)--> MemTile, MemTile consumer applies c_dims
    #         (mmul C-tile -> row-major)        [dims_from_stream, on MemTile]
    #   memH: MemTile --(a_dims, row-major -> mmul A-tile)--> core2
    #         [dims_to_stream, on MemTile producer side]
    c_dims_h = [(m // r, r * p), (r, t), (p // t, r * t), (t, 1)]   # mm1 C [m,p]
    a_dims_h = [(m // r, r * p), (p // s, s), (r, p), (s, 1)]        # mm2 A [m,p]

    # ofH consumer is the MemTile -> c_dims as dims_from_stream_per_cons runs
    # on the MemTile incoming DMA (4-dim OK).  forward() makes the MemTile the
    # producer of memH -> a_dims as dims_to_stream runs on the MemTile outgoing
    # DMA (4-dim OK).  core2 consumes plain (no core-side transform).
    ofH = ObjectFifo(
        h_prod_ty, name="ofH", depth=2, dims_from_stream_per_cons=c_dims_h
    )
    memH = ofH.cons().forward(name="memH", dims_to_stream=a_dims_h)

    worker1 = Worker(
        core1_fn,
        [memA.cons(), memW1.cons(), ofH.prod(), zero_h, mm1],
        stack_size=0xD00,
    )

    # Output C fifo (same relayout-out as single_core).
    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    worker2 = Worker(
        core2_fn,
        [memH.cons(), memW2.cons(), memC.prod(), zero_c, mm2],
        stack_size=0xD00,
    )

    # ---- TensorTiler access patterns for the host DMAs ----
    rows_per_block = 4
    A_tiles = TensorTiler2D.group_tiler(
        (M, K), (m, k), (1, K_div_k), pattern_repeat=P_div_p, prune_step=False
    )
    w1_tap = TensorTiler2D.group_tiler(
        (K, P), (k, p), (K_div_k, P_div_p), tile_group_col_major=True, prune_step=False
    )[0]
    w2_tap = TensorTiler2D.group_tiler(
        (P, N), (p, n), (P_div_p, N_div_n), tile_group_col_major=True, prune_step=False
    )[0]
    c_group_rows = min(rows_per_block // 2, M_div_m)
    C_tiles = TensorTiler2D.group_tiler(
        (M, N), (m, n), (c_group_rows, N_div_n), prune_step=False
    )

    rt = Runtime()
    with rt.sequence(A_ty, W1_ty, W2_ty, C_ty) as (A, W1, W2, C):
        rt.start(worker1, worker2)
        c_index = 0
        tgs = []
        for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                if num_tile_rows <= 0:
                    break
                tgs.append(rt.task_group())
                for tile_row in range(num_tile_rows):
                    tile_offset = (row_base + tile_row) % len(A_tiles)
                    rt.fill(inA.prod(), A, tap=A_tiles[tile_offset], task_group=tgs[-1])
                    rt.fill(inW1.prod(), W1, tap=w1_tap, task_group=tgs[-1])
                    rt.fill(inW2.prod(), W2, tap=w2_tap, task_group=tgs[-1])
                rt.drain(outC.cons(), C, tap=C_tiles[c_index], task_group=tgs[-1], wait=True)
                c_index += 1
                if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                    rt.finish_task_group(tgs[-2])
                    del tgs[-2]
        rt.finish_task_group(tgs[-1])
        del tgs[-1]

    dev_ty = NPU1() if dev == "npu" else NPU2()
    return Program(dev_ty, rt).resolve_program()


def main():
    ap = argparse.ArgumentParser(prog="Fused two-matmul (FFN) IRON design")
    ap.add_argument("--dev", choices=["npu", "npu2"], default="npu2")
    ap.add_argument("-M", type=int, default=64)
    ap.add_argument("-K", type=int, default=128)
    ap.add_argument("-P", type=int, default=128)
    ap.add_argument("-N", type=int, default=128)
    ap.add_argument("-m", type=int, default=64)
    ap.add_argument("-k", type=int, default=64)
    ap.add_argument("-p", type=int, default=64)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--dtype_in", default="bf16")
    ap.add_argument("--dtype_out", default="f32")
    ap.add_argument("--h_dtype", choices=["bf16", "f32"], default="bf16")
    args = ap.parse_args()

    cfg = my_ffn(
        args.dev, args.M, args.K, args.P, args.N,
        args.m, args.k, args.p, args.n,
        args.dtype_in, args.dtype_out,
    )
    cfg["dev"] = args.dev
    module = build_program(cfg, args.h_dtype)
    print(module)


if __name__ == "__main__":
    main()
