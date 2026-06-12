#
# STEP C — O1/G2 GEMM+LN CO-RESIDENCY PROBE design.
#
# THE QUESTION (internal notes VERDICT + 01-design §2): the measured 2.44 ms/switch
# (dual_precision_probe) is a TWO-XCLBIN array reload. V2 already proves ONE resident xclbin +
# many instruction streams alternates switch-free (~floor 0.7-1.0 ms) — but every V2 stream
# drives the SAME matmul core ELF. The open bit: can ONE xclbin hold TWO *genuinely different
# core programs* (a GEMM .elf and an LN .elf) at all — compile on open Peano and run without the
# "compiles-clean-but-hangs" hazard (internal notes Risk; docs/10 s2 GEMM->GEMM deadlock)? If
# yes, then because it is one xclbin = one hwctx, alternating its dispatches is switch-free BY
# CONSTRUCTION (no 2.44 ms reload) — which is the whole point.
#
# A FINDING THAT SHAPED THIS PROBE: the whole_array/IRON dataflow model statically requires every
# ObjectFifo to have BOTH endpoints. So you cannot "start a core but leave its fifo unfed this
# dispatch" — i.e. the naive fixed-partition picture (dedicate columns to LN, idle them during
# matmul) is NOT directly expressible. The two switch-free forms that ARE expressible:
#   (1) CONCURRENT fixed-partition — all cores active every dispatch, split spatially by function
#       (GEMM core || LN core, this probe), and
#   (2) same-core RTP time-mux — one core ELF holding both behaviors, selected per stream (the
#       shipped modal silu/identity pattern; the harder full-width path, separate workstream).
# This probe builds (1): the minimal concurrent co-residency of a GEMM core and an LN core.
#
# mode=both : GEMM core (matmul_bf16_f32, mm.cc) || LN core (layer_norm_welford f32, layer_norm.cc)
#             both active in ONE dispatch on DISJOINT regions of the f32 C buffer (row0 = GEMM out,
#             row1 = LN in+out) so there is no shared-buffer DMA race. This is THE artifact.
# mode=gemm / mode=ln : single-core latency baselines (different topology; for the cost delta only).
# All three keep within the fixed run_matmul8 (A,B,C,tmp,trace) host ABI (A,B bf16 in; C f32 in/out).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D


# bf16 matmul micro-MAC dims on AIE2P (matches ffn_gemm2 / whole_array native).
R, S, T = 4, 8, 8


def build(dev, mode, m, k, n):
    # Single GEMM tile (M/m = K/k = N/n = 1) so the GEMM is exactly one core.
    gm, gk, gn = m, k, n
    # LN slab reuses one f32 C row (lr*lc == gm*gn) so everything stays in the A,B,C ABI.
    lr, lc = m, n
    assert gm % R == 0 and gk % S == 0 and gn % T == 0, "tile must satisfy bf16 MAC dims"
    assert lc % 16 == 0, "layer_norm_welford<float,16> vectorizes cols by 16"

    f32 = np.float32

    # ---- host buffers (flat) + tile types ----
    A_ty = np.ndarray[(gm * gk,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(gk * gn,), np.dtype[bfloat16]]
    # C carries BOTH outputs on disjoint rows: row0 = GEMM [gm,gn], row1 = LN [lr*lc == gm*gn].
    C_ty = np.ndarray[(2 * gm * gn,), np.dtype[f32]]

    a_ty = np.ndarray[(gm, gk), np.dtype[bfloat16]]
    b_ty = np.ndarray[(gk, gn), np.dtype[bfloat16]]
    c_ty = np.ndarray[(gm, gn), np.dtype[f32]]
    ln_ty = np.ndarray[(lr * lc,), np.dtype[f32]]  # plain f32 slab, no relayout

    want_gemm = mode in ("both", "gemm")
    want_ln = mode in ("both", "ln")

    workers = []

    # ---- GEMM dataflow (4-dim mmul relayouts run on the MemTile, like ffn_gemm2) ----
    if want_gemm:
        inA = ObjectFifo(a_ty, name="inA")
        a_dims = [(gm // R, R * gk), (gk // S, S), (R, gk), (S, 1)]
        memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

        inB = ObjectFifo(b_ty, name="inB")
        b_dims = [(gk // S, S * gn), (gn // T, T), (S, gn), (T, 1)]
        memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

        memC = ObjectFifo(c_ty, name="memC")
        c_dims = [(gm // R, R * gn), (R, T), (gn // T, R * T), (T, 1)]
        outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

        zero_c = Kernel("zero_f32", f"mm_{gm}x{gk}x{gn}.o", [c_ty])
        mm = Kernel("matmul_bf16_f32", f"mm_{gm}x{gk}x{gn}.o", [a_ty, b_ty, c_ty])

        def gemm_fn(of_a, of_b, of_c, zero, matmul):
            ec = of_c.acquire(1)
            zero(ec)
            ea = of_a.acquire(1)
            eb = of_b.acquire(1)
            matmul(ea, eb, ec)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

        gemm_worker = Worker(
            gemm_fn, [memA.cons(), memB.cons(), memC.prod(), zero_c, mm], stack_size=0xD00
        )
        workers.append(gemm_worker)

    # ---- LN dataflow (plain host<->core fifos, like ml/layernorm) ----
    if want_ln:
        inLN = ObjectFifo(ln_ty, name="inLN")
        outLN = ObjectFifo(ln_ty, name="outLN")
        ln_k = Kernel(
            "layer_norm_welford", "layer_norm.o", [ln_ty, ln_ty, np.int32, np.int32]
        )

        def ln_fn(of_in, of_out, ln):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            ln(ei, eo, lr, lc)
            of_in.release(1)
            of_out.release(1)

        ln_worker = Worker(ln_fn, [inLN.cons(), outLN.prod(), ln_k], stack_size=0xD00)
        workers.append(ln_worker)

    # ---- host DMA access patterns ----
    A_tap = TensorTiler2D.simple_tiler((gm, gk), (gm, gk))[0]
    B_tap = TensorTiler2D.simple_tiler((gk, gn), (gk, gn))[0]
    # C as 2 disjoint rows of (gm*gn): row0 GEMM out, row1 LN in/out.
    C_rows = TensorTiler2D.simple_tiler((2, gm * gn), (1, gm * gn))

    dev_ty = NPU1() if dev == "npu" else NPU2()
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
        rt.start(*workers)
        if want_gemm:
            rt.fill(inA.prod(), A, A_tap)
            rt.fill(inB.prod(), B, B_tap)
            rt.drain(outC.cons(), C, C_rows[0], wait=True)
        if want_ln:
            rt.fill(inLN.prod(), C, C_rows[1])  # LN reads row1 (host pre-filled)
            rt.drain(outLN.cons(), C, C_rows[1], wait=True)  # LN writes row1
    return Program(dev_ty, rt).resolve_program()


def main():
    ap = argparse.ArgumentParser(prog="STEP C O1/G2 GEMM+LN co-residency probe")
    ap.add_argument("--dev", choices=["npu", "npu2"], default="npu2")
    ap.add_argument("--mode", choices=["both", "gemm", "ln"], default="both")
    ap.add_argument("-m", type=int, default=32)
    ap.add_argument("-k", type=int, default=32)
    ap.add_argument("-n", type=int, default=64)
    # accepted for makefile-common compatibility (unused by this single-tile probe)
    ap.add_argument("-M", type=int, default=32)
    ap.add_argument("-K", type=int, default=32)
    ap.add_argument("-N", type=int, default=64)
    ap.add_argument("--n-aie-cols", type=int, default=1)
    ap.add_argument("--b-col-maj", type=int, default=0)
    ap.add_argument("--dtype_in", default="bf16")
    ap.add_argument("--dtype_out", default="f32")
    ap.add_argument("--trace_size", type=int, default=0)
    args = ap.parse_args()
    print(build(args.dev, args.mode, args.m, args.k, args.n))


if __name__ == "__main__":
    main()
