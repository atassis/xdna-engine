# dwconv1d/dwconv_silu_tmajor_iron.py -*- Python -*-
#
# TIME-MAJOR fused depthwise-conv1d -> SiLU in ONE xclbin (conv step 3b -- the transpose-DISSOLVING
# layout). The channel-major dwconv_silu_iron.py owns [C,T] (one channel's time series per tile, FIR
# sliding along t) and is bracketed by two HOST transposes (GLU[T,D]->[D,T] in, [D,T]->[T,D] out). This
# rotates the layout to [T,D] so the brick consumes GLU's [T,D] directly and emits pw2's [T,D] directly
# -- the two host transposes are dissolved. A feasibility spike proved the on-chip transpose route
# re-hits the n-D-DMA co-residency hang; the time-major FIR has NO shuffle and NO cross-column DMA
# (every DMA is a plain strided read, inner stride 1), so it sidesteps the hang entirely.
#
# LAYOUT + TILING ([T,D] = [400,1024]):
#   * D (d_model) is the VECTORIZED per-lane axis; TIME carries the k=9 'same' halo.
#   * Host pads the input to [T+2P, D]=[408,1024] (P=4 zero rows top+bottom). Each core owns one
#     D-chunk mb_D = C // num_columns (=128 at 8 cols) and streams nT = T/MB_T overlapping time-tiles:
#     input tile [MB_T+K-1, mb_D]=[28,128] (halo K-1=8 rows), output tile [MB_T, mb_D]=[20,128].
#     Consecutive input tiles step MB_T rows but span MB_T+8 -> overlap by 8 (the halo), zero-padded
#     at the sequence ends by the host padding.
#   * dwconv core (f32 out) -> on-chip f32 ObjectFifo -> silu core (bf16-tanh SILU_MODE 0), two
#     SEPARATE simple-loop cores (immune to the alt-channel per-tile-loop miscompile), one hw-context.
#   host --in(bf16 padded)--> [dwconv_tmajor core] --mid(f32) on-chip--> [silu core] --out(f32)--> host
#   host --w(bf16 tap-major)--^
# ABI mirrors dwconv_silu_iron.py (1=instr, 3=in, 4=weights, 5=out) so the Rust ConvDwSiluT path reuses
# the 3-buffer dispatch shape (in bf16 g3, w bf16 g4, out f32 g5).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.controlflow import range_

C = 1024   # D = d_model (the per-lane vectorized axis)
T = 400    # time steps baked at build (Parakeet frame cap)
P = 4      # 'same' pad = (K-1)/2
K = 9      # depthwise kernel width
KH = K - 1 # halo rows carried in the overlapping input tile (=8)
MB_T = 20  # output time-rows per tile -- MUST match the kernel's baked TT (dwconv1d_k9_tmajor)


def my_dwconv_silu_tmajor(dev, num_columns):
    bf16 = bfloat16
    f32 = np.float32
    if C % num_columns != 0:
        raise ValueError(f"C={C} must be divisible by columns={num_columns}")
    mb_D = C // num_columns   # D-chunk per column -- MUST match the kernel's baked DD (=128 at 8 cols)
    if mb_D != 128:
        raise ValueError(f"mb_D={mb_D} must be 128 (kernel bakes DD=128); use num_columns=8")
    if T % MB_T != 0:
        raise ValueError(f"T={T} must be divisible by MB_T={MB_T}")
    nT = T // MB_T            # time-tiles per column

    in_tile_ty = np.ndarray[((MB_T + KH) * mb_D,), np.dtype[bf16]]   # [MB_T+8, mb_D] halo tile
    w_tile_ty = np.ndarray[((K + 1) * mb_D,), np.dtype[bf16]]        # [K+1, mb_D] tap-major (taps+bias)
    mid_tile_ty = np.ndarray[(MB_T * mb_D,), np.dtype[f32]]          # dwconv f32 out -> silu in (on-chip)
    out_tile_ty = np.ndarray[(MB_T * mb_D,), np.dtype[f32]]

    in_tensor_ty = np.ndarray[((T + 2 * P) * C,), np.dtype[bf16]]    # padded [T+2P, C]
    w_tensor_ty = np.ndarray[((K + 1) * C,), np.dtype[bf16]]         # tap-major [K+1, C]
    out_tensor_ty = np.ndarray[(T * C,), np.dtype[f32]]

    of_ins = [ObjectFifo(in_tile_ty, name=f"in_{i}") for i in range(num_columns)]
    of_ws = [ObjectFifo(w_tile_ty, name=f"w_{i}") for i in range(num_columns)]
    of_mids = [ObjectFifo(mid_tile_ty, name=f"mid_{i}") for i in range(num_columns)]  # core->core
    of_outs = [ObjectFifo(out_tile_ty, name=f"out_{i}") for i in range(num_columns)]

    dwconv = Kernel(
        "dwconv1d_k9_tmajor", "kernels.a", [in_tile_ty, w_tile_ty, mid_tile_ty]
    )
    silu = Kernel("silu_row", "silu_brick.o", [mid_tile_ty, out_tile_ty, np.int32])

    # Weights depend only on D (same for every time-tile of a column), so the dwconv core acquires its
    # [K+1, mb_D] tap tile ONCE and reuses it across all nT tiles (stationary weights, no re-DMA).
    def dwconv_body(of_in, of_w, of_mid, dwconv_fn):
        ew = of_w.acquire(1)
        for _ in range_(nT):
            ei = of_in.acquire(1)
            em = of_mid.acquire(1)
            dwconv_fn(ei, ew, em)
            of_in.release(1)
            of_mid.release(1)
        of_w.release(1)

    # SiLU is elementwise, so the [MB_T, mb_D] tile is just MB_T*mb_D contiguous f32 (layout-agnostic).
    silu_cols = MB_T * mb_D
    if silu_cols % 16 != 0:
        raise ValueError(f"silu cols={silu_cols} must be a multiple of the 16-lane vector width")

    def silu_body(of_mid, of_out, silu_fn):
        for _ in range_(nT):
            em = of_mid.acquire(1)
            eo = of_out.acquire(1)
            silu_fn(em, eo, silu_cols)
            of_mid.release(1)
            of_out.release(1)

    workers = []
    for i in range(num_columns):
        workers.append(
            Worker(dwconv_body, [of_ins[i].cons(), of_ws[i].cons(), of_mids[i].prod(), dwconv])
        )
    # Keep the silu-core stack past the frame (the same guard the channel-major brick carries: the IRON
    # default 1024 B stack overflows into the adjacent EVEN objectfifo output buffer -> even/odd
    # corruption or hang). See docs/log/2026-07/silu-stack-overflow-root-cause-and-wer-reframe.md.
    silu_stack = 8192
    for i in range(num_columns):
        workers.append(
            Worker(silu_body, [of_mids[i].cons(), of_outs[i].prod(), silu],
                   stack_size=silu_stack)
        )

    # Explicit n-D taps (mirrors transpose_iron.py). Every inner stride is 1 -> plain strided read, NOT
    # an element-transposing DMA. Column i owns D-chunk [i*mb_D : (i+1)*mb_D].
    #   INPUT (overlapping, from padded [T+2P, C]): tile (j) = padded rows [j*MB_T : j*MB_T+MB_T+KH),
    #     cols [i*mb_D : i*mb_D+mb_D]. linear = j*MB_T*C + ii*C + jj + i*mb_D.
    #     sizes=[nT, MB_T+KH, mb_D], strides=[MB_T*C, C, 1]. Consecutive j overlap by KH rows (the halo).
    #   WEIGHT (one [K+1, mb_D] tile, tap-major): sizes=[K+1, mb_D], strides=[C, 1].
    #   OUTPUT (non-overlapping, to [T, C]): sizes=[nT, MB_T, mb_D], strides=[MB_T*C, C, 1].
    in_taps = [
        TensorAccessPattern(
            tensor_dims=(T + 2 * P, C),
            offset=i * mb_D,
            sizes=[nT, MB_T + KH, mb_D],
            strides=[MB_T * C, C, 1],
        )
        for i in range(num_columns)
    ]
    w_taps = [
        TensorAccessPattern(
            tensor_dims=(K + 1, C),
            offset=i * mb_D,
            sizes=[K + 1, mb_D],
            strides=[C, 1],
        )
        for i in range(num_columns)
    ]
    out_taps = [
        TensorAccessPattern(
            tensor_dims=(T, C),
            offset=i * mb_D,
            sizes=[nT, MB_T, mb_D],
            strides=[MB_T * C, C, 1],
        )
        for i in range(num_columns)
    ]

    rt = Runtime()
    with rt.sequence(in_tensor_ty, w_tensor_ty, out_tensor_ty) as (X, W, Y):
        rt.start(*workers)
        for i in range(num_columns):
            rt.fill(of_ins[i].prod(), X, in_taps[i])
            rt.fill(of_ws[i].prod(), W, w_taps[i])
        for i in range(num_columns):
            rt.drain(of_outs[i].cons(), Y, out_taps[i], wait=True)

    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="npu or npu2")
p.add_argument("-co", "--columns", required=True, dest="cols")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"[ERROR] Device name {opts.device} is unknown")

print(my_dwconv_silu_tmajor(dev, int(opts.cols)))
