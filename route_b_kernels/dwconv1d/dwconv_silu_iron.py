# dwconv1d/dwconv_silu_iron.py -*- Python -*-
#
# FUSED depthwise-conv1d -> SiLU in ONE xclbin (conv-module step 3+4, roadmap 5-A rung).
# A TWO-STAGE on-chip pipeline per column: a dwconv core (row 0) computes the k=9 FIR and
# hands its f32 result to an on-chip f32 ObjectFifo that a silu core (row 1) consumes
# device-to-device -- the post-dwconv SiLU never touches host, and there is NO second
# hw-context switch (the SEPARATE silu xclbin cost a measured ~1 ms/block switch + host
# round-trip; this collapses both bricks into a single resident hw-context).
#
# Both cores stay SIMPLE single-op loops (dwconv core = FIR only; silu core = silu only),
# so this is IMMUNE to the alt-channel per-tile-loop miscompile (that needs a HEAVY fused
# epilogue in one loop; here the two ops live on two separate cores). See
# the dwconv-fused-epilogue-alt-channel-miscompile notes.
#
# Layout: [C=1024, T=400]. One ObjectFifo tile == one channel's time series (dwconv) ==
# one row (silu -- a channel IS a row). C channels split across `columns`; each column's
# dwconv core loops over its share, streaming per-channel f32 tiles to its silu core.
#   host --in(bf16)--> [dwconv core] --mid(f32) on-chip--> [silu core] --out(f32)--> host
#   host --w(bf16)--^
# ABI mirrors dwconv1d.py (1=instr, 3=in, 4=weights, 5=out) so the Rust ConvDwSilu path
# reuses the dwconv dispatch shape.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D
from aie.iron.controlflow import range_

C = 1024  # channels (Parakeet d_model)
T = 400   # time steps (encoder frame count baked at build)
KW = 16   # weight tile size (taps[0..8] + bias[9], rest unused)


def my_dwconv_silu(dev, num_columns):
    bf16 = bfloat16
    f32 = np.float32
    if C % num_columns != 0:
        raise ValueError(f"C={C} must be divisible by columns={num_columns}")
    cpc = C // num_columns  # channels per column

    in_tile_ty = np.ndarray[(T,), np.dtype[bf16]]
    w_tile_ty = np.ndarray[(KW,), np.dtype[bf16]]
    mid_tile_ty = np.ndarray[(T,), np.dtype[f32]]   # dwconv f32 out -> silu in (on-chip)
    out_tile_ty = np.ndarray[(T,), np.dtype[f32]]

    in_tensor_ty = np.ndarray[(C * T,), np.dtype[bf16]]
    w_tensor_ty = np.ndarray[(C * KW,), np.dtype[bf16]]
    out_tensor_ty = np.ndarray[(C * T,), np.dtype[f32]]

    of_ins = [ObjectFifo(in_tile_ty, name=f"in_{i}") for i in range(num_columns)]
    of_ws = [ObjectFifo(w_tile_ty, name=f"w_{i}") for i in range(num_columns)]
    of_mids = [ObjectFifo(mid_tile_ty, name=f"mid_{i}") for i in range(num_columns)]  # core->core
    of_outs = [ObjectFifo(out_tile_ty, name=f"out_{i}") for i in range(num_columns)]

    dwconv = Kernel(
        "dwconv1d_k9_bf16_f32o", "kernels.a", [in_tile_ty, w_tile_ty, mid_tile_ty]
    )
    silu = Kernel("silu_row", "silu_brick.o", [mid_tile_ty, out_tile_ty, np.int32])

    def dwconv_body(of_in, of_w, of_mid, dwconv_fn):
        for _ in range_(cpc):
            ei = of_in.acquire(1)
            ew = of_w.acquire(1)
            em = of_mid.acquire(1)
            dwconv_fn(ei, ew, em)
            of_in.release(1)
            of_w.release(1)
            of_mid.release(1)

    def silu_body(of_mid, of_out, silu_fn):
        for _ in range_(cpc):
            em = of_mid.acquire(1)
            eo = of_out.acquire(1)
            silu_fn(em, eo, T)
            of_mid.release(1)
            of_out.release(1)

    workers = []
    for i in range(num_columns):
        workers.append(
            Worker(dwconv_body, [of_ins[i].cons(), of_ws[i].cons(), of_mids[i].prod(), dwconv])
        )
    # EXACT-f32 silu (SILU_MODE=2) spills a >1024 B frame. The IRON default worker stack is
    # 1024 B and the allocator places the EVEN objectfifo output buffer immediately after it,
    # so an oversize frame overflows into that buffer (even/odd corruption) or the lock region
    # (hang). Size the silu-core stack window past the exact-f32 frame. Root cause + fix:
    # the dwconv-fused-epilogue-alt-channel-miscompile notes.
    silu_stack = 8192
    for i in range(num_columns):
        workers.append(
            Worker(silu_body, [of_mids[i].cons(), of_outs[i].prod(), silu],
                   stack_size=silu_stack)
        )

    # Modern IRON idiom (place-tiles toolchain): simple_tiler + plain fill/drain (mirrors dwconv1d.py /
    # silu_iron.py). Each column i handles cpc contiguous channels == cpc contiguous rows.
    in_taps = TensorTiler2D.simple_tiler((C, T), (cpc, T))
    w_taps = TensorTiler2D.simple_tiler((C, KW), (cpc, KW))
    out_taps = TensorTiler2D.simple_tiler((C, T), (cpc, T))

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

print(my_dwconv_silu(dev, int(opts.cols)))
