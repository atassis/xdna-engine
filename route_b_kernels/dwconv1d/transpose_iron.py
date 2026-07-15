# dwconv1d/transpose_iron.py -*- Python -*-
#
# On-chip COMPUTE-tile transpose design for the Conformer conv module (Task 0.1).
# Transposes [M, N] -> [N, M] with the ELEMENT transpose done on the compute core
# (transpose_tile.cc), NOT in the DMA -- this is the de-risking enabler for step 3b
# (kill the two HOST transposes GLU[T,D]->dwconv[D,T]->pw2[T,D]) while AVOIDING the
# transposing n-D DMA that is known to hang when co-resident (blocker npu.rs:740).
#
# DATAFLOW (mirrors dwconv_silu_iron.py / silu_iron.py: place-tiles, simple_tiler,
# per-core loop, Program(...).resolve_program()):
#   * Row axis M split across `cols` cores; each core owns rpc = M/cols contiguous
#     rows == tpc = rpc/mb tiles of shape [mb, N].
#   * INPUT tap = simple_tiler((M,N),(rpc,N))[i]  -> contiguous rows, NO DMA transpose.
#   * Core transposes each [mb, N] tile to [N, mb] IN-CORE (byte copy, bit-exact).
#   * OUTPUT tap = a block-SCATTER TensorAccessPattern with UNIT inner stride that
#     places the tpc transposed [N, mb] tiles into output columns [i*rpc : (i+1)*rpc]
#     of the [N, M] result. sizes=[tpc, N, mb], strides=[mb, M, 1], offset=i*rpc.
#     Inner stride 1 => this is a plain strided scatter, NOT an element-transposing DMA.
#
# Same design serves BOTH directions: forward [D=1024,T=400]->[T,D] is (M=1024,N=400);
# inverse [T,D]->[D,T] is (M=400,N=1024). Element type bf16 (uint16) or f32 (uint32);
# the transpose is a byte copy so it is bit-exact for either.
#
# Build:  see Makefile.transpose. Args: -d npu2 -M .. -N .. -mb .. -co ..  -dt bf16|f32
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorTiler2D, TensorAccessPattern
from aie.iron.controlflow import range_


def my_transpose(dev, M, N, mb, nb, num_cores, dt, objname):
    elem = bfloat16 if dt == "bf16" else np.float32
    ebytes = 2 if dt == "bf16" else 4
    if M % num_cores != 0:
        raise ValueError(f"M={M} must be divisible by cores={num_cores}")
    rpc = M // num_cores            # rows per core (each core owns contiguous rows)
    if rpc % mb != 0:
        raise ValueError(f"rows-per-core={rpc} must be divisible by mb={mb}")
    if N % nb != 0:
        raise ValueError(f"N={N} must be divisible by nb={nb}")
    rbpc = rpc // mb                # row-blocks per core
    nC = N // nb                    # col-blocks (full N width)
    # HW DMA constraints (see Makefile.transpose notes): the innermost stride-1 run of
    # BOTH taps must be a multiple of 4 bytes, and NO BD dim size may exceed 1023.
    if (nb * ebytes) % 4 != 0 or (mb * ebytes) % 4 != 0:
        raise ValueError(f"inner runs nb={nb},mb={mb} at {ebytes}B must be multiples of 4 bytes")
    for s in (rbpc, nC, mb, nb):
        if s > 1023:
            raise ValueError(f"BD dim size {s} exceeds hardware 1023 limit -- tile smaller")

    in_tile_ty = np.ndarray[(mb * nb,), np.dtype[elem]]   # one [mb, nb] block
    out_tile_ty = np.ndarray[(nb * mb,), np.dtype[elem]]  # its [nb, mb] transpose
    in_tensor_ty = np.ndarray[(M * N,), np.dtype[elem]]
    out_tensor_ty = np.ndarray[(N * M,), np.dtype[elem]]

    of_ins = [ObjectFifo(in_tile_ty, name=f"in_{i}") for i in range(num_cores)]
    of_outs = [ObjectFifo(out_tile_ty, name=f"out_{i}") for i in range(num_cores)]

    tpose = Kernel(
        "transpose_tile",
        objname,
        [in_tile_ty, out_tile_ty, np.int32, np.int32],
    )

    tiles_per_core = rbpc * nC

    def core_body(of_in, of_out, tpose_fn):
        for _ in range_(tiles_per_core):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            tpose_fn(ei, eo, mb, nb)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, fn_args=[of_ins[i].cons(), of_outs[i].prod(), tpose])
        for i in range(num_cores)
    ]

    # Core i owns contiguous input rows [i*rpc:(i+1)*rpc]. It streams tiles in
    # (row-block rb, col-block cb) order; each tile is a [mb, nb] sub-block.
    #
    # INPUT tap (4D): elem (rb, cb, ii, jj) = input[i*rpc + rb*mb + ii, cb*nb + jj]
    #   linear = i*rpc*N + rb*mb*N + cb*nb + ii*N + jj
    #   -> sizes=[rbpc, nC, mb, nb], strides=[mb*N, nb, N, 1], inner run nb.
    # OUTPUT tap (4D): transposed tile [nb, mb] goes to
    #   output[cb*nb + p, i*rpc + rb*mb + q]  (p<nb original col, q<mb original row)
    #   linear = i*rpc + rb*mb + cb*nb*M + p*M + q
    #   -> sizes=[rbpc, nC, nb, mb], strides=[mb, nb*M, M, 1], inner run mb.
    # Inner stride is 1 in BOTH => plain strided scatter, NOT an element-transposing DMA.
    in_taps = [
        TensorAccessPattern(
            tensor_dims=(M, N),
            offset=i * rpc * N,
            sizes=[rbpc, nC, mb, nb],
            strides=[mb * N, nb, N, 1],
        )
        for i in range(num_cores)
    ]
    out_taps = [
        TensorAccessPattern(
            tensor_dims=(N, M),
            offset=i * rpc,
            sizes=[rbpc, nC, nb, mb],
            strides=[mb, nb * M, M, 1],
        )
        for i in range(num_cores)
    ]

    rt = Runtime()
    with rt.sequence(in_tensor_ty, out_tensor_ty) as (X, Y):
        rt.start(*workers)
        for i in range(num_cores):
            rt.fill(of_ins[i].prod(), X, in_taps[i])
        for i in range(num_cores):
            rt.drain(of_outs[i].cons(), Y, out_taps[i], wait=True)

    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="npu or npu2")
p.add_argument("-M", required=True, type=int, dest="M", help="input rows")
p.add_argument("-N", required=True, type=int, dest="N", help="input cols")
p.add_argument("-mb", required=True, type=int, dest="mb", help="row-block per tile")
p.add_argument("-nb", type=int, default=0, dest="nb",
               help="col-block per tile (0 => full N; tile the wide axis to respect the 1023 BD limit)")
p.add_argument("-co", "--cores", required=True, type=int, dest="cores")
p.add_argument("-dt", "--dtype", default="bf16", dest="dt", choices=["bf16", "f32"])
p.add_argument("-obj", "--objname", default="transpose_tile.o", dest="objname",
               help="kernel object filename the emitted MLIR references")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"[ERROR] Device name {opts.device} is unknown")

nb = opts.nb if opts.nb > 0 else opts.N
print(my_transpose(dev, opts.M, opts.N, opts.mb, nb, opts.cores, opts.dt, opts.objname))
