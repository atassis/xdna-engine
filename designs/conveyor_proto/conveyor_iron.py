# Trivial 2-stage CONVEYOR prototype (mechanism proof for the MHA conveyor).
#
# DDR --in--> [tile A: stage_add1] --of_ab--> [tile B: stage_mul2] --out--> DDR
#
# The point: of_ab is an ObjectFifo between two DIFFERENT Workers, so place-tiles
# puts stage A and stage B on DISTINCT compute tiles and routes of_ab tile-to-tile
# (the "conveyor belt"). Each tile's program holds only ITS one stage -> the per-tile
# program-memory ceiling is shared across tiles, not crammed onto one. This is the
# raw-ObjectFifo template (like vision/edge_detect) stripped to 2 heterogeneous stages.
# Bare Program(dev, rt).resolve_program() -- place-tiles model, no explicit Tile() pins.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys
import argparse
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

N = 256  # int32 elements per tile (1 KB) -- fits L1 trivially


def build(dev):
    tile_ty = np.ndarray[(N,), np.dtype[np.int32]]

    add1 = Kernel("stage_add1", "kernels.a", [tile_ty, tile_ty, np.int32])
    mul2 = Kernel("stage_mul2", "kernels.a", [tile_ty, tile_ty, np.int32])

    of_in = ObjectFifo(tile_ty, name="in", depth=2)      # DDR -> tile A
    of_ab = ObjectFifo(tile_ty, name="ab", depth=2)      # tile A -> tile B  (THE BELT)
    of_out = ObjectFifo(tile_ty, name="out", depth=2)    # tile B -> DDR

    def stage_a(f_in, f_ab, k_add1):
        ei = f_in.acquire(1)
        eo = f_ab.acquire(1)
        k_add1(ei, eo, N)
        f_in.release(1)
        f_ab.release(1)

    def stage_b(f_ab, f_out, k_mul2):
        ei = f_ab.acquire(1)
        eo = f_out.acquire(1)
        k_mul2(ei, eo, N)
        f_ab.release(1)
        f_out.release(1)

    worker_a = Worker(stage_a, [of_in.cons(), of_ab.prod(), add1])
    worker_b = Worker(stage_b, [of_ab.cons(), of_out.prod(), mul2])

    rt = Runtime()
    with rt.sequence(tile_ty, tile_ty) as (inp, outp):
        rt.start(worker_a, worker_b)
        rt.fill(of_in.prod(), inp)
        rt.drain(of_out.cons(), outp, wait=True)

    return Program(dev, rt).resolve_program()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dev", required=True, dest="device", help="npu or npu2")
opts = ap.parse_args(sys.argv[1:])
dev = NPU2() if opts.device == "npu2" else NPU1()
print(build(dev))
