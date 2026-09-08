# MINIMAL 2-worker QUERY-TILED passthrough -- isolates the even/odd streaming bug.
#   DDR{x, N_QT tiles} -> [A: +1] --belt--> [B: *2] -> DDR{y, N_QT tiles}
# NO attention math, NO k/V, NO f32 belt -- just 2 chained workers streaming N tiles. If this shows
# even/odd, the bug is the generic 2-worker-belt-N-tile streaming (not attention-specific). Uses all
# the fixes found: whole-buffer sequence args, advancing taps, range_ hardware loop.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import sys, argparse
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.controlflow import range_

TILE = 256
N_QT = 16


def build(dev):
    t_ty = np.ndarray[(TILE,), np.dtype[np.int32]]                 # int32 tile (in/out fifo object)
    tf_ty = np.ndarray[(TILE,), np.dtype[np.float32]]              # f32 tile (THE BELT -- like attn ac)
    full_ty = np.ndarray[(N_QT * TILE,), np.dtype[np.int32]]       # whole stream (seq arg)
    a = Kernel("stage_f32_a", "kernels.a", [t_ty, tf_ty, np.int32])   # int32 -> f32 belt
    b = Kernel("stage_f32_b", "kernels.a", [tf_ty, t_ty, np.int32])   # f32 belt -> int32

    of_x = ObjectFifo(t_ty, name="x", depth=2)
    of_ab = ObjectFifo(tf_ty, name="ab", depth=2)   # f32 belt
    of_y = ObjectFifo(t_ty, name="y", depth=2)

    def stage_a(f_x, f_ab, k1):
        for _ in range_(N_QT):
            ex = f_x.acquire(1); eab = f_ab.acquire(1)
            k1(ex, eab, TILE)
            f_x.release(1); f_ab.release(1)

    def stage_b(f_ab, f_y, k2):
        for _ in range_(N_QT):
            eab = f_ab.acquire(1); ey = f_y.acquire(1)
            k2(eab, ey, TILE)
            f_ab.release(1); f_y.release(1)

    wa = Worker(stage_a, [of_x.cons(), of_ab.prod(), a])
    wb = Worker(stage_b, [of_ab.cons(), of_y.prod(), b])

    tap = TensorAccessPattern([N_QT * TILE], 0, [N_QT, 1, 1, TILE], [TILE, 0, 0, 1])
    rt = Runtime()
    with rt.sequence(full_ty, full_ty) as (X, Y):
        rt.start(wa, wb)
        rt.fill(of_x.prod(), X, tap=tap)
        rt.drain(of_y.cons(), Y, tap=tap, wait=True)
    return Program(dev, rt).resolve_program()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dev", required=True, dest="device")
opts = ap.parse_args(sys.argv[1:])
dev = NPU2() if opts.device == "npu2" else NPU1()
print(build(dev))
