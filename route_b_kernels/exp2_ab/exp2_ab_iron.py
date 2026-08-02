# exp2_ab_iron.py -- A/B device probe. Single core, in[N] f32 -> out[4*N] f32
# ([raw|hw2x|sw2x|hwex]). Pattern copied from
# route_b_kernels/relpos_mha/probe_floor_iron.py (a working single-core IRON design
# already migrated to the current IRON API -- see toolchain.lock for the API-break
# notes this depends on: Runtime(seq_fn, fn_args), fill/drain on the ObjectFifo
# handle, Worker's core_body, Program(..., workers=).resolve_program()).
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse, sys
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

N = 1024  # must match -DEXP2AB_N in the Makefile (L1 budget: 40*N bytes with depth-2 double buffering, <64KB)


def my_exp2_ab(dev):
    in_ty = np.ndarray[(N,), np.dtype[np.float32]]
    out_ty = np.ndarray[(4 * N,), np.dtype[np.float32]]
    of_in = ObjectFifo(in_ty, name="ein")
    of_out = ObjectFifo(out_ty, name="eout")
    probe = Kernel("exp2_ab", "kernels.a", [in_ty, out_ty])

    def core_body(pin, pout, fn):
        ei = pin.acquire(1)
        eo = pout.acquire(1)
        fn(ei, eo)
        pin.release(1)
        pout.release(1)

    worker = Worker(core_body, [of_in.cons(), of_out.prod(), probe])

    def sequence(I, O, in_h, out_h):
        in_h.fill(I)
        out_h.drain(O, wait=True)

    rt = Runtime(sequence, [in_ty, out_ty, of_in.prod(), of_out.cons()])
    return Program(dev, rt, workers=[worker]).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-T", "--frames", dest="T", type=int, default=0)  # ignored; Makefile uniformity
opts, _ = p.parse_known_args(sys.argv[1:])
dev = NPU2() if opts.device == "npu2" else NPU1()
print(my_exp2_ab(dev))
