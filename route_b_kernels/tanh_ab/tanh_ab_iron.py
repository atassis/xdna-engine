# tanh_ab_iron.py -- A/B device probe. Single core, in[N] f32 -> out[5*N] f32
# ([raw|hwtanh|swtanh|gelu_hw|gelu_sw]). Pattern copied from
# route_b_kernels/exp2_ab/exp2_ab_iron.py, which is itself the migrated single-core
# shape from relpos_mha/probe_floor_iron.py -- see toolchain.lock for the IRON API
# break this depends on (Runtime(seq_fn, fn_args), fill/drain on the ObjectFifo
# handle, Worker's core_body, Program(..., workers=).resolve_program()).
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse, sys
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

# Must match -DTANHAB_N in the Makefile. L1 budget: depth-2 double buffering over a
# 4-byte in and a 20-byte out per element = 48*N bytes = 24KB at N=512, under the
# 64KB tile memory. 512 rather than 1024 so the 5-column out buffer (10240 B) still
# fits ONE 16KB bank -- at 1024 it did not, and the allocator fell back from
# bank-aware to basic sequential placement, which is the clearance class that has
# silently overwritten objectFIFO buffers in this tree before.
N = 512


def my_tanh_ab(dev):
    in_ty = np.ndarray[(N,), np.dtype[np.float32]]
    out_ty = np.ndarray[(5 * N,), np.dtype[np.float32]]
    of_in = ObjectFifo(in_ty, name="tin")
    of_out = ObjectFifo(out_ty, name="tout")
    probe = Kernel("tanh_ab", "kernels.a", [in_ty, out_ty])

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
print(my_tanh_ab(dev))
