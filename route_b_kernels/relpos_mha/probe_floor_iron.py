# probe_floor_iron.py -- STEP=9 micro-probe. Single core, in[PROBE_N] f32 -> out[4*PROBE_N] f32
# ([trunc|floor|frac|pow2k]). Isolates which aie2p f32<->int vector op mangles the sw-exp2 floor.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import argparse, sys
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

N = 1024  # must match -DPROBE_N in the kernel


def my_probe(dev):
    in_ty = np.ndarray[(N,), np.dtype[np.float32]]
    out_ty = np.ndarray[(4 * N,), np.dtype[np.float32]]
    of_in = ObjectFifo(in_ty, name="pin")
    of_out = ObjectFifo(out_ty, name="pout")
    probe = Kernel("probe_floor", "kernels.a", [in_ty, out_ty])

    def core_body(pin, pout, fn):
        ei = pin.acquire(1)
        eo = pout.acquire(1)
        fn(ei, eo)
        pin.release(1)
        pout.release(1)

    worker = Worker(core_body, [of_in.cons(), of_out.prod(), probe])
    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (I, O):
        rt.start(worker)
        rt.fill(of_in.prod(), I)
        rt.drain(of_out.cons(), O, wait=True)
    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device")
p.add_argument("-T", "--frames", dest="T", type=int, default=0)  # ignored; for Makefile uniformity
p.add_argument("--tq", type=int, default=0)
p.add_argument("--kb", type=int, default=0)
p.add_argument("--tactive", type=int, default=0)
opts, _ = p.parse_known_args(sys.argv[1:])
dev = NPU2() if opts.device == "npu2" else NPU1()
print(my_probe(dev))
