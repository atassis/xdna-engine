# relpos_mha/relpos_scores_softmax_iron.py -*- Python -*-
#
# STEP-1 STANDALONE IRON design for the Parakeet rel-pos MHA scores->softmax
# brick. It drives ONE compute core with the relpos_scores_softmax_bake kernel
# (route_b_kernels/relpos_mha/relpos_mha.cc): given host-precomputed AC[T,T] f32
# and BD[T,P] f32 (P = 2T-1), it returns probs[T,T] bf16 =
#     softmax_over_keys( rel_shift(BD) + AC , scaled by 1/sqrt(DK) )
# with NO matmul on device. This de-risks the two hard rel-pos bricks (the
# zero-arithmetic strided-relayout rel_shift + the vectorized-exp2 softmax) as a
# real dataflow before the full resident block is attempted.
#
# 3-buffer ABI (AC in / BD in / probs out) -- mirrors the proven dwconv1d
# (in / w / out) template, one ObjectFifo per buffer, one tile == the whole
# tensor (single-core, no tiling). T is baked at build (must match the kernel's
# -DRELPOS_T). Single-tile so AC+BD+probs fit L1: T=32 (block 0) uses ~14 KB;
# large T needs the row-tiled resident block, not this de-risk kernel.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2


def my_relpos_scores_softmax(dev, T):
    P = 2 * T - 1

    # Tensor / tile types. One tile == the whole tensor (single compute core).
    ac_ty = np.ndarray[(T * T,), np.dtype[np.float32]]
    bd_ty = np.ndarray[(T * P,), np.dtype[np.float32]]
    probs_ty = np.ndarray[(T * T,), np.dtype[bfloat16]]

    of_ac = ObjectFifo(ac_ty, name="ac")
    of_bd = ObjectFifo(bd_ty, name="bd")
    of_probs = ObjectFifo(probs_ty, name="probs")

    # Zero-scalar-arg kernel: T, P and inv_scale are baked into the .cc wrapper.
    relpos = Kernel(
        "relpos_scores_softmax_bake", "kernels.a", [ac_ty, bd_ty, probs_ty]
    )

    def core_body(ac_in, bd_in, probs_out, relpos_fn):
        ea = ac_in.acquire(1)
        eb = bd_in.acquire(1)
        eo = probs_out.acquire(1)
        relpos_fn(ea, eb, eo)
        ac_in.release(1)
        bd_in.release(1)
        probs_out.release(1)

    worker = Worker(
        core_body,
        [of_ac.cons(), of_bd.cons(), of_probs.prod(), relpos],
    )

    rt = Runtime()
    with rt.sequence(ac_ty, bd_ty, probs_ty) as (AC, BD, PR):
        rt.start(worker)
        rt.fill(of_ac.prod(), AC)
        rt.fill(of_bd.prod(), BD)
        rt.drain(of_probs.cons(), PR, wait=True)

    return Program(dev, rt).resolve_program()


p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="npu or npu2")
p.add_argument("-T", "--frames", required=True, dest="T", type=int,
               help="encoder frame count T (P = 2T-1); must match -DRELPOS_T")
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"[ERROR] Device name {opts.device} is unknown")

print(my_relpos_scores_softmax(dev, int(opts.T)))
