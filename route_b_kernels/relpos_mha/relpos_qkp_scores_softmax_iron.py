# relpos_qkp_scores_softmax_iron.py -- STEP 3 of the Parakeet resident MHA block.
#
# Both score matmuls resident: given packed qk[2T,DK] (qu=qk[0:T], k=qk[T:2T]) and
# packed qvp[(T+P),DK] (qv=qvp[0:T], p=qvp[T:T+P]), all bf16, the core computes
#   AC = qu @ k^T   [T,T]   (on chip, resident L1)
#   BD = qv @ p^T   [T,P]   (on chip, resident L1)
# then returns probs[T,T] bf16 = softmax_over_keys( rel_shift(BD) + AC , /sqrt(DK) ).
# NO host score buffer -- step 2 host-fed BD; step 3 computes it on chip too.
#
# 3-buffer ABI (qk in / qvp in / probs out): TWO packed bf16 inputs keep the core
# within the NPU2 compute tile's 2 input-DMA-channel budget. The pos_bias adds
# (qu = q+u, qv = q+v) are folded host-side into the packed buffers. T is baked
# (must match the kernel's -DRELPOS_T). Single compute core, single tile (small T).
#
# PLACE-TILES toolchain: bare Program(dev, rt).resolve_program(), NO SequentialPlacer.
import sys
import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

# head_dim baked into the kernel (RELPOS_DK, Parakeet = 128). Must match -DRELPOS_DK.
DK = 128


def my_relpos_qkp_scores_softmax(dev, T):
    P = 2 * T - 1

    # qk packs qu[T,DK] then k[T,DK]; qvp packs qv[T,DK] then p[P,DK] (bf16).
    qk_ty = np.ndarray[(2 * T * DK,), np.dtype[bfloat16]]
    qvp_ty = np.ndarray[((T + P) * DK,), np.dtype[bfloat16]]
    probs_ty = np.ndarray[(T * T,), np.dtype[bfloat16]]

    of_qk = ObjectFifo(qk_ty, name="qk")
    of_qvp = ObjectFifo(qvp_ty, name="qvp")
    of_probs = ObjectFifo(probs_ty, name="probs")

    # Zero-scalar-arg kernel: T, P, DK and inv_scale are baked into the .cc wrapper.
    relpos = Kernel(
        "relpos_qkp_scores_softmax_bake", "kernels.a", [qk_ty, qvp_ty, probs_ty]
    )

    def core_body(qk_in, qvp_in, probs_out, relpos_fn):
        eqk = qk_in.acquire(1)
        eqvp = qvp_in.acquire(1)
        eo = probs_out.acquire(1)
        relpos_fn(eqk, eqvp, eo)
        qk_in.release(1)
        qvp_in.release(1)
        probs_out.release(1)

    worker = Worker(
        core_body,
        [of_qk.cons(), of_qvp.cons(), of_probs.prod(), relpos],
    )

    rt = Runtime()
    with rt.sequence(qk_ty, qvp_ty, probs_ty) as (QK, QVP, PR):
        rt.start(worker)
        rt.fill(of_qk.prod(), QK)
        rt.fill(of_qvp.prod(), QVP)
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
    raise ValueError(f"unknown device {opts.device}")

print(my_relpos_qkp_scores_softmax(dev, opts.T))
