# relpos_mha/relpos_ac_scores_softmax_iron.py -*- Python -*-
#
# STEP-2 COMPOSED IRON design for the Parakeet rel-pos MHA resident block: the
# on-chip AC score matmul feeding the scores->softmax brick, with the f32 score
# tile staying RESIDENT in L1 between the matmul and the softmax (never round-
# tripping to host). This is the first real test of the resident-block thesis.
#
# It drives ONE compute core with relpos_ac_scores_softmax_bake
# (route_b_kernels/relpos_mha/relpos_mha.cc): given packed qk[2T,DK] bf16
# (qu = qk[0:T], k = qk[T:2T]) and host-precomputed BD[T,P] f32 (P = 2T-1), it
# returns probs[T,T] bf16 =
#     softmax_over_keys( rel_shift(BD) + (qu @ k^T) , scaled by 1/sqrt(DK) )
# with the AC = qu @ k^T matmul done ON DEVICE and its f32 output kept in L1.
#
# 3-buffer ABI (qk in / BD in / probs out) -- mirrors step 1 (which was ac in /
# BD in / probs out) but replaces the host AC buffer with the packed qu+k inputs
# that the on-chip matmul consumes. qu and k are PACKED into ONE ObjectFifo so the
# core stays within the NPU2 compute tile's 2 input-DMA-channel budget (qk + BD),
# the same 2-input discipline the whole_array modal design uses. One tile == the
# whole tensor (single core, no tiling). T is baked at build (must match the
# kernel's -DRELPOS_T). Single-tile so qk + BD + the resident AC + probs fit L1:
# T=32 (block 0). Larger T needs the row-tiled resident block, not this kernel.
#
# Bare Program(dev, rt).resolve_program() -- PLACE-TILES model, NO SequentialPlacer
# (that is the stale wheel API; the current fork instance has no Python placer).
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ml_dtypes import bfloat16
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1, NPU2

# head_dim baked into the kernel (RELPOS_DK, Parakeet = 128). Must match -DRELPOS_DK.
DK = 128


def my_relpos_ac_scores_softmax(dev, T):
    P = 2 * T - 1

    # Tensor / tile types. One tile == the whole tensor (single compute core).
    # qk packs qu[T,DK] then k[T,DK] contiguously (bf16) -> 2*T*DK elements.
    qk_ty = np.ndarray[(2 * T * DK,), np.dtype[bfloat16]]
    bd_ty = np.ndarray[(T * P,), np.dtype[np.float32]]
    probs_ty = np.ndarray[(T * T,), np.dtype[bfloat16]]

    of_qk = ObjectFifo(qk_ty, name="qk")
    of_bd = ObjectFifo(bd_ty, name="bd")
    of_probs = ObjectFifo(probs_ty, name="probs")

    # Zero-scalar-arg kernel: T, P, DK and inv_scale are baked into the .cc wrapper.
    relpos = Kernel(
        "relpos_ac_scores_softmax_bake", "kernels.a", [qk_ty, bd_ty, probs_ty]
    )

    def core_body(qk_in, bd_in, probs_out, relpos_fn):
        eqk = qk_in.acquire(1)
        eb = bd_in.acquire(1)
        eo = probs_out.acquire(1)
        relpos_fn(eqk, eb, eo)
        qk_in.release(1)
        bd_in.release(1)
        probs_out.release(1)

    worker = Worker(
        core_body,
        [of_qk.cons(), of_bd.cons(), of_probs.prod(), relpos],
    )

    rt = Runtime()
    with rt.sequence(qk_ty, bd_ty, probs_ty) as (QK, BD, PR):
        rt.start(worker)
        rt.fill(of_qk.prod(), QK)
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

print(my_relpos_ac_scores_softmax(dev, int(opts.T)))
