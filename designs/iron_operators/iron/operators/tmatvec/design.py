# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker

"""
Transposed-A matvec: the reduction runs DOWN the rows of a row-major matrix.

    C[b][j] = sum over p of  W[b][p] * A[b // batch_group][p][j]

gemv's contraction is the other one -- one output per row, reducing ALONG a row -- and attention's
context step wants this one: out[d] = sum_p softmax[p] * V[p][d] with V stored [S][head_dim]. Doing
it as a dot product is what forces a physical transpose of the whole cache first.

THE MAPPING IS THE POINT, and it is why `n_matrices == cols` is required rather than convenient.
Core c owns matrix c and EVERY batch that reads it, so it streams that matrix ONCE and applies each
group member's own W vector out of L1. The alternative (splitting the output width across columns)
reads the same matrix batch_group times AND makes each row-run M/cols elements wide -- 32 B at
head_dim 128 over 8 columns, well under the ~128 B contiguity knee measured for this fabric, where
whole rows are 256 B and past it.

 - cols: AIE columns; must equal num_batches // batch_group (one matrix per column)
 - M: output width == the matrix's row width (head_dim)
 - K: reduction extent == the matrix's row COUNT (sequence length)
 - rows_per_chunk: rows of A streamed per kernel call; sets the L1 A tile (rows_per_chunk*M elems)
 - batch_group: batches sharing one matrix (GQA: query heads per kv head)
 - l1_bytes: core-tile local memory to size against; None = the AIE2P 64 KB
"""


# AIE2P core-tile local memory. Stated, not derived: the Python bindings expose no accessor
# (AIETargetModel::getLocalMemorySize() is C++ only). Callers may override per target.
AIE2P_L1_BYTES = 65536
# The core's stack and locals. The stack alone defaults to 0x400, and this tree has twice paid for
# a frame that silently overwrote the objectFIFO buffers placed above it.
L1_HEADROOM_BYTES = 4096


def l1_footprint_bytes(M, K, batch_group, rows_per_chunk):
    """Bytes this design places in one core's L1, by term.

    Only the A term scales with rows_per_chunk, which is why that is the knob the error names.
    """
    return (
        2 * rows_per_chunk * M * 2      # A objectfifo, depth 2, bf16
        + batch_group * K * 2           # W objectfifo, depth 1, bf16
        + 2 * batch_group * M * 2       # C objectfifo, depth 2, bf16
        + batch_group * M * 4           # the f32 accumulator Buffer
    )


def largest_fitting_rows_per_chunk(M, K, batch_group, l1_bytes=None):
    """The largest legal rows_per_chunk that FITS, or 0 if no chunking makes this shape fit."""
    budget = AIE2P_L1_BYTES if l1_bytes is None else l1_bytes
    ok = [r for r in (1, 2, 4, 8, 16, 32, 64, 128, 256)
          if r <= K and K % r == 0
          and l1_footprint_bytes(M, K, batch_group, r) + L1_HEADROOM_BYTES <= budget]
    return max(ok) if ok else 0


def check_l1_fits(M, K, batch_group, rows_per_chunk, l1_bytes=None):
    """Raise if the tiling does not FIT. Returns the message, so callers pick the exception type.

    KERNEL-CONTRACT K008: the tiling must FIT, not merely divide -- and nothing downstream checks
    it. rows_per_chunk sets the A tile at rows_per_chunk*M, so the footprint scales with head_dim:
    the default 64 fits at M=128 (42.0 KB) and does not at M=256 (88.0 KB), where aiecc reports
    "'aie.tile' op Basic sequential allocation also failed" -- naming a TILE and not a SIZE, so it
    reads as a placement bug rather than "your chunk is too big". MEASURED 2026-09-07 bringing up
    Gemma-3-270M (M=256, batch_group=4): the default fails to build and 32 succeeds, which this
    arithmetic reproduces exactly.
    """
    budget = AIE2P_L1_BYTES if l1_bytes is None else l1_bytes
    used = l1_footprint_bytes(M, K, batch_group, rows_per_chunk)
    if used + L1_HEADROOM_BYTES <= budget:
        return None
    fits = largest_fitting_rows_per_chunk(M, K, batch_group, l1_bytes)
    a, w = 2 * rows_per_chunk * M * 2, batch_group * K * 2
    c, acc = 2 * batch_group * M * 2, batch_group * M * 4
    # Report the BREAKDOWN, not just the total. Only A scales with rows_per_chunk, so when a shape
    # cannot fit at any chunk size the blocker is one of the other three and shrinking the chunk can
    # never help. MEASURED 2026-09-07 on Gemma-4-12B's global layers (head_dim 512, one kv head,
    # gqa_group 16): W alone is 16*2048*2 = 65536 B, the ENTIRE L1, before A, C or acc get a byte.
    # An earlier version of this message said "needs a MemTile stage for A" in that case, which
    # names the wrong operand and would have sent the reader to tune the one knob that does nothing.
    terms = f"A {a} + W {w} + C {c} + acc {acc}"
    if fits:
        advice = f"largest rows_per_chunk that fits here is {fits}."
    else:
        worst = max((w, "W (batch_group*K)"), (c, "C"), (acc, "acc"), key=lambda t: t[0])
        advice = (
            f"NO rows_per_chunk fits: A is the only term it scales, and the other three already "
            f"total {w + c + acc} B. The dominant one is {worst[1]} at {worst[0]} B -- reduce "
            f"batch_group or K, or stage that operand through a MemTile. Shrinking rows_per_chunk "
            f"cannot help."
        )
    return (
        f"TMatVec does not fit L1: {used} B ({terms}) + {L1_HEADROOM_BYTES} B headroom exceeds "
        f"{budget} B at M={M} K={K} batch_group={batch_group} rows_per_chunk={rows_per_chunk}. "
        + advice
    )


def transposed_matvec(
    dev,
    cols,
    M,
    K,
    num_batches=1,
    batch_group=1,
    rows_per_chunk=64,
    kernel_object="mv_taccum.o",
    func_prefix="",
    verbose=False,
    alloc_K=None,
    l1_bytes=None,
):
    assert num_batches % batch_group == 0, (
        f"num_batches ({num_batches}) must be a multiple of batch_group ({batch_group})"
    )
    n_matrices = num_batches // batch_group
    assert n_matrices == cols, (
        f"this design places one matrix per column: num_batches//batch_group ({n_matrices}) "
        f"must equal cols ({cols})"
    )
    assert K % rows_per_chunk == 0, (
        f"rows_per_chunk ({rows_per_chunk}) must divide K ({K})"
    )
    # Rows ALLOCATED per matrix, when that differs from the rows REDUCED. Same idea as gemv's
    # alloc_M one axis over: the window is a row PREFIX here, so only the per-matrix stride moves.
    assert alloc_K is None or alloc_K >= K, (
        f"alloc_K ({alloc_K}) must be >= K ({K})"
    )
    _AK = K if alloc_K is None else alloc_K

    n_chunks = K // rows_per_chunk

    # K008: the tiling must FIT, not merely divide. op.py raises this at construction; the
    # assert here covers the generator being driven directly.
    _msg = check_l1_fits(M, K, batch_group, rows_per_chunk, l1_bytes)
    assert _msg is None, _msg

    L1_A_ty = np.ndarray[(rows_per_chunk * M,), np.dtype[bfloat16]]
    L1_W_ty = np.ndarray[(batch_group * K,), np.dtype[bfloat16]]
    L1_C_ty = np.ndarray[(batch_group * M,), np.dtype[bfloat16]]
    ACC_ty = np.ndarray[(batch_group * M,), np.dtype[np.float32]]

    L3_A_ty = np.ndarray[(n_matrices * _AK * M,), np.dtype[bfloat16]]
    L3_W_ty = np.ndarray[(num_batches * K,), np.dtype[bfloat16]]
    L3_C_ty = np.ndarray[(num_batches * M,), np.dtype[bfloat16]]

    # The fused dispatch prefixes BOTH the symbol and the object FILENAME with op{idx}_, so the
    # object reference has to carry func_prefix too -- prefixing only the symbol builds an object
    # nothing links against ("cannot open tmv_128n.o").
    obj = f"{func_prefix}{kernel_object}"
    k_zero = Kernel(f"{func_prefix}taccum_zero_f32", obj, [np.int32, ACC_ty])
    k_rows = Kernel(
        f"{func_prefix}taccum_rows_bf16_f32",
        obj,
        [np.int32, np.int32, np.int32, np.int32, L1_A_ty, L1_W_ty, ACC_ty],
    )
    k_finish = Kernel(
        f"{func_prefix}taccum_finish_bf16", obj, [np.int32, ACC_ty, L1_C_ty]
    )

    A_fifos = [ObjectFifo(L1_A_ty, name=f"A_L3L1_{c}", depth=2) for c in range(cols)]
    W_fifos = [ObjectFifo(L1_W_ty, name=f"W_L3L1_{c}", depth=1) for c in range(cols)]
    C_fifos = [ObjectFifo(L1_C_ty, name=f"C_L1L3_{c}", depth=2) for c in range(cols)]

    def core_body(A_cons, W_cons, C_prod, acc, zero, rows, finish):
        for _ in range_(0xFFFFFFFF):
            w = W_cons.acquire(1)
            zero(batch_group, acc)
            for i in range_(n_chunks):
                a = A_cons.acquire(1)
                # w_off advances by whole chunks; the group stride is this batch's own W length.
                rows(rows_per_chunk, batch_group, K, i * rows_per_chunk, a, w, acc)
                A_cons.release(1)
            c = C_prod.acquire(1)
            finish(batch_group, acc, c)
            C_prod.release(1)
            W_cons.release(1)

    # A: column c reads matrix c, whole contiguous rows, chunk by chunk. The per-matrix stride is
    # the ALLOCATION (_AK), not the reduced extent, so a windowed read still lands on the right one.
    # KNOWN DEFECT, diagnosed 2026-09-07, NOT fixed: this single fill of K*M elements lowers to
    # ONE BD against a fifo whose object is rows_per_chunk*M -- 262144 elements streamed through a
    # depth-2 16384-element L1 buffer with no per-object lock to gate it. Correct standalone
    # (30/30 identical dispatches, rel-L2 at the bf16 floor) but it RACES once the design is
    # invoked more than once inside one runtime sequence, which is what 2+ decoder layers are.
    # Neither cheap fix works: one fill per object exceeds the shim's 16 BDs, and bounding them
    # with a task group per chunk deadlocks (ERT_CMD_STATE_TIMEOUT). The real fix is to route A
    # L3->L2->L1 through a MemTile, which has 48 BDs and 512 KB and does object-sized chunking with
    # proper locks -- the pattern strided_copy uses via .forward().
    A_taps = [
        TensorAccessPattern(
            tensor_dims=L3_A_ty.__args__[0],
            offset=c * _AK * M,
            sizes=[1, 1, 1, K * M],
            strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]
    # W: column c takes its group's batches, which are contiguous because batches sharing a matrix
    # are consecutive by construction (batch b reads matrix b // batch_group).
    W_taps = [
        TensorAccessPattern(
            tensor_dims=L3_W_ty.__args__[0],
            offset=c * batch_group * K,
            sizes=[1, 1, 1, batch_group * K],
            strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]
    C_taps = [
        TensorAccessPattern(
            tensor_dims=L3_C_ty.__args__[0],
            offset=c * batch_group * M,
            sizes=[1, 1, 1, batch_group * M],
            strides=[0, 0, 0, 1],
        )
        for c in range(cols)
    ]

    workers = [
        Worker(
            core_body,
            [
                A_fifos[c].cons(),
                W_fifos[c].cons(),
                C_fifos[c].prod(),
                Buffer(type=ACC_ty, name=f"acc_{c}"),
                k_zero,
                k_rows,
                k_finish,
            ],
        )
        for c in range(cols)
    ]

    def sequence(A, W, C, A_prods, W_prods, C_conss):
        # Mirrors gemv's structure, and the shape matters. W is the LONG-LIVED operand -- the core
        # holds it across the whole chunk loop -- so it gets its OWN task group, finished LAST,
        # exactly as gemv does with B. Putting it in the same group as the per-chunk A fills and the
        # C drain, and interleaving the three per column, raced: two back-to-back invocations of
        # this design inside one runtime sequence (which is what 2+ decoder layers are) produced
        # different results run to run.
        tg_w = TaskGroup()
        for c in range(cols):
            W_prods[c].fill(W, W_taps[c], group=tg_w)
        # One group per chunk, finished before the next. A shim tile has 16 BDs, so issuing all
        # n_chunks fills for all columns at once ("Too many simultaneously active buffer
        # descriptors on tile (0,0)") is not an option -- gemv bounds the same way with its
        # per-wait groups.
        tg_ac = TaskGroup()
        for c in range(cols):
            A_prods[c].fill(A, A_taps[c], group=tg_ac)
        for c in range(cols):
            C_conss[c].drain(C, C_taps[c], group=tg_ac, wait=True)
        tg_ac.finish()
        tg_w.finish()

    rt = Runtime(
        sequence,
        [
            L3_A_ty,
            L3_W_ty,
            L3_C_ty,
            [f.prod() for f in A_fifos],
            [f.prod() for f in W_fifos],
            [f.cons() for f in C_fifos],
        ],
    )
    return Program(dev, rt, workers=workers).resolve_program()
