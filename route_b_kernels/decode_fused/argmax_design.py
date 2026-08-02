# SPDX-License-Identifier: Apache-2.0
# Per-column partial argmax design (e2e/NPU lm-head step-2). Splits a [N] bf16 vector across `cols` AIE
# columns; each column scans its contiguous N/cols slice and emits the LOCAL max index (i32) + max value
# (f32). The host does the trivial cols-way reduce (global = col*slice + local; pick the largest value).
# Modeled on iron/operators/channeled_unary_design.py (multi-column, per-column taps), but the outputs are
# tiny scalars (1 per column) instead of a same-size line.
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def my_argmax(dev, N, cols, kernel_object="argmax_slice.o", func_prefix="", verbose=False,
              chunk=None):
    """`chunk` tiles each column's slice so the L1 working set stays bounded.

    Default (chunk=None) keeps the original one-object-per-column behaviour, which is what
    the ASR proj_out path uses (VOCAB_PAD=52224 -> 6528 bf16 = 13 KB, fits).

    At an LLM vocab that does NOT fit: Gemma's 262144 gives a 32768 bf16 slice = 64 KB, i.e.
    the WHOLE L1 of a core, and double-buffered it is 128 KB -- aiecc fails with "allocated
    buffers exceeded available memory". Passing e.g. chunk=4096 streams the slice as 8 tiles
    of 8 KB instead. The core loop already acquires one tile at a time, so nothing about the
    kernel or the core changes; only the fifo type and the taps do. The cost is that each
    column now emits nchunks partials instead of 1, so the host reduces over cols*nchunks
    entries -- still bytes, not megabytes.
    """
    assert N % cols == 0, "N must split evenly across cols"
    slice_n = N // cols
    if chunk is None:
        chunk = slice_n
    assert slice_n % chunk == 0, "each column's slice must split evenly into chunks"
    nchunks = slice_n // chunk
    PACK = 4  # bf16 slots per partial = 8 bytes = [val:f32 | idx:i32] (fusion is uniform-bf16)

    in_line_ty = np.ndarray[(chunk,), np.dtype[bfloat16]]     # one tile of a column's slice
    out_line_ty = np.ndarray[(PACK,), np.dtype[bfloat16]]     # one packed (val, local_idx)
    in_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    out_ty = np.ndarray[(cols * nchunks * PACK,), np.dtype[bfloat16]]

    of_ins = [ObjectFifo(in_line_ty, name=f"amin{i}", depth=2) for i in range(cols)]
    of_out = [ObjectFifo(out_line_ty, name=f"amout{i}", depth=2) for i in range(cols)]

    kernel_fcn = Kernel(
        f"{func_prefix}argmax_slice_bf16",
        f"{func_prefix}{kernel_object}",
        [in_line_ty, out_line_ty, np.int32],
    )

    def core_fn(of_in, of_o, knl):
        for _ in range_(0xFFFFFFFF):
            ei = of_in.acquire(1)
            eo = of_o.acquire(1)
            knl(ei, eo, chunk)
            of_in.release(1)
            of_o.release(1)

    workers = [
        Worker(core_fn, [of_ins[i].cons(), of_out[i].prod(), kernel_fcn])
        for i in range(cols)
    ]

    # Per-column input slice = contiguous [i*slice_n : (i+1)*slice_n], streamed as nchunks tiles.
    in_taps = [TensorAccessPattern((1, N), slice_n * i, [1, 1, nchunks, chunk], [0, 0, chunk, 1])
               for i in range(cols)]
    # Per-column partials gather into out[i*nchunks*PACK : (i+1)*nchunks*PACK].
    out_taps = [TensorAccessPattern((1, cols * nchunks * PACK), PACK * nchunks * i,
                                    [1, 1, nchunks, PACK], [0, 0, PACK, 1]) for i in range(cols)]

    def sequence(A, C, in_prods, out_conses):
        tg = TaskGroup()
        for i in range(cols):
            in_prods[i].fill(A, in_taps[i], group=tg)
        for i in range(cols):
            out_conses[i].drain(C, out_taps[i], wait=True, group=tg)
        tg.finish()

    rt = Runtime(
        sequence,
        [
            in_ty,
            out_ty,
            [of_ins[i].prod() for i in range(cols)],
            [of_out[i].cons() for i in range(cols)],
        ],
    )

    return Program(dev, rt, workers=workers).resolve_program()
