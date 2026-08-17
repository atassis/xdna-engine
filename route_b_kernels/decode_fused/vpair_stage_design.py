# SPDX-License-Identifier: Apache-2.0
# Single-column parity-staging design: merge [N] bf16 into the parity slot of [N,2], keeping the
# other slot. Parity arrives per dispatch as a core-kind scratchpad parameter, the same delivery
# the self-softmax mask width uses.
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, ScratchpadParameter, TaskGroup, Worker
from aie.iron.runtime import sync_parameters
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def vpair_stage(
    dev,
    N,
    parity_parameter,
    kernel_object="vpair_stage.o",
    func_prefix="",
    pair_sizes=None,
    pair_strides=None,
    pair_buffer_size=None,
    pair_offset_parameter=None,
):
    # DIRECT mode: the pair is a strided window INTO a larger buffer (the transposed vcache) rather
    # than its own arena buffer, so the stage's fill/drain do the cache read/write themselves and the
    # follow-on StridedCopy disappears. pair_offset_parameter is an addr-kind scratchpad parameter
    # added onto the BD base per dispatch -- the same delivery StridedCopy's output_offset_parameter
    # uses, because both are just ObjectFifoHandle.fill/drain(tap=..., offset_parameter=...).
    direct = pair_sizes is not None
    v_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    pair_ty = np.ndarray[(2 * N,), np.dtype[bfloat16]]
    buf_ty = np.ndarray[(int(pair_buffer_size),), np.dtype[bfloat16]] if direct else pair_ty

    of_v = ObjectFifo(v_ty, name="vpsv", depth=2)
    of_pin = ObjectFifo(pair_ty, name="vpspi", depth=2)
    of_pout = ObjectFifo(pair_ty, name="vpspo", depth=2)

    kernel_fcn = Kernel(
        f"{func_prefix}vpair_stage_bf16",
        f"{func_prefix}{kernel_object}",
        [v_ty, pair_ty, pair_ty, np.int32, np.int32],
    )
    parity_param = ScratchpadParameter(parity_parameter, np.int32)
    # ONE ScratchpadParameter shared by the fill and the drain. StridedCopy builds two because its
    # in/out offsets are separate symbols; here both transfers address the same window, and a second
    # object on the same name fails codegen with `redefinition of symbol named 'vcache_off'`.
    pair_off_param = (
        ScratchpadParameter(pair_offset_parameter, np.int32)
        if direct and pair_offset_parameter is not None
        else None
    )

    def core_fn(of_vi, of_pi, of_po, knl, parity_src):
        parity = parity_src.read()
        for _ in range_(0xFFFFFFFF):
            ev = of_vi.acquire(1)
            ei = of_pi.acquire(1)
            eo = of_po.acquire(1)
            knl(ev, ei, eo, N, parity)
            of_vi.release(1)
            of_pi.release(1)
            of_po.release(1)

    worker = Worker(
        core_fn, [of_v.cons(), of_pin.cons(), of_pout.prod(), kernel_fcn, parity_param]
    )

    v_tap = TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1])
    if direct:
        # pad to 4D: dropping leading dims leaves BD registers uninitialised (same rule as StridedCopy).
        sizes = [1] * (4 - len(pair_sizes)) + list(pair_sizes)
        strides = [0] * (4 - len(pair_strides)) + list(pair_strides)
        pair_tap = TensorAccessPattern((1, int(pair_buffer_size)), 0, sizes, strides)
    else:
        pair_tap = TensorAccessPattern((1, 2 * N), 0, [1, 1, 1, 2 * N], [0, 0, 0, 1])

    def sequence(V, PIN, POUT, v_h, pin_h, pout_h):
        if pair_off_param is not None:
            sync_parameters()
        tg = TaskGroup()
        v_h.fill(V, v_tap, group=tg)
        pin_h.fill(PIN, pair_tap, group=tg, offset_parameter=pair_off_param)
        pout_h.drain(POUT, pair_tap, wait=True, group=tg, offset_parameter=pair_off_param)
        tg.finish()

    rt = Runtime(
        sequence,
        [v_ty, buf_ty, buf_ty, of_v.prod(), of_pin.prod(), of_pout.cons()],
    )

    return Program(dev, rt, workers=[worker]).resolve_program()
