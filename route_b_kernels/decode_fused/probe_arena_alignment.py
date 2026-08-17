# SPDX-License-Identifier: Apache-2.0
"""Can padding the fused arena satisfy the 64-byte coherence granule? For which layouts?

`_calculate_buffer_layout` packs sub-buffers back-to-back (`offset += length`) while pin #3430's
`NpuTensor.subview()` refuses a view that starts off-granule, OR whose length is not a whole number
of granules unless it ends where the parent ends. The recorded plan was to pad the layout. This
tests whether that is sufficient, on the real designs plus synthetic layouts, CPU-only.

The rule under test, applied to a view of a buffer's LOGICAL length (the compile path turns that
same length into the memref the kernel is handed, so it is not ours to inflate):

    legal  <=>  offset % G == 0  AND  (length % G == 0  OR  offset + length == parent_size)

Call a buffer whose length is not a multiple of G "ragged". Padding fixes offsets; it cannot fix a
ragged buffer's length. A ragged buffer is legal only as the last one in its arena -- so an arena
holding two or more of them cannot be made legal by any layout at all.

Run:  python probe_arena_alignment.py
"""
import numpy as np
import ml_dtypes

import newstack_compat  # noqa: F401 -- MUST precede iron imports
from iron.common import AIEContext
from elf_dispatch_compat import OperatorSequence
from iron.operators.elementwise_mul.op import ElementwiseMul
from iron.operators.strided_copy.op import StridedCopy
from identity_op import Identity
from vpair_stage_op import VPairStage

G = 64  # COHERENCE_GRANULE as detected on this box; see coherence._detect_coherence_granule
BF16 = ml_dtypes.bfloat16


def align_up(v, a=G):
    return -(-v // a) * a


def legal(offset, length, parent):
    return offset % G == 0 and (length % G == 0 or offset + length == parent)


def lay_out(lengths, pad):
    """(offsets, arena_size) for buffers of `lengths`, packed or granule-padded.

    The arena ends exactly where its last buffer does: padding the tail would cost
    the last buffer its ends-at-parent-end exemption, which is the only thing that
    lets a ragged length be legal at all.
    """
    offsets, cursor = [], 0
    for i, n in enumerate(lengths):
        offsets.append(cursor)
        cursor += n if i == len(lengths) - 1 or not pad else align_up(n)
    return offsets, cursor


def verdict(lengths):
    """Is this arena legal packed / padded / padded-with-the-ragged-buffer-last?"""
    ragged = [n for n in lengths if n % G]
    out = {}
    for name, pad in (("packed", False), ("padded", True)):
        offsets, total = lay_out(lengths, pad)
        out[name] = all(legal(o, n, total) for o, n in zip(offsets, lengths))
    # Best case reachable by reordering: every ragged buffer but one must vanish.
    order = [n for n in lengths if n % G == 0] + ragged
    offsets, total = lay_out(order, True)
    out["reordered"] = all(legal(o, n, total) for o, n in zip(offsets, order))
    out["ragged"] = len(ragged)
    return out


def arenas_of(op):
    """buffer lengths grouped by arena, from a built layout."""
    totals = dict(zip(("input", "output", "scratch"), op.buffer_sizes))
    grouped = {k: [] for k in totals}
    for _buf, (btype, _off, length) in sorted(op.subbuffer_layout.items()):
        grouped[btype].append(length)
    return {k: v for k, v in grouped.items() if v}


def report(name, op):
    op.subbuffer_layout, op.buffer_sizes, op.slice_info = op._calculate_buffer_layout()
    print(f"\n{name}")
    unfixable = 0
    for arena, lengths in arenas_of(op).items():
        v = verdict(lengths)
        unfixable += v["ragged"] > 1
        print(f"  {arena:<8} lengths={lengths} ragged={v['ragged']}  "
              f"packed={'OK' if v['packed'] else 'trips':<5} "
              f"padded={'OK' if v['padded'] else 'trips':<5} "
              f"reordered={'OK' if v['reordered'] else 'UNFIXABLE'}")
    return unfixable


def main():
    ctx = AIEContext()
    unfixable = 0

    N, TILE = 4096, 1024
    ident = Identity(N=N, tile=TILE, context=ctx)
    unfixable += report("probe_fusion_roundtrip (identity)", OperatorSequence(
        "fusion_roundtrip_probe", [(ident, "src", "dst")],
        input_args=["src"], output_args=["dst"],
        buffer_sizes={"src": N * 2, "dst": N * 2}, context=ctx))

    emul = ElementwiseMul(size=N, tile_size=N // 8, num_aie_columns=8, context=ctx)
    unfixable += report("probe_fusion_roundtrip (emul, 2 inputs)", OperatorSequence(
        "fusion_roundtrip_emul", [(emul, "src", "ones", "dst")],
        input_args=["src", "ones"], output_args=["dst"],
        buffer_sizes={"src": N * 2, "ones": N * 2, "dst": N * 2}, context=ctx))

    H, HD, S = 2, 4, 8
    NV = H * HD
    stage = VPairStage(N=NV, parity_parameter="par", context=ctx)
    pair_write = StridedCopy(
        input_sizes=(H, HD, 2), input_strides=(HD * 2, 2, 1), input_offset=0,
        output_sizes=(H, HD, 2), output_strides=(HD * S, S, 1), output_offset=0,
        input_buffer_size=NV * 2, output_buffer_size=H * HD * S,
        num_aie_channels=1, output_offset_parameter="vc_off", context=ctx,
    )
    unfixable += report("probe_vpair_stage (measured green on device)", OperatorSequence(
        "vpairprobe", [(stage, "v", "pair", "pair"), (pair_write, "pair", "vcache")],
        input_args=["v"], output_args=["vcache"], context=ctx))

    # The shape the next step builds: vpair_stage's sub-granule buffers joining a shared
    # arena, which is what folding the stage into op_qkv's epilogue does.
    print("\nsynthetic: sub-granule buffers sharing one arena")
    for lengths in ([16, 32], [16, 128], [96, 96], [8192, 16], [64, 128, 16]):
        v = verdict(lengths)
        print(f"  lengths={str(lengths):<20} ragged={v['ragged']}  "
              f"packed={'OK' if v['packed'] else 'trips':<5} "
              f"padded={'OK' if v['padded'] else 'trips':<5} "
              f"reordered={'OK' if v['reordered'] else 'UNFIXABLE'}")
        unfixable += v["ragged"] > 1

    print(f"\narenas no layout can make legal (>=2 ragged buffers): {unfixable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
