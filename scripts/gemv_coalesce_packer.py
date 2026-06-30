#!/usr/bin/env python3
"""Canonical batch-coalesce BD packer for the IRON GEMV (single source of truth).

Computes ONE TensorAccessPattern (offset, sizes, strides) per AIE column that covers
all `num_batches` of a GEMV per-column DRAM transfer, replacing the stock per-batch
unrolled BDs. The same logic is mirrored verbatim in iron/operators/gemv/design.py;
this module is the offline-tested spec.

AIE2p shim BD caps (NpuWriteBdOp::verify, mlir-aie AIEXDialect.cpp):
  D0 size <=1023, D1 size <=1023, D2 size UNCAPPED, iteration size <=63;
  all strides <= 2**20-1. TAP sizes/strides are OUTERMOST-first; the verifier reverses
  to innermost-first, so sizes=[s0,s1,s2,s3] maps s3->D0, s2->D1, s1->D2, s0->iter.
  The coalesced layout places num_batches in D2 (uncapped) and splits the contiguous
  run across D0,D1:  sizes=[1, num_batches, run_hi, run_lo]  strides=[0, bstride, run_lo, 1].
"""
import itertools

MAX_WRAP = 1023
MAX_STRIDE = (1 << 20) - 1
# 4-byte shim address granularity / 2-byte bf16 element: the innermost size (run_lo)
# and every stride must be even (AIEXDialect.cpp verifyStridesWraps, not skipped even
# for linear transfers).
GRAN_ELEMS = 2


def split_run(run, lim=MAX_WRAP, gran=GRAN_ELEMS):
    """(run_hi, run_lo): run_hi*run_lo == run, both <= lim, run_lo a multiple of gran
    (granularity-aligned inner size) and maximal. None if no such split exists."""
    lo_start = (lim // gran) * gran
    for lo in range(lo_start, 0, -gran):
        if run % lo == 0 and (run // lo) <= lim:
            return (run // lo, lo)
    return None


def can_coalesce(run, bstride, num_batches):
    """True iff this config can use the single coalesced BD (else caller unrolls)."""
    return (
        num_batches > 1
        and bstride <= MAX_STRIDE
        and bstride % GRAN_ELEMS == 0
        and split_run(run) is not None
    )


def coalesced_tap(col_off, run, bstride, num_batches):
    """(offset, sizes, strides) for the single coalesced per-column BD.
    Precondition: can_coalesce(run, bstride, num_batches)."""
    run_hi, run_lo = split_run(run)
    return col_off, [1, num_batches, run_hi, run_lo], [0, bstride, run_lo, 1]


def unrolled_taps(col_off, run, bstride, num_batches):
    """The stock per-batch BDs (the fallback + the num_batches==1 path)."""
    return [
        (col_off + b * bstride, [1, 1, 1, run], [0, 0, 0, 1])
        for b in range(num_batches)
    ]


def enum_tap(offset, sizes, strides, total):
    """Linear DRAM indices a TAP touches, row-major (last dim fastest), mod total."""
    out = []
    for idx in itertools.product(*[range(s) for s in sizes]):
        out.append((offset + sum(i * st for i, st in zip(idx, strides))) % total)
    return out


def verifier_legal(sizes, strides):
    """Cheap structural check of the lowered-BD caps for a coalesced TAP (outermost-first
    sizes=[iter, D2, D1, D0]). Does NOT enumerate; used for large configs (B=128)."""
    assert len(sizes) == 4 and len(strides) == 4
    s_iter, s_d2, s_d1, s_d0 = sizes
    st_iter, st_d2, st_d1, st_d0 = strides
    if s_iter > 63:
        return False, f"iteration size {s_iter} > 63"
    if s_d1 > MAX_WRAP:
        return False, f"D1 size {s_d1} > {MAX_WRAP}"
    if s_d0 > MAX_WRAP:
        return False, f"D0 size {s_d0} > {MAX_WRAP}"
    for nm, st in [("iter", st_iter), ("D2", st_d2), ("D1", st_d1), ("D0", st_d0)]:
        if st > MAX_STRIDE:
            return False, f"{nm} stride {st} > {MAX_STRIDE}"
    return True, "ok"
