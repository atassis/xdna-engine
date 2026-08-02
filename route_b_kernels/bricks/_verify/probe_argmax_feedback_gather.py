#!/usr/bin/env python3
"""Prove the decode feedback mechanism on-chip: argmax -> device-computed scalar index ->
resident-embedding row read, inside ONE dispatch, with nothing crossing to host in between.

WHY THIS SHAPE, and why it is NOT a parallel_lookup gather. The task frames the feedback as
`gather embedding[argmax]` and points at `aie::parallel_lookup`. That primitive gathers ONE bf16 per
key LANE from a 256-entry table -- it is the right brick when the keys VARY per lane (rope-lut's
sin/cos). Decode feeds back ONE token per step, so `embedding[idx]` is a scalar-offset CONTIGUOUS row
of width D, not a per-lane gather. Using parallel_lookup here would need D fetches (one per embedding
dimension, each wasting 15 of 16 lanes) or a table layout that caps the vocabulary at 256/D.

So the mechanism under test is a scalar-indexed load, and the real risk is not the gather at all --
it is whether a DEVICE-COMPUTED scalar index can address a resident buffer in the same dispatch that
produced it. That is the one thing a host-driven loop cannot prove for us, and it is what this probe
isolates.

WHAT WOULD MAKE THIS PASS VACUOUS, and how each is closed:
  * a constant index -- logits are randomised per case so the argmax lands on a different row each
    time, and the expected row differs from every other row (see the table construction);
  * an off-by-one that a ramp table hides -- the embedding table is NOT a ramp in the row index;
    row r holds a value pattern keyed to r so a neighbouring row cannot be mistaken for it;
  * the host secretly doing the argmax -- only logits and the table are inputs; the index is never
    an input and never leaves the core.

Run from the repo root, single-tenant (npu-vox stopped), under scripts/npu_lock.sh.
"""
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, "route_b_kernels/bricks/_verify")
from bricklib import GEN, iron, _build_oneshot  # noqa: E402

V = 64      # vocabulary rows (small on purpose: this proves the mechanism, not the scale)
D = 32      # embedding width

SHIM = r"""
#include <aie_api/aie.hpp>
#include <stdint.h>

// logits[V] f32 -> argmax on core -> row `idx` of emb[V*D] bf16 -> out[D] bf16.
// The index is computed here and used here; it is never an input and never leaves.
extern "C" void argmax_feedback(float *restrict logits, bfloat16 *restrict emb,
                                bfloat16 *restrict out) {
  // Vector argmax over V f32 lanes. aie::max_cmp co-produces the running max and its index,
  // which is why the reduction does not need a second pass to recover the position.
  const unsigned W = 16;
  ::aie::vector<float, 16> vbest = ::aie::load_v<16>(logits);
  ::aie::vector<int32_t, 16> vidx = ::aie::broadcast<int32_t, 16>(0);
  for (unsigned i = W; i < %(V)d; i += W) {
    ::aie::vector<float, 16> v = ::aie::load_v<16>(logits + i);
    ::aie::vector<int32_t, 16> ix = ::aie::broadcast<int32_t, 16>((int32_t)i);
    ::aie::mask<16> gt = ::aie::gt(v, vbest);
    vbest = ::aie::select(vbest, v, gt);
    vidx = ::aie::select(vidx, ix, gt);
  }
  // Horizontal reduce of the 16 lane-champions to one (value, lane) pair.
  float best = vbest.get(0);
  int32_t idx = vidx.get(0);
  for (unsigned l = 1; l < 16; ++l) {
    float c = vbest.get(l);
    if (c > best) { best = c; idx = vidx.get(l) + (int32_t)l; }
  }
  // lane 0's champion carries lane offset 0; the others need their lane added (done above).
  if (idx < 0) idx = 0;
  if (idx >= %(V)d) idx = %(V)d - 1;

  // THE MECHANISM: a device-computed scalar index addressing a resident buffer.
  const bfloat16 *restrict row = emb + (unsigned)idx * %(D)d;
  for (unsigned j = 0; j < %(D)d; j += 16)
    ::aie::store_v(out + j, ::aie::load_v<16>(row + j));
}
""" % {"V": V, "D": D}


def emb_table():
    """Row r is distinguishable from every other row, and is NOT r broadcast (a ramp would let an
    off-by-one in the row index masquerade as a correct read of a neighbour)."""
    t = np.zeros((V, D), dtype=np.float32)
    for r in range(V):
        # two independent facts per row: a row-specific base and a row-specific stride
        t[r] = (1 + r) * 3.0 + np.arange(D, dtype=np.float32) * (1 + (r % 7))
    return t.astype(ml_dtypes.bfloat16)


def main():
    rng = np.random.default_rng(20260802)
    emb = emb_table()
    shim = GEN / "argmax_feedback_shim.cc"
    shim.write_text(SHIM)
    design = _build_oneshot("argmax_feedback", shim, [V, V * D], D,
                            [np.float32, ml_dtypes.bfloat16], ml_dtypes.bfloat16, [])
    embt = iron.tensor(np.ascontiguousarray(emb.reshape(-1)),
                       dtype=ml_dtypes.bfloat16, device="npu")
    fails, seen = 0, []
    for case in range(8):
        logits = rng.standard_normal(V).astype(np.float32)
        # force an unambiguous, case-dependent winner
        want = int(rng.integers(0, V))
        logits[want] = 10.0 + case
        lt = iron.tensor(np.ascontiguousarray(logits), dtype=np.float32, device="npu")
        ot = iron.zeros((D,), dtype=ml_dtypes.bfloat16, device="npu")
        design(lt, embt, ot)
        exp = emb[want].astype(np.float32)
        gotf = ot.numpy().astype(np.float32)
        ok = np.array_equal(gotf, exp)
        # which row did it ACTUALLY read? decodes the fault instead of just failing.
        match = [r for r in range(V) if np.array_equal(gotf, emb[r].astype(np.float32))]
        seen.append(want)
        print(f"  case {case}: want row {want:3d}  ok={ok}  "
              f"read_row={match if match else 'NONE (not any table row)'}")
        if not ok:
            fails += 1
    distinct = len(set(seen))
    print(f"\n{8 - fails}/8 cases exact; {distinct} distinct rows exercised "
          f"(a constant-index bug cannot pass more than 1)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
