#!/usr/bin/env python3
"""P1 end-to-end: vocab GEMM -> on-core argmax -> resident-embedding row read, ONE dispatch.

Two halves were device-validated separately: `lm-head-argmax` fuses the vocab projection with an
argmax and emits (best_value, best_index), and
[[decode-feedback-is-a-scalar-indexed-row-read-and-it-works-on-device]] showed a device-computed
scalar index can address a resident buffer in the dispatch that produced it. This joins them, which is
what `onchip-decode-feedback-prototype` P1 asks for: logits -> argmax -> next input, nothing crossing
to host in between.

WHAT IS CHECKED, and why it is the right check. The claim under test is the COMPOSITION: does the
index the GEMM+argmax computes on core correctly drive the row read on the same core, same dispatch?
So the probe emits BOTH the index and the gathered row and asserts

    gathered_row == embedding[returned_index]        bit-exact

which needs no knowledge of the brick's on-chip weight tiling. The GEMM's own numerics are gated
separately by the brick's own golden; re-deriving its [K_TILE, N_TILE] layout here would test that
instead of the seam, and get it wrong more easily than the seam itself.

The argmax is ALSO compared to a host reference over the untiled weight, reported as INFORMATIONAL: a
mismatch there means the weight layout I fed differs from the device's expected tiling, which would
say nothing about the seam.

Run from the repo root, single-tenant (npu-vox stopped), under scripts/npu_lock.sh.
"""
import pathlib
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, "route_b_kernels/bricks/_verify")
from bricklib import GEN, iron, _build_oneshot  # noqa: E402

# Sized by TWO hardware limits, both hit while building this probe:
#   * a CoreTile has 2 input / 2 output DMA channels, so hidden+weight are PACKED into one buffer
#     (3 separate inputs + 1 output needs 3 in and the placer rejects it);
#   * a dma_bd carries at most 16383 32-bit words, so the packed buffer must stay under 32766 bf16.
#   * and every objectFIFO here is DOUBLE-buffered, so each operand costs 2x its size in a 64 KB L1 --
#     which is what HID=VOC=128 (34816 B packed, 69632 B doubled) overflowed on.
# HID=VOC=64 gives 4608 bf16 packed (9216 B), so 2x(9216 + 4096 + 136) + stack fits comfortably.
# Both dims must be multiples of kKTile = kNTile = 8.
HID, VOC, MPAD, D = 64, 64, 8, 32

SHIM = r"""
#include <aie_api/aie.hpp>
#include <stdint.h>
#define LMHEAD_HIDDEN %(HID)d
#define LMHEAD_VOCAB  %(VOC)d
#define LMHEAD_M_PAD  %(MPAD)d
#include "%(BRICK)s"

// hw[MPAD*HID + HID*VOC] bf16 (hidden then weight, PACKED), emb[VOC*D] bf16
//   -> out[2 + D] f32 = { best_index, best_value, embedding[best_index][0..D) }
//
// PACKED because a CoreTile has only 2 input / 2 output DMA channels: three separate input buffers
// plus one output needs 3 in and the placer rejects it ("requires 3 input/1 output DMA channels, but
// only 2 input/2 output available"). hidden and weight are both bf16 and both read-once, so
// concatenating them costs nothing and buys the third operand slot.
//
// The index is produced and consumed on core; it is never an input and never read back separately.
extern "C" void lmhead_feedback(bfloat16 *restrict hw, bfloat16 *restrict emb,
                                float *restrict out) {
  const bfloat16 *restrict hidden = hw;
  const bfloat16 *restrict weight = hw + %(MPAD)d * %(HID)d;
  float best_value;
  int32_t best_index;
  lm_head_argmax(hidden, weight, &best_value, &best_index);

  out[0] = (float)best_index;
  out[1] = best_value;

  // THE SEAM: a device-computed scalar index addressing a resident buffer, same dispatch.
  if (best_index < 0) best_index = 0;
  if (best_index >= %(VOC)d) best_index = %(VOC)d - 1;
  const bfloat16 *restrict row = emb + (unsigned)best_index * %(D)d;
  for (unsigned j = 0; j < %(D)d; ++j) out[2 + j] = (float)row[j];
}
""" % {"HID": HID, "VOC": VOC, "MPAD": MPAD, "D": D,
       # absolute: the shim is compiled from ~/.npu/cache, so a repo-relative include cannot resolve
       "BRICK": str(pathlib.Path("route_b_kernels/bricks/lm-head-argmax/lm_head_argmax.cc").resolve())}


def emb_table(rng):
    """Row r carries a row-specific base AND stride, so a neighbouring row cannot masquerade as it."""
    t = np.zeros((VOC, D), dtype=np.float32)
    for r in range(VOC):
        t[r] = (1 + (r % 251)) * 2.0 + np.arange(D, dtype=np.float32) * (1 + (r % 7))
    return t.astype(ml_dtypes.bfloat16)


def main():
    rng = np.random.default_rng(20260802)
    emb = emb_table(rng)
    shim = GEN / "lmhead_feedback_shim.cc"
    shim.write_text(SHIM)
    design = _build_oneshot("lmhead_feedback", shim,
                            [MPAD * HID + HID * VOC, VOC * D], 2 + D,
                            [ml_dtypes.bfloat16] * 2, np.float32, [])
    embt = iron.tensor(np.ascontiguousarray(emb.reshape(-1)),
                       dtype=ml_dtypes.bfloat16, device="npu")

    seam_fail = idx_seen = 0
    for case in range(6):
        hidden = np.zeros((MPAD, HID), dtype=np.float32)
        hidden[0] = rng.standard_normal(HID).astype(np.float32)
        weight = (rng.standard_normal((HID, VOC)).astype(np.float32) * 0.05)
        hw = np.concatenate([hidden.reshape(-1), weight.reshape(-1)]).astype(ml_dtypes.bfloat16)
        hwt = iron.tensor(np.ascontiguousarray(hw), dtype=ml_dtypes.bfloat16, device="npu")
        ot = iron.zeros((2 + D,), dtype=np.float32, device="npu")
        design(hwt, embt, ot)
        o = ot.numpy()

        idx, val, row = int(o[0]), float(o[1]), o[2:]
        exp = emb[idx].astype(np.float32)
        seam_ok = np.array_equal(row, exp)
        seam_fail += (not seam_ok)

        # informational only -- depends on the weight layout matching the device's tiling
        hb = hidden[0].astype(ml_dtypes.bfloat16).astype(np.float64)
        wb = weight.astype(ml_dtypes.bfloat16).astype(np.float64)
        host_idx = int(np.argmax((hb @ wb).astype(np.float32)))
        idx_seen = idx_seen or idx
        print(f"  case {case}: device_idx={idx:4d}  seam={'OK' if seam_ok else 'FAIL'}   "
              f"(host_argmax={host_idx}{'' if host_idx == idx else '  <- layout differs, informational'})")

    print(f"\nSEAM (index -> resident row read, same dispatch): "
          f"{6 - seam_fail}/6 bit-exact")
    return 1 if seam_fail else 0


if __name__ == "__main__":
    sys.exit(main())
