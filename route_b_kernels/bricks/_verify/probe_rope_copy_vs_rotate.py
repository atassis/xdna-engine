#!/usr/bin/env python3
"""Is rope-lut's pinned-aie_api damage the COPY/ROTATE interaction, or the rotate itself?

WHY. Under the pinned aie_api, the pos=0 identity comes back with row 0's lanes reading exactly 0.0.
I first read that as "never written" -- WRONG, and the shim says so: it COPIES qk_in into qk_out for
every element and only then calls rope_lut_prologue, which rotates qk_out IN PLACE. So those lanes
were written by the copy. And at pos=0 the rotation is out1 = x1*cos - x2*sin = x1*1 - x2*0 = x1, so
a zero output means the LOAD of row+i returned zero, not that a store went missing. The disassembly
agrees: both arms emit the same two vst.conv.bf16.fp32 output stores.

That points at the copy loop and the rotate loop -- which both touch qk_out -- not being ordered, so
the rotate's first-iteration load reads qk_out before the copy's store to that region is visible.

TWO SHIMS, one difference:
  A. copy-then-rotate  -- today's shim: qk_out = qk_in, then rotate qk_out in place.
  B. rotate-only       -- the host pre-fills the OUTPUT buffer and the shim rotates it directly,
                          so no kernel store precedes the rotate's loads.

A damaged and B clean  => the fault is the copy/rotate ordering, and the kernel body is innocent.
Both damaged          => the fault is inside the rotate itself; the copy is a red herring.

Run under both arms:  arm3.sh pin probe_rope_copy_vs_rotate.py  /  arm3.sh wheel ...
"""
from pathlib import Path

import ml_dtypes
import numpy as np

import aie.iron as iron
import bricklib
import verify_rope_lut as V

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

M, D, ROT = 2, V.D, V.ROT          # M=2 is the smallest damaged shape under the pin
BF = ml_dtypes.bfloat16
FLAGS = [f"-DROPE_D={D}", f"-DROPE_ROT={ROT}", f"-DROPE_M={M}",
         f"-DROPE_SCALE_INV={V.SCALE_INV:.1f}f"]

g, cc = V.load_golden()
rng = np.random.default_rng(0)
qk = rng.standard_normal((M, D)).astype(np.float32).astype(BF)
qk_ref = qk.astype(np.float32)
inv_freq = g.build_inv_freq(ROT).astype(np.float32)
pos = np.zeros(M, dtype=np.int32)
cbuf = np.concatenate([pos.view(np.int32), inv_freq.view(np.int32)]).astype(np.int32)

ARMS = {
    "A copy-then-rotate": (
        'extern "C" void rope_probe(bfloat16 *qk_in, int32_t *cbuf, bfloat16 *qk_out) {\n'
        '  for (unsigned i = 0; i < (unsigned)(ROPE_M * ROPE_D); ++i) qk_out[i] = qk_in[i];\n'
        '  const int32_t *pos = cbuf;\n'
        '  const float *inv_freq = (const float *)(cbuf + ROPE_M);\n'
        '  rope_lut_prologue(qk_out, pos, inv_freq);\n'
        '}\n'),
    # ARM B IS RETIRED AND MUST NOT COME BACK. It pre-filled the OUTPUT tensor on the host and had
    # the shim rotate it without copying. iron output tensors are NOT uploaded, so the kernel
    # rotated zeros: 0/256 on BOTH arms, including the wheel, which is clean under every other test.
    # That impossible result is the only reason the arm was caught. To get data into a buffer the
    # kernel reads, it must arrive as an INPUT.
    "C rotate-input-first": (
        'extern "C" void rope_probe(bfloat16 *qk_in, int32_t *cbuf, bfloat16 *qk_out) {\n'
        '  const int32_t *pos = cbuf;\n'
        '  const float *inv_freq = (const float *)(cbuf + ROPE_M);\n'
        '  // Rotate the INPUT buffer in place -- the host already uploaded real data there, so no\n'
        '  // kernel store precedes the rotate\'s loads. Copy out only afterwards.\n'
        '  rope_lut_prologue(qk_in, pos, inv_freq);\n'
        '  for (unsigned i = 0; i < (unsigned)(ROPE_M * ROPE_D); ++i) qk_out[i] = qk_in[i];\n'
        '}\n'),
}

for label, body in ARMS.items():
    shim = GEN / f"rope_cvr_{label[0]}.cc"
    shim.write_text(f'#include <stdint.h>\n#include "{cc}"\n{body}')
    design = bricklib._build_oneshot("rope_probe", shim, [M * D, cbuf.size], M * D,
                                     [BF, np.int32], BF, FLAGS)
    i0 = iron.tensor(np.ascontiguousarray(qk.reshape(-1)), dtype=BF, device="npu")
    i1 = iron.tensor(np.ascontiguousarray(cbuf), dtype=np.int32, device="npu")
    o = iron.zeros((M * D,), dtype=BF, device="npu")
    design(i0, i1, o)
    got = np.asarray(np.array(o.numpy(), copy=True), np.float32).reshape(M, D)

    exact = int((got == qk_ref).sum())
    bad_rows = sorted(set(np.nonzero(got != qk_ref)[0].tolist()))
    zero_lanes = int(((got == 0.0) & (qk_ref != 0.0)).sum())
    print(f"{label:22s} exact {exact:5d}/{got.size:5d}  damaged rows {bad_rows}  "
          f"zero-lanes {zero_lanes}", flush=True)
