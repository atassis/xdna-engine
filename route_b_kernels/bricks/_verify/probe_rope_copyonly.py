#!/usr/bin/env python3
"""Is the harness shim's qk_in -> qk_out copy itself lossless?

Everything upstream of the apply loop is now cleared: the gather is correct (probe_rope_sincos,
sin==0/cos==1 at pos=0) and the local sin_buf/cos_buf round trip is clean on every row
(probe_rope_bufroundtrip). Yet probe_rope_identity shows rows 0-1 of qk corrupted at pos=0, where
the rotation is the exact identity.

That leaves the qk buffer itself, and the first thing that touches it is the harness shim's copy:

    for (unsigned i = 0; i < ROPE_M*ROPE_D; ++i) qk_out[i] = qk_in[i];

a SCALAR element-by-element bf16 loop over 2048 elements. rope_lut.cc's own comment records that the
aie2p scalar-f32 path miscompiles ("scalar (float)pos[m] / scalar float mul collapse to a constant on
device"), so a scalar bf16 copy is not obviously safe either.

This runs the copy AND NOTHING ELSE. Output must equal input bit-for-bit. If rows 0-1 differ here,
the defect is the harness, not the kernel -- and the brick has been failing on its test rig.
"""
import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot

M, D = 16, 128
N = M * D

SHIM = f"""#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void copy_only(bfloat16 *restrict qk_in, bfloat16 *restrict qk_out) {{
  for (unsigned i = 0; i < {N}u; ++i) qk_out[i] = qk_in[i];
}}
"""


def main():
    p = GEN / "copy_only_shim.cc"
    p.write_text(SHIM)
    design = _build_oneshot(
        "copy_only", p, [N], N, [ml_dtypes.bfloat16], ml_dtypes.bfloat16, []
    )
    rng = np.random.default_rng(0)
    src = rng.standard_normal(N).astype(np.float32).astype(ml_dtypes.bfloat16)
    it = iron.tensor(np.ascontiguousarray(src), dtype=ml_dtypes.bfloat16, device="npu")
    ot = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
    design(it, ot)

    got = ot.numpy().astype(np.float32).reshape(M, D)
    want = src.astype(np.float32).reshape(M, D)
    bad = got != want
    print(f"elements exactly equal: {N - int(bad.sum())}/{N}")
    rows = sorted(set(np.nonzero(bad)[0].tolist()))
    print(f"rows with any mismatch: {rows if rows else 'none'}")
    if rows:
        r = rows[0]
        cols = np.nonzero(bad[r])[0]
        print(f"  row {r}: {cols.size} bad cols, first {cols[:12].tolist()}")
        print(f"  want[:8] {want[r][:8].tolist()}")
        print(f"  got [:8] {got[r][:8].tolist()}")


if __name__ == "__main__":
    main()
