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

CBUF_N = M + 64  # same packed-const shape the rope harness passes: pos[M] | inv_freq[ROT/2]

SHIM_1IN = f"""#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void copy_only(bfloat16 *restrict qk_in, bfloat16 *restrict qk_out) {{
  for (unsigned i = 0; i < {N}u; ++i) qk_out[i] = qk_in[i];
}}
"""

# Same copy, but with the rope harness's exact THREE-buffer signature (qk_in, cbuf, qk_out).
# cbuf is deliberately unread: if only row 0 arrives here, nothing in rope_lut.cc is implicated
# and the fault is the two-input oneshot rail, i.e. the brick has been failing on its test rig.
SHIM_2IN = f"""#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void copy_only2(bfloat16 *restrict qk_in, int32_t *restrict cbuf,
                           bfloat16 *restrict qk_out) {{
  (void)cbuf;
  for (unsigned i = 0; i < {N}u; ++i) qk_out[i] = qk_in[i];
}}
"""


def check(label, got, want):
    bad = got != want
    rows = sorted(set(np.nonzero(bad)[0].tolist()))
    print(f"{label}: exact {N - int(bad.sum())}/{N}   damaged rows {rows if rows else 'NONE'}")
    if rows:
        r = rows[0]
        print(f"    row {r} want[:6] {want[r][:6].tolist()}")
        print(f"    row {r} got [:6] {got[r][:6].tolist()}")


def main():
    rng = np.random.default_rng(0)
    src = rng.standard_normal(N).astype(np.float32).astype(ml_dtypes.bfloat16)
    want = src.astype(np.float32).reshape(M, D)

    p1 = GEN / "copy_only_shim.cc"
    p1.write_text(SHIM_1IN)
    d1 = _build_oneshot("copy_only", p1, [N], N, [ml_dtypes.bfloat16], ml_dtypes.bfloat16, [])
    it = iron.tensor(np.ascontiguousarray(src), dtype=ml_dtypes.bfloat16, device="npu")
    o1 = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
    d1(it, o1)
    check("1 input ", o1.numpy().astype(np.float32).reshape(M, D), want)

    p2 = GEN / "copy_only2_shim.cc"
    p2.write_text(SHIM_2IN)
    d2 = _build_oneshot(
        "copy_only2", p2, [N, CBUF_N], N,
        [ml_dtypes.bfloat16, np.int32], ml_dtypes.bfloat16, [],
    )
    cb = iron.tensor(np.zeros(CBUF_N, dtype=np.int32), dtype=np.int32, device="npu")
    o2 = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
    d2(it, cb, o2)
    check("2 inputs", o2.numpy().astype(np.float32).reshape(M, D), want)


if __name__ == "__main__":
    main()
