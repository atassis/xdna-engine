#!/usr/bin/env python3
"""Narrow rope-lut probe: pos=0 must be the IDENTITY rotation.

With pos=0 every theta is 0, so key=0, sin=0, cos=1, and the split-half rotation
degenerates to out1=x1, out2=x2. Input and output are both bf16, so the device output
must equal the input EXACTLY -- no tolerance, no golden model, no trig.

That splits the 6.846e+10 blow-up cleanly in two:
  * identity holds  -> the data path (packed const split, qk copy, bf16 marshaling) is
    fine and the bug is in the angle/LUT math for pos!=0;
  * identity fails  -> the trig is irrelevant, something in the plumbing is corrupt,
    and the huge magnitudes are garbage memory rather than bad sin/cos.

Also dumps raw values, because "how wrong" discriminates further: uniformly huge says
uninitialized memory, a few huge lanes says an indexing/stride fault, exact-but-permuted
says a de-interleave problem.
"""
import importlib.util
from pathlib import Path

import numpy as np
import ml_dtypes

import bricklib
import verify_rope_lut as V

BRICKS = Path(__file__).parent.parent


def main():
    g, cc = V.load_golden()
    rng = np.random.default_rng(0)
    qk_f32 = rng.standard_normal((V.M, V.D)).astype(np.float32)
    qk_bf16 = qk_f32.astype(ml_dtypes.bfloat16)
    qk_in_f32 = qk_bf16.astype(np.float32)

    inv_freq = g.build_inv_freq(V.ROT).astype(np.float32)

    for label, pos in (("pos=0 (identity)", np.zeros(V.M, dtype=np.int32)),
                       ("pos=arange (real)", np.arange(V.M, dtype=np.int32))):
        cbuf = np.concatenate([pos.view(np.int32), inv_freq.view(np.int32)]).astype(np.int32)
        # golden=input for the identity case is only meaningful for pos=0; for the real
        # case we just want the raw device numbers, so pass the input as a placeholder
        # golden and read rel_l2/nonzero off the result rather than trusting the verdict.
        res = bricklib.verify_oneshot(
            f"rope-probe-{label.split()[0]}", cc, V.SHIM_BODY, "rope_lut_verify",
            inputs=[(qk_bf16.reshape(-1), ml_dtypes.bfloat16), (cbuf, np.int32)],
            out_numel=V.M * V.D, out_shape=(V.M, V.D),
            unpack=lambda flat: np.asarray(flat, np.float32).reshape(V.M, V.D),
            golden=qk_in_f32, gate=1e9, out_dt=ml_dtypes.bfloat16,
            compile_flags=V.COMPILE_FLAGS)
        got = res.get("got")
        if got is None:
            print(f"\n### {label}: no 'got' in result dict; keys={list(res)}")
            continue
        got = np.asarray(got, np.float32)
        print(f"\n### {label}")
        print(f"  input  row0[:8]  {np.array2string(qk_in_f32[0, :8], precision=4)}")
        print(f"  device row0[:8]  {np.array2string(got[0, :8], precision=4)}")
        print(f"  input  row0[64:72] {np.array2string(qk_in_f32[0, 64:72], precision=4)}")
        print(f"  device row0[64:72] {np.array2string(got[0, 64:72], precision=4)}")
        finite = np.isfinite(got)
        print(f"  |device| max {np.abs(got[finite]).max():.4e}   non-finite {(~finite).sum()}/{got.size}")
        exact = int((got == qk_in_f32).sum())
        print(f"  elements EXACTLY equal to input: {exact}/{got.size}")
        # where does the damage live -- which rows, which half of the feature dim?
        bad = ~np.isclose(got, qk_in_f32, rtol=1e-2, atol=1e-2)
        print(f"  rows with any mismatch: {sorted(set(np.nonzero(bad)[0].tolist()))}")
        cols = np.nonzero(bad)[1]
        if cols.size:
            print(f"  col range of mismatch: {cols.min()}..{cols.max()} "
                  f"(kRotHalf={V.ROT // 2}; first-half={int((cols < V.ROT // 2).sum())}, "
                  f"second-half={int((cols >= V.ROT // 2).sum())})")


if __name__ == "__main__":
    main()
