#!/usr/bin/env python3
"""Is the rows 0-1 corruption carried across the kernel's `m` loop? Sweep ROPE_M.

probe_rope_identity shows, at pos=0 where the rotation is the exact bf16 identity, that rows 0 and 1
of 16 are corrupted and rows 2-15 are bit-exact. Everything upstream is cleared: gather, the local
sin_buf/cos_buf round trip, the harness copy, the LUT layout.

The live hypothesis is a dependency carried across `m` -- row m+1's gather overwriting sin_buf/cos_buf
before row m's apply loop has read them. Neither available barrier could test it: Peano cannot
translate inline asm at all, and chess_separator_scheduler_local() produced byte-identical output,
which reads as a no-op rather than a refutation.

ROPE_M settles it without a barrier and without touching the kernel:
  * M=1 clean  -> nothing carried across m can be involved; the hypothesis stands by construction.
  * M=1 broken -> the fault is inside ONE iteration, and the suspicion moves to the qk pointer or the
    first vector store.
And the M=2/4 rows tell whether "the first two" is fixed or scales with M.
"""
from pathlib import Path

import numpy as np
import ml_dtypes

import bricklib
import verify_rope_lut as V


def run(M):
    g, cc = V.load_golden()
    rng = np.random.default_rng(0)
    qk_bf16 = rng.standard_normal((M, V.D)).astype(np.float32).astype(ml_dtypes.bfloat16)
    qk_in = qk_bf16.astype(np.float32)
    inv_freq = g.build_inv_freq(V.ROT).astype(np.float32)
    pos = np.zeros(M, dtype=np.int32)  # identity case
    cbuf = np.concatenate([pos.view(np.int32), inv_freq.view(np.int32)]).astype(np.int32)

    flags = [f for f in V.COMPILE_FLAGS if not f.startswith("-DROPE_M=")] + [f"-DROPE_M={M}"]
    res = bricklib.verify_oneshot(
        f"rope-msweep-M{M}", cc, V.SHIM_BODY, "rope_lut_verify",
        inputs=[(qk_bf16.reshape(-1), ml_dtypes.bfloat16), (cbuf, np.int32)],
        out_numel=M * V.D, out_shape=(M, V.D),
        unpack=lambda flat: np.asarray(flat, np.float32).reshape(M, V.D),
        golden=qk_in, gate=1e9, out_dt=ml_dtypes.bfloat16, compile_flags=flags)
    got = res.get("got")
    if got is None:
        print(f"M={M}: no 'got'; keys={list(res)}")
        return
    got = np.asarray(got, np.float32)
    bad = got != qk_in
    rows = sorted(set(np.nonzero(bad)[0].tolist()))
    print(f"M={M:>2}: exact {M * V.D - int(bad.sum())}/{M * V.D}   damaged rows {rows if rows else 'NONE'}")
    if rows:
        print(f"      row {rows[0]} want[:6] {qk_in[rows[0], :6].tolist()}")
        print(f"      row {rows[0]} got [:6] {got[rows[0], :6].tolist()}")


def main():
    print("pos=0 on every row, so the device output must equal the input bit-for-bit\n")
    for M in (1, 2, 4, 16):
        run(M)


if __name__ == "__main__":
    main()
