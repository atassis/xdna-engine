#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal standalone repro: a single GEMV at the lm-head's shape.

The fused 28-layer graph takes ~7 minutes to rebuild, which is a terrible loop for a one-op bug. This
builds ONE GEMV(M, K) with a chosen tiling, runs it on device against a known input, and reports both
the error and whether the output is a PERMUTATION of the right answer -- the two have different fixes.

A permutation (sorted values match, positions do not) is a tile-to-row mapping bug: mv.cc writes its
output at `c + row_offset * m`, so the mapping from (column, tile) to output rows is the suspect.
Wrong VALUES would instead be a compute or streaming bug.

  python repro_lmhead_gemv.py --m 151936 --k 1024            # the shipped lm-head shape
  python repro_lmhead_gemv.py --m 151936 --k 1024 --tso 16   # sweep the tiling
"""
import argparse
import os
import sys

import numpy as np
import ml_dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newstack_compat  # noqa: F401,E402
from iron.common import AIEContext  # noqa: E402
from elf_dispatch_compat import OperatorSequence  # noqa: E402
from iron.operators.gemv.op import GEMV  # noqa: E402
from gen_llm_decode import gemv_tile_output, COLS  # noqa: E402

BF16 = ml_dtypes.bfloat16


def run_one(M, K, cols, tsi, tso, seed):
    """Build + run ONE GEMV; return (rel-L2, rel-L2 of sorted values, tiles per column)."""
    ctx = AIEContext()
    op = GEMV(M=M, K=K, num_aie_columns=cols, tile_size_input=tsi,
              tile_size_output=tso, context=ctx)
    fused = OperatorSequence(f"gemv_{M}_{K}_{tsi}_{tso}", [(op, "W", "x", "y")],
                              input_args=["x"], output_args=["y"],
                              buffer_sizes={"y": M * 2}, context=ctx)
    fused.compile()
    c = fused.get_callable()
    rng = np.random.default_rng(seed)
    W = np.asarray(rng.standard_normal((M, K)) * 0.05, BF16).astype(np.float32)
    x = np.asarray(rng.standard_normal(K) * 0.5, BF16).astype(np.float32)
    np.copyto(c.get_buffer("W").data, np.asarray(W, BF16).reshape(-1))
    np.copyto(c.get_buffer("x").data, np.asarray(x, BF16).reshape(-1))
    c()
    got = np.asarray(c.get_buffer("y").data, np.float32)[:M]
    ref = np.asarray(W @ x, BF16).astype(np.float32)
    n = np.linalg.norm(np.float64(ref))
    d = np.linalg.norm(np.float64(ref - got)) / n
    ds = np.linalg.norm(np.float64(np.sort(ref) - np.sort(got))) / n
    return d, ds, (M // cols) // tso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=151936)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--tsi", type=int, default=None)
    ap.add_argument("--tso", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", default=None,
                    help="comma-separated M:K:tsi:tso configs, run in one device session")
    a = ap.parse_args()

    if a.sweep:
        print(f"{'M':>8} {'K':>6} {'tsi':>4} {'tso':>7} {'tiles/col':>9} "
              f"{'rel-L2':>11} {'sorted':>11}  verdict")
        for spec in a.sweep.split(","):
            m, k, tsi, tso = (int(v) for v in spec.split(":"))
            try:
                d, ds, nt = run_one(m, k, a.cols, tsi, tso, a.seed)
            except Exception as e:
                print(f"{m:8} {k:6} {tsi:4} {tso:7} {'-':>9} {'BUILD FAIL':>11}  {type(e).__name__}: {str(e)[:40]}")
                continue
            verdict = "ok" if d < 0.05 else ("PERMUTED" if ds < d / 10 else "WRONG VALUES")
            print(f"{m:8} {k:6} {tsi:4} {tso:7} {nt:9} {d:11.4e} {ds:11.4e}  {verdict}")
        return 0

    if a.tso is None or a.tsi is None:
        tsi, tso = gemv_tile_output(a.m, a.k)
    else:
        tsi, tso = a.tsi, a.tso
    n_tiles = (a.m // a.cols) // tso
    print(f"[repro] GEMV M={a.m} K={a.k} cols={a.cols} tsi={tsi} tso={tso} "
          f"-> {n_tiles} tiles/column, {n_tiles * a.cols} total")

    ctx = AIEContext()
    op = GEMV(M=a.m, K=a.k, num_aie_columns=a.cols, tile_size_input=tsi,
              tile_size_output=tso, context=ctx)
    fused = OperatorSequence("lmhead_repro", [(op, "W", "x", "y")],
                              input_args=["x"], output_args=["y"],
                              buffer_sizes={"y": a.m * 2}, context=ctx)
    fused.compile()
    c = fused.get_callable()

    rng = np.random.default_rng(a.seed)
    W = np.asarray(rng.standard_normal((a.m, a.k)) * 0.05, BF16).astype(np.float32)
    x = np.asarray(rng.standard_normal(a.k) * 0.5, BF16).astype(np.float32)
    np.copyto(c.get_buffer("W").data, np.asarray(W, BF16).reshape(-1))
    np.copyto(c.get_buffer("x").data, np.asarray(x, BF16).reshape(-1))
    c()
    got = np.asarray(c.get_buffer("y").data, np.float32)[:a.m]
    ref = np.asarray(W @ x, BF16).astype(np.float32)

    d = np.linalg.norm(np.float64(ref - got)) / np.linalg.norm(np.float64(ref))
    print(f"[repro] rel-L2 = {d:.4e}   argmax ref {int(np.argmax(ref))} got {int(np.argmax(got))}")

    # permutation test: same multiset of values, different order
    ds = np.linalg.norm(np.float64(np.sort(ref) - np.sort(got))) / np.linalg.norm(np.float64(ref))
    print(f"[repro] rel-L2 of SORTED values = {ds:.4e}"
          f"   {'<-- values are right, ORDER is wrong (mapping bug)' if ds < d / 10 else ''}")
    nz = int((got == 0).sum())
    print(f"[repro] exact zeros in output: {nz}/{a.m}")
    # where does it go wrong? per-tile, in the units the design actually emits
    per = a.m // (n_tiles * a.cols) if n_tiles else a.m
    errs = [np.linalg.norm(np.float64(ref[i * per:(i + 1) * per] - got[i * per:(i + 1) * per]))
            / max(np.linalg.norm(np.float64(ref[i * per:(i + 1) * per])), 1e-12)
            for i in range(min(n_tiles * a.cols, 16))]
    print("[repro] per-tile rel-L2 (first 16): " + " ".join(f"{e:.2e}" for e in errs))
    return 0 if d < 0.05 else 1


if __name__ == "__main__":
    raise SystemExit(main())
