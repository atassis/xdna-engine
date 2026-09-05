#!/usr/bin/env python3
"""F2 family device-verify: GEMM/GEMV (single-shot, block-tiled layout).
   Start with exact int8 (gemm-int8, gemv-int8): int8xint8->int32, expect ~bit-exact.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path
import numpy as np

import bricklib

BRICKS = Path(__file__).parent.parent


def golden_mod(brick, fname):
    p = BRICKS / brick / "golden.py"
    spec = importlib.util.spec_from_file_location(f"{brick.replace('-', '_')}_golden", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, str(BRICKS / brick / fname)


def tile_pack(x, R, C):
    """[Rows,Cols] -> 8x8-block-tiled flat (block order row-major, row-major within block)."""
    Rows, Cols = x.shape
    return np.ascontiguousarray(
        x.reshape(Rows // R, R, Cols // C, C).transpose(0, 2, 1, 3).reshape(-1))


def tile_unpack(flat, Rows, Cols, R, C):
    return flat.reshape(Rows // R, Cols // C, R, C).transpose(0, 2, 1, 3).reshape(Rows, Cols)


results = []


def guard(fn):
    try:
        results.append(fn())
    except Exception as e:
        print(f"[{fn.brick_name:22s}] ERROR: {e}")
        traceback.print_exc()
        results.append(dict(name=fn.brick_name, status="ERROR", ok=False, err=str(e)))


def do_gemm_int8():
    g, cc = golden_mod("gemm-int8", "gemm_int8.cc")
    M = K = N = 64
    rng = np.random.default_rng(123)
    a = rng.integers(-127, 128, size=(M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-127, 128, size=(K, N), dtype=np.int64).astype(np.int8)
    ref = g.gemm_int8_ref(a, b)
    return bricklib.verify_oneshot(
        "gemm-int8", cc, "", "gemm_int8_64x64x64",
        inputs=[(tile_pack(a, 8, 8), np.int8), (tile_pack(b, 8, 8), np.int8)],
        out_numel=M * N, out_shape=(M, N),
        unpack=lambda flat: tile_unpack(flat, M, N, 8, 8),
        golden=ref, gate=3e-2, out_dt=np.int32)
do_gemm_int8.brick_name = "gemm-int8"


def do_gemv_int8():
    # Full 768x768 B (576KB) exceeds a single core tile's L1 (256KB); verify at
    # K=N=128 (fits L1) via -D overrides -- same kernel numerics, smaller resident B.
    g, cc = golden_mod("gemv-int8", "gemv_int8.cc")
    M, K, N = 8, 128, 128
    rng = np.random.default_rng(7)
    query = rng.integers(-127, 128, size=(K,), dtype=np.int64).astype(np.int8)
    a = g.make_padded_query(query, M)          # [M,K] int8, row 0 live
    b = rng.integers(-127, 128, size=(K, N), dtype=np.int64).astype(np.int8)
    ref = g.gemv_int8_i32(a, b)                 # [M,N] int32
    return bricklib.verify_oneshot(
        "gemv-int8", cc, "", "gemv_i8_i8_i32",
        inputs=[(tile_pack(a, 8, 8), np.int8), (tile_pack(b, 8, 8), np.int8)],
        out_numel=M * N, out_shape=(M, N),
        unpack=lambda flat: tile_unpack(flat, M, N, 8, 8),
        golden=ref, gate=3e-2, out_dt=np.int32,
        compile_flags=[f"-DGEMV_M={M}", f"-DGEMV_K={K}", f"-DGEMV_N={N}"])
do_gemv_int8.brick_name = "gemv-int8"


if __name__ == "__main__":
    for fn in (do_gemm_int8, do_gemv_int8):
        guard(fn)
    print("\n==== F2 (int8) SUMMARY ====")
    for r in results:
        print(f"  {r['name']:22s} {r['status']:10s} rel_l2={r.get('rel_l2', float('nan')):.3e}")
    passed = sum(1 for r in results if r.get("ok"))
    print(f"F2-int8: {passed}/{len(results)} PASS")
    print("JSON " + json.dumps(results))
    sys.exit(0 if passed == len(results) else 1)
