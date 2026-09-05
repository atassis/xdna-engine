#!/usr/bin/env python3
"""Localize the int4 GEMM multi-block bug: the single-tile identity probe is clean, so the
error is in tiling/accumulation. Sweep shapes to isolate which dim's multi-block breaks:
single tile, +k-blocks, +n-blocks, +m-blocks, then all.
"""
import importlib.util
from pathlib import Path
import numpy as np

import aie.iron as iron
import bricklib
from verify_f2 import tile_pack, tile_unpack

BRICKS = Path(__file__).parent.parent
CC = str(BRICKS / "gemm-int8xint4" / "gemm_int8xint4.cc")
spec = importlib.util.spec_from_file_location("gi4", BRICKS / "gemm-int8xint4" / "golden.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)


def gemm_i4(M, K, N, seed):
    sym = f"gi4_{M}x{K}x{N}"
    shim = GEN / f"{sym}_shim.cc"
    shim.write_text(
        '#include <stdint.h>\n'
        f'#include "{CC}"\n'
        f'extern "C" void {sym}(const int8_t*a,const int4*b,int32_t*c)'
        f'{{gemm_int8xint4_tile<{M},{K},{N}>(a,b,c);}}\n'
    )
    design = bricklib._build_oneshot(sym, shim, [M * K, (K * N) // 2], M * N,
                                     [np.int8, np.int8], np.int32, [])
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
    ref = g.gemm_int8xint4_ref(a, b)
    a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
    b_packed = g.pack_int4(tile_pack(b, 16, 16)).astype(np.int8)
    b_t = iron.tensor(np.ascontiguousarray(b_packed), dtype=np.int8, device="npu")
    c_t = iron.zeros((M * N,), dtype=np.int32, device="npu")
    design(a_t, b_t, c_t)
    got = np.array(tile_unpack(c_t.numpy().copy(), M, N, 4, 16), copy=True)
    num = np.linalg.norm((got.astype(np.float64) - ref.astype(np.float64)).ravel())
    den = np.linalg.norm(ref.astype(np.float64).ravel())
    rl2 = num / den if den else num
    print(f"[i4 {M:3d}x{K:3d}x{N:3d}]  mTiles={M//4} kSteps={K//16} nTiles={N//16}  "
          f"rel_l2={rl2:.3e}  {'PASS' if rl2 <= 3e-2 else 'FAIL'}", flush=True)
    return rl2


SHAPES = [
    (16, 16, 16),   # single tile (probe was clean)
    (16, 32, 16),   # +k-blocks (tests K-accumulation)
    (16, 16, 32),   # +n-blocks
    (32, 16, 16),   # +m-blocks
    (64, 64, 64),   # all
]
for i, (M, K, N) in enumerate(SHAPES):
    gemm_i4(M, K, N, seed=100 + i)

import os
os._exit(0)
