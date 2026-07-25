#!/usr/bin/env python3
"""Test the int4 B-block byte-stride hypothesis: multi-block fails because the kernel strides
B by MMUL::size_B (256) int4-units and int4* arithmetic advances full bytes -> 256 bytes/block,
but packed blocks are only 128 bytes. Fix candidate: pad each 128-byte packed block to 256 bytes.
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


def pack_b(b, K, N, pad):
    """block-tiled (ki,ni) 16x16 int4 blocks; each block pack_int4 -> 128 bytes.
    pad=True -> pad each block to 256 bytes (128 data + 128 zero)."""
    Kb, Nb = K // 16, N // 16
    blocks = b.reshape(Kb, 16, Nb, 16).transpose(0, 2, 1, 3)  # (ki,ni,16,16)
    out = []
    for ki in range(Kb):
        for ni in range(Nb):
            blk = blocks[ki, ni].reshape(-1)                    # 256 int4
            packed = g.pack_int4(blk).astype(np.int8)           # 128 bytes
            if pad:
                packed = np.concatenate([packed, np.zeros(128, np.int8)])
            out.append(packed)
    return np.concatenate(out)


def gemm_i4(M, K, N, pad, seed):
    sym = f"gi4s_{M}x{K}x{N}"
    shim = GEN / f"{sym}_shim.cc"
    shim.write_text(
        '#include <stdint.h>\n'
        f'#include "{CC}"\n'
        f'extern "C" void {sym}(const int8_t*a,const int4*b,int32_t*c)'
        f'{{gemm_int8xint4_tile<{M},{K},{N}>(a,b,c);}}\n'
    )
    b_bytes = pack_b_len = ((K * N) // 2) if not pad else ((K // 16) * (N // 16) * 256)
    design = bricklib._build_oneshot(sym, shim, [M * K, b_bytes], M * N,
                                     [np.int8, np.int8], np.int32, [])
    rng = np.random.default_rng(seed)
    a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
    b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
    ref = g.gemm_int8xint4_ref(a, b)
    a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
    b_t = iron.tensor(np.ascontiguousarray(pack_b(b, K, N, pad)), dtype=np.int8, device="npu")
    c_t = iron.zeros((M * N,), dtype=np.int32, device="npu")
    design(a_t, b_t, c_t)
    got = np.array(tile_unpack(c_t.numpy().copy(), M, N, 4, 16), copy=True)
    num = np.linalg.norm((got.astype(np.float64) - ref.astype(np.float64)).ravel())
    den = np.linalg.norm(ref.astype(np.float64).ravel())
    rl2 = num / den if den else num
    print(f"[i4 {M}x{K}x{N} pad={int(pad)}] rel_l2={rl2:.3e} {'PASS' if rl2 <= 3e-2 else 'FAIL'}",
          flush=True)
    return rl2


for pad in (True, False):
    for (M, K, N) in [(16, 32, 16), (16, 16, 32), (64, 64, 64)]:
        gemm_i4(M, K, N, pad, seed=200)

import os
os._exit(0)
