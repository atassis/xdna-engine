#!/usr/bin/env python3
"""One-hot / identity probe for the int4 mmul<4,16,16> (mmul_8_4) packed-B layout.

16x16x16 single tile, A = identity (A row-major is confirmed good: int8 GEMM is bit-exact
with the same tiling). Then C[m,n] = B_kernel[m,n] -- the output directly reads out B as
the mmul datapath places it. Feed B encoding its logical k in run-K and logical n in run-N
-> recover, for every kernel output cell (m,n), which LOGICAL (k,n) my current packing
delivered there. That map IS the layout error; from it we build the corrected pack + verify.
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

M = K = N = 16
GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)
SHIM = GEN / "gi4_probe_shim.cc"
SHIM.write_text(
    '#include <stdint.h>\n'
    f'#include "{CC}"\n'
    'extern "C" void gi4_probe(const int8_t*a,const int4*b,int32_t*c){'
    'gemm_int8xint4_tile<16,16,16>(a,b,c);}\n'
)

design = bricklib._build_oneshot("gi4_probe", SHIM, [M * K, (K * N) // 2], M * N,
                                 [np.int8, np.int8], np.int32, [])


def run(a, b_logical):
    a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
    b_tiled = tile_pack(b_logical, 16, 16)
    b_packed = g.pack_int4(b_tiled).astype(np.int8)
    b_t = iron.tensor(np.ascontiguousarray(b_packed), dtype=np.int8, device="npu")
    c_t = iron.zeros((M * N,), dtype=np.int32, device="npu")
    design(a_t, b_t, c_t)
    return np.array(tile_unpack(c_t.numpy().copy(), M, N, 4, 16), copy=True)


np.set_printoptions(linewidth=200, threshold=10000)
A = np.eye(M, dtype=np.int8)
Bk = np.tile((np.arange(16) - 8).reshape(16, 1), (1, 16)).astype(np.int8)  # B[k,n]=k-8
Bn = np.tile((np.arange(16) - 8).reshape(1, 16), (16, 1)).astype(np.int8)  # B[k,n]=n-8

print("probe: running run-K ...", flush=True)
Ck = run(A, Bk)
print("run-K done; running run-N ...", flush=True)
Cn = run(A, Bn)
print("run-N done", flush=True)
OUT = GEN / "int4_probe_out.npz"
np.savez(OUT, Ck=Ck, Cn=Cn)
print(f"saved raw device outputs -> {OUT}", flush=True)
print("=== recovered k[m,n] ===", flush=True)
print((Ck + 8).astype(int), flush=True)
print("=== recovered n[m,n] ===", flush=True)
print((Cn + 8).astype(int), flush=True)
import os
os._exit(0)  # skip XRT teardown (segfaults on this stack); data already saved
