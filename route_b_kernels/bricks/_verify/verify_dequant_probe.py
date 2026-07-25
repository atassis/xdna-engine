#!/usr/bin/env python3
"""Bisect the all-zero gemm-int8xint4-dequant output: bake scale=1.0 locally in the shim
(bypass the wbuf-packed scale). If C is then nonzero+correct -> the scale packing is the bug;
if still zero -> the bf16 output DMA or kernel exec is the problem."""
import importlib.util
from pathlib import Path
import numpy as np
import ml_dtypes

import aie.iron as iron
import bricklib
from verify_f2 import tile_pack, tile_unpack
from verify_f2b import pack_int4_blocks, pack_int4

BRICKS = Path(__file__).parent.parent
CC = str(BRICKS / "gemm-int8xint4-dequant" / "gemm_int8xint4_dequant.cc")
spec = importlib.util.spec_from_file_location("dq", BRICKS / "gemm-int8xint4-dequant" / "golden.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
GEN = Path(__file__).parent / "gen"

M, K, N, G = 8, 64, 64, 64
rng = np.random.default_rng(17)
a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
scale_ones = np.ones((K // G, N), dtype=np.float32)          # scale = 1 everywhere
_, ref_bf16 = g.dequant_gemm_ref(a, b, scale_ones, G)

sym = "gi4dq_probe"
shim = ('#include <stdint.h>\n'
        f'#include "{CC}"\n'
        'extern "C" void gi4dq_probe(const int8_t*a,const int8_t*b_only,bfloat16*c){\n'
        '  const int4*b=(const int4*)b_only;\n'
        '  float s[64]; for(int i=0;i<64;i++) s[i]=1.0f;\n'
        '  gemm_int8xint4_dequant_tile<8,64,64,64>(a,b,s,c);\n}')
(GEN / f"{sym}_shim.cc").write_text(shim)
b_pad = pack_int4_blocks(b, K, N, pack_int4)
design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc", [M * K, b_pad.size], M * N,
                                 [np.int8, np.int8], ml_dtypes.bfloat16, [])
a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
b_t = iron.tensor(np.ascontiguousarray(b_pad), dtype=np.int8, device="npu")
c_t = iron.zeros((M * N,), dtype=ml_dtypes.bfloat16, device="npu")
design(a_t, b_t, c_t)
dev = np.array(c_t.numpy().copy(), copy=True).astype(np.float32)   # RAW flat, kernel-tiled
ref = ref_bf16.astype(np.float32)                                  # [M,N] logical


def rl2(x, y):
    return float(np.linalg.norm((x - y).ravel()) / (np.linalg.norm(y.ravel()) + 1e-12))


# The kernel writes C tile-blocked (mi,ni)x4x16. Build the int-GEMM ref tiled several
# ways and compare to the RAW dev to identify the actual C-block element order.
def tile_rowmajor(x, R, C):  # (mi,ni,4,16) row-major within block
    Rows, Cols = x.shape
    return x.reshape(Rows // R, R, Cols // C, C).transpose(0, 2, 1, 3).reshape(-1)


def tile_colmajor(x, R, C):  # (mi,ni,16,4) col-major within block
    Rows, Cols = x.shape
    return x.reshape(Rows // R, R, Cols // C, C).transpose(0, 2, 3, 1).reshape(-1)


cands = {
    "rowmajor-4x16": tile_rowmajor(ref, 4, 16),
    "colmajor-4x16": tile_colmajor(ref, 4, 16),
    "flat-rowmajor": ref.reshape(-1),
}
print(f"[dequant scale=1] nz-sum={np.abs(dev).sum():.3e}", flush=True)
for name, cand in cands.items():
    print(f"  raw-dev vs {name}: rel_l2={rl2(dev, cand):.3e}", flush=True)
np.savez(GEN / "dequant_probe.npz", dev=dev, ref=ref)
import os
os._exit(0)
