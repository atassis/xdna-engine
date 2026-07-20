#!/usr/bin/env python3
"""Bisect the dequant ni=0 bug: reimplement the EXACT dequant epilogue in a shim but store
f32 (skip the bf16 narrow). If ni=0 is still wrong in f32 -> bug is in to_float/scale/accumulate;
if f32 is correct -> the bug is the bf16 store/narrow. scale=1 baked."""
import importlib.util
from pathlib import Path
import numpy as np
import aie.iron as iron
import bricklib
from verify_f2 import tile_pack, tile_unpack
from verify_f2b import pack_int4_blocks, pack_int4

BRICKS = Path(__file__).parent.parent
GEN = Path(__file__).parent / "gen"
CC = str(BRICKS / "gemm-int8xint4-dequant" / "gemm_int8xint4_dequant.cc")
spec = importlib.util.spec_from_file_location("dq", BRICKS / "gemm-int8xint4-dequant" / "golden.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

M, K, N, G = 8, 64, 64, 64
rng = np.random.default_rng(17)
a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
_, ref_bf16 = g.dequant_gemm_ref(a, b, np.ones((K // G, N), np.float32), G)  # scale=1
ref = ref_bf16.astype(np.float32)

# f32 reimplementation of gemm_int8xint4_dequant_tile<8,64,64,64> with scale=1, f32 out.
sym = "gi4dq_f32"
shim = (f'#include <stdint.h>\n#include "{CC}"\n' + r'''
extern "C" void gi4dq_f32(const int8_t* pA, const int8_t* pBb, float* pC){
  const int4* pB = (const int4*)pBb;
  using MMUL = aie::mmul<4,16,16,int8_t,int4,accauto>;
  constexpr unsigned M=8,K=64,N=64,G=64;
  constexpr unsigned kStepsPerGroup=G/16, numGroups=K/G, kSteps=K/16, mTiles=M/4, nTiles=N/16;
  aie::set_rounding(aie::rounding_mode::conv_even);
  for(unsigned mi=0;mi<mTiles;++mi) for(unsigned ni=0;ni<nTiles;++ni){
    MMUL acc;
    for(unsigned gg=0;gg<numGroups;++gg){
      for(unsigned ks=0;ks<kStepsPerGroup;++ks){
        const unsigned ki=gg*kStepsPerGroup+ks;
        aie::vector<int8_t,MMUL::size_A> A=aie::load_v<MMUL::size_A>(pA+(mi*kSteps+ki)*MMUL::size_A);
        aie::vector<int4,MMUL::size_B> B=aie::load_v<MMUL::size_B>(pB+(ki*nTiles+ni)*MMUL::size_B);
        if(ki==0) acc.mul(A,B); else acc.mac(A,B);
      }
    }
    // CHUNKED epilogue: to_float + store in 16-wide pieces (test the 64-wide codegen bug).
    aie::vector<int32_t,MMUL::size_C> ci = acc.template to_vector<int32_t>();
    float* pc = pC+(mi*nTiles+ni)*MMUL::size_C;
    for(unsigned q=0;q<MMUL::size_C/16;++q){
      aie::vector<float,16> cf16 = aie::to_float(ci.template extract<16>(q), 0);
      aie::store_v(pc + q*16, cf16);
    }
  }
}''')
(GEN / f"{sym}_shim.cc").write_text(shim)
b_pad = pack_int4_blocks(b, K, N, pack_int4)
design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc", [M * K, b_pad.size], M * N,
                                 [np.int8, np.int8], np.float32, [])
a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
b_t = iron.tensor(np.ascontiguousarray(b_pad), dtype=np.int8, device="npu")
c_t = iron.zeros((M * N,), dtype=np.float32, device="npu")
design(a_t, b_t, c_t)
got = tile_unpack(np.array(c_t.numpy().copy(), copy=True), M, N, 4, 16).astype(np.float64)
for mi in range(M // 4):
    for ni in range(N // 16):
        gg = got[mi * 4:mi * 4 + 4, ni * 16:ni * 16 + 16]
        rr = ref[mi * 4:mi * 4 + 4, ni * 16:ni * 16 + 16].astype(np.float64)
        rl = np.linalg.norm(gg - rr) / (np.linalg.norm(rr) + 1e-9)
        print(f"  f32-reimpl tile(mi={mi},ni={ni}): rel_l2={rl:.3e} {'WRONG' if rl > 1e-2 else 'ok'}", flush=True)
import os; os._exit(0)
