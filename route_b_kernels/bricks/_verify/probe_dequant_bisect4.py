#!/usr/bin/env python3
"""DEVICE BISECT stage 4 for the int4 dequant zero-output: the bf16 narrow + store.

Where the bisect stands:
  stage 1 -- int32 accumulate and to_float bit-exact on all 8 tiles.
  stage 2 -- divergence at the scale multiply, BUT with a LOCAL `float pScale[64]`, and
             patterned by ni, which identical inputs cannot cause.
  readback -- the kernel reads the PACKED scale f32-exact.
  stage 3 -- with the PACKED pointer, ALL of int32 / to_float / mul-via-concat /
             mul-via-stack-array are bit-exact on all 8 tiles. So the multiply is fine and
             stage 2's damage was its own local array.

Everything up to and including the multiply is therefore clean. What stage 3 did NOT
reproduce is the tail of the brick: the cross-group `cf` accumulation and the bf16 narrow +
store. That tail is also the ONLY part unique to this brick -- its int32-output sibling
gemm-int8xint4 passes bit-exact at the same shape, and differs precisely by outputting int32
instead of bf16.

So: run the brick's exact epilogue, bf16 out, and compare against the bf16-rounded golden.

  reproduces zero -> the defect is in the bf16 narrow/store or the cf accumulation, and the
                     per-tile map says which.
  comes back correct -> the epilogue is sound in isolation and the difference lives in the
                     brick file / its invocation, so diff this shim against the .cc.

Run:  ./run.sh probe_dequant_bisect4.py
"""
from pathlib import Path
import numpy as np
import ml_dtypes

import aie.iron as iron
import bricklib
from verify_f2 import tile_pack
from verify_f2b import pack_int4_blocks, pack_int4, tile_unpack

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

M, K, N, G = 8, 64, 64, 64
ngrp = K // G
mTiles, nTiles = M // 4, N // 16

rng = np.random.default_rng(17)
a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
scale = rng.uniform(0.005, 0.02, (ngrp, N)).astype(np.float32)   # REAL scale, not 1.0

ref_f32 = np.zeros((M, N), np.float64)
for g in range(ngrp):
    sl = slice(g * G, (g + 1) * G)
    ref_f32 += (a[:, sl].astype(np.int32) @ b[sl].astype(np.int32)).astype(np.float64) * scale[g]
ref = ref_f32.astype(ml_dtypes.bfloat16).astype(np.float32)

b_pad = pack_int4_blocks(b, K, N, pack_int4)
b_bytes = int(b_pad.size)
wbuf = np.concatenate([b_pad, scale.reshape(-1).view(np.int8)])

sym = "gi4dq_bisect4"
shim = f'''#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void {sym}(const int8_t* __restrict pA,
                      const int8_t* pWbuf,
                      bfloat16* __restrict pC) {{
  using MMUL = aie::mmul<4,16,16,int8_t,int4,accauto>;
  const int4*  pB     = (const int4*)pWbuf;
  const float* pScale = (const float*)(pWbuf + {b_bytes});
  constexpr unsigned kStepsPerGroup={G}/16, numGroups={K}/{G}, kSteps={K}/16;
  constexpr unsigned mTiles={M}/4, nTiles={N}/16;
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (unsigned mi=0; mi<mTiles; ++mi) {{
    for (unsigned ni=0; ni<nTiles; ++ni) {{
      ::aie::vector<float, MMUL::size_C> cf = ::aie::zeros<float, MMUL::size_C>();
      for (unsigned g=0; g<numGroups; ++g) {{
        MMUL acc;
        for (unsigned ks=0; ks<kStepsPerGroup; ++ks) {{
          const unsigned ki = g*kStepsPerGroup + ks;
          const int8_t* pA_tile = pA + (mi*kSteps+ki)*MMUL::size_A;
          const int4*   pB_tile = pB + (ki*nTiles+ni)*(MMUL::size_B/2);
          ::aie::vector<int8_t, MMUL::size_A> A = ::aie::load_v<MMUL::size_A>(pA_tile);
          ::aie::vector<int4,   MMUL::size_B> B = ::aie::load_v<MMUL::size_B>(pB_tile);
          if (ks==0) acc.mul(A,B); else acc.mac(A,B);
        }}
        ::aie::vector<float, MMUL::size_C> pf =
            ::aie::to_float(acc.template to_vector<int32_t>(), 0);
        const float* sg = pScale + g*{N} + ni*16;
        ::aie::vector<float,16> s16 = ::aie::load_v<16>(sg);
        ::aie::vector<float, MMUL::size_C> sv = ::aie::concat(s16,s16,s16,s16);
        cf = ::aie::add(cf, ::aie::mul(pf, sv).template to_vector<float>());
      }}
      ::aie::accum<accfloat, MMUL::size_C> ac;
      ac.from_vector(cf);
      bfloat16* pC_tile = pC + (mi*nTiles+ni)*MMUL::size_C;
      ::aie::store_v(pC_tile, ac.template to_vector<bfloat16>());
    }}
  }}
}}'''
(GEN / f"{sym}_shim.cc").write_text(shim)

design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc",
                                 [M * K, wbuf.size], M * N,
                                 [np.int8, np.int8], ml_dtypes.bfloat16, [])
a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
w_t = iron.tensor(np.ascontiguousarray(wbuf), dtype=np.int8, device="npu")
c_t = iron.zeros((M * N,), dtype=ml_dtypes.bfloat16, device="npu")
design(a_t, w_t, c_t)
dev = np.array(c_t.numpy().copy(), copy=True)
got = tile_unpack(dev, M, N, 4, 16).astype(np.float32)

nz = float(np.abs(got).sum())
rl2 = float(np.linalg.norm((got - ref).ravel()) / (np.linalg.norm(ref.ravel()) + 1e-12))
print(f"[bisect4 {M}x{K}x{N} g{G}] REAL scale, packed at wbuf+{b_bytes}, bf16 out\n")
print(f"  rel_l2={rl2:.3e}  nz={nz:.3e}  (gate 2e-2)")
print(f"  ref[0,:4] = {ref[0,:4]}")
print(f"  got[0,:4] = {got[0,:4]}")
print("\n  per-tile rel-L2:")
for mi in range(mTiles):
    for ni in range(nTiles):
        gs = got[mi*4:(mi+1)*4, ni*16:(ni+1)*16]
        rs = ref[mi*4:(mi+1)*4, ni*16:(ni+1)*16]
        t = float(np.linalg.norm((gs-rs).ravel()) / (np.linalg.norm(rs.ravel()) + 1e-12))
        print(f"    ({mi},{ni}): {t:.3e}" + ("   <== BAD" if t > 1e-2 else ""))

if nz == 0.0:
    v = ("reproduces the EXACT-ZERO. The defect is in the cf accumulation or the bf16 "
         "narrow/store -- the only part stage 3 did not cover. Bisect within this tail.")
elif rl2 <= 2e-2:
    v = ("epilogue is CORRECT in isolation (rel-L2 within gate). The brick's own file or its "
         "invocation differs -- diff this shim against gemm_int8xint4_dequant.cc.")
else:
    v = f"nonzero but wrong (rel_l2={rl2:.3e}) -- a numeric bug in the tail, not a dead path."
print(f"\n  VERDICT: {v}")
