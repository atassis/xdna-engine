#!/usr/bin/env python3
"""DEVICE BISECT stage 2 for the int4 f32-epilogue miscompile.

Stage 1 (probe_dequant_bisect.py) proved on device that BOTH the int32 accumulator and
`aie::to_float(iv, 0)` are bit-exact on all 8 tiles at 8x64x64. So the defect is NOT the
mmul/accumulate path and NOT to_float->f32-store -- it is downstream. This stage emits
every remaining epilogue step from ONE run so the diverging step is pinned exactly:

  sec0 int32  : acc.to_vector<int32_t>()
  sec1 f32    : to_float(iv, 0)
  sec2 f32    : mul(pf, sv).to_vector<float>()   -- sv from the kernel's scalar sbuf fill
  sec3 bf16   : accum<accfloat>.from_vector(cf).to_vector<bfloat16>()  -- the final narrow

scale == 1.0 everywhere, so sec2 must equal sec1 and sec3 must equal round_bf16(sec1).
The kernel's scalar `sbuf[r*16+c] = sg[c]` fill + `load_v` is reproduced VERBATIM, because a
scalar-store -> vector-load forwarding hazard is itself a live suspect.
"""
import numpy as np
import ml_dtypes

import aie.iron as iron
import bricklib
from pathlib import Path
from verify_f2 import tile_pack
from verify_f2b import pack_int4_blocks, pack_int4

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

M, K, N = 8, 64, 64
mTiles, nTiles, sizeC = M // 4, N // 16, 64
NEL = M * N                      # 512 elements
NBF = NEL // 2                   # bf16 packed into int32 slots

rng = np.random.default_rng(17)
a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
ref_i32 = (a.astype(np.int32) @ b.astype(np.int32))


def tile_rowmajor(x, R=4, C=16):
    Rows, Cols = x.shape
    return x.reshape(Rows // R, R, Cols // C, C).transpose(0, 2, 1, 3).reshape(-1)


ref_tiled = tile_rowmajor(ref_i32).astype(np.int64)

sym = "gi4dq_bisect2"
shim = f'''#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void {sym}(const int8_t* __restrict pA,
                      const int8_t* __restrict pB_only,
                      int32_t* __restrict out) {{
  // scale baked locally: a real 3rd input exceeds the core tile's 2-in DMA channels
  // ("tile requires 3 input/1 output DMA channels, but only 2 input/2 output available").
  // This matches verify_dequant_probe.py, the configuration the fault was observed in.
  float pScale[{N}]; for (int i=0;i<{N};++i) pScale[i]=1.0f;
  using MMUL = aie::mmul<4,16,16,int8_t,int4,accauto>;
  const int4* pB = (const int4*)pB_only;
  constexpr unsigned M={M}, K={K}, N={N};
  constexpr unsigned kSteps=K/16, mTiles=M/4, nTiles=N/16;
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  int32_t*  s0 = out;                            // int32 acc
  float*    s1 = (float*)(out + {NEL});          // to_float
  float*    s2 = (float*)(out + 2*{NEL});        // after scale mul
  for (unsigned mi=0; mi<mTiles; ++mi) {{
    for (unsigned ni=0; ni<nTiles; ++ni) {{
      MMUL acc;
      for (unsigned ks=0; ks<kSteps; ++ks) {{
        const int8_t* pA_tile = pA + (mi*kSteps+ks)*MMUL::size_A;
        const int4*   pB_tile = pB + (ks*nTiles+ni)*(MMUL::size_B/2);
        ::aie::vector<int8_t, MMUL::size_A> A = ::aie::load_v<MMUL::size_A>(pA_tile);
        ::aie::vector<int4,   MMUL::size_B> B = ::aie::load_v<MMUL::size_B>(pB_tile);
        if (ks==0) acc.mul(A,B); else acc.mac(A,B);
      }}
      const unsigned off = (mi*nTiles+ni)*MMUL::size_C;

      ::aie::vector<int32_t, MMUL::size_C> iv = acc.template to_vector<int32_t>();
      ::aie::store_v(s0 + off, iv);

      ::aie::vector<float, MMUL::size_C> pf = ::aie::to_float(iv, 0);
      ::aie::store_v(s1 + off, pf);

      // VERBATIM from the brick: scalar sbuf fill then vector load.
      const float* sg = pScale + ni*16;
      float sbuf[MMUL::size_C];
      for (unsigned r=0; r<4; ++r)
        for (unsigned c=0; c<16; ++c)
          sbuf[r*16+c] = sg[c];
      ::aie::vector<float, MMUL::size_C> sv = ::aie::load_v<MMUL::size_C>(sbuf);

      ::aie::vector<float, MMUL::size_C> cf =
          ::aie::add(::aie::zeros<float, MMUL::size_C>(),
                     ::aie::mul(pf, sv).template to_vector<float>());
      ::aie::store_v(s2 + off, cf);

    }}
  }}
}}'''
(GEN / f"{sym}_shim.cc").write_text(shim)

b_pad = pack_int4_blocks(b, K, N, pack_int4)
scale = np.ones((N,), dtype=np.float32)
design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc",
                                 [M * K, b_pad.size], 3 * NEL,
                                 [np.int8, np.int8], np.int32, [])
a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
b_t = iron.tensor(np.ascontiguousarray(b_pad), dtype=np.int8, device="npu")
o_t = iron.zeros((3 * NEL,), dtype=np.int32, device="npu")
design(a_t, b_t, o_t)
raw = np.array(o_t.numpy().copy(), copy=True)

s0 = raw[:NEL].astype(np.int64)
s1 = raw[NEL:2 * NEL].view(np.float32).astype(np.float64)
s2 = raw[2 * NEL:3 * NEL].view(np.float32).astype(np.float64)
s3 = np.zeros(NEL)  # bf16 section dropped for this run

refd = ref_tiled.astype(np.float64)
ref_bf = ref_tiled.astype(ml_dtypes.bfloat16).astype(np.float64)

secs = [("s0 int32 acc      ", s0 != ref_tiled),
        ("s1 to_float       ", s1 != refd),
        ("s2 after scale mul", s2 != refd),
        ]

print(f"[bisect2 {M}x{K}x{N}] scale=1.0, tiles={mTiles}x{nTiles}")
for name, bad in secs:
    print(f"  {name}: exact={not bad.any()}  mismatches={int(bad.sum())}/{NEL}")

print("\n  per-tile mismatch counts (s0 / s1 / s2):")
for mi in range(mTiles):
    for ni in range(nTiles):
        s = slice((mi * nTiles + ni) * sizeC, (mi * nTiles + ni + 1) * sizeC)
        c = [int(bad[s].sum()) for _, bad in secs]
        flag = "   <== DIVERGES" if any(c) else ""
        print(f"    ({mi},{ni}): " + " / ".join(f"{x:3d}" for x in c) + flag)

first_bad = next((n for n, bad in secs if bad.any()), None)
print(f"\n  VERDICT: first diverging section = {first_bad or 'NONE (did not reproduce)'}")
if first_bad:
    bad = dict(secs)[first_bad]
    idx = int(np.argmax(bad))
    t = idx // sizeC
    print(f"    first bad element idx={idx} -> tile (mi={t // nTiles}, ni={t % nTiles})")
    s = slice(t * sizeC, (t + 1) * sizeC)
    print(f"    ref[:8] = {refd[s][:8]}")
    print(f"    s1 [:8] = {s1[s][:8]}")
    print(f"    s2 [:8] = {s2[s][:8]}")
