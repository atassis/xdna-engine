#!/usr/bin/env python3
"""DEVICE BISECT stage 3 for the int4 dequant zero-output: PACKED pScale, two constructions.

Stage 1 proved int32-accumulate and `to_float` bit-exact on all 8 tiles.
Stage 2 (probe_dequant_bisect2.py) showed the divergence at `aie::mul(pf, sv)` -- but it
baked the scale into a LOCAL `float pScale[64]`, and the damage was patterned by ni
(tiles (0,2)(0,3)(1,2)(1,3) -> 0/NaN) even though every ni builds an identical multiplier.
Separately, probe_scale_readback.py proved the kernel reads the PACKED scale correctly
(f32-exact at wbuf+b_bytes), so scale delivery is not the fault.

That leaves exactly one question, which this stage answers: with the scale read from the
PACKED buffer (the brick's real path, not a local array), does the multiply still break --
and does it depend on HOW the [4,16] scale vector is built?

  sec0 int32 : acc.to_vector<int32_t>()                       (control, expected exact)
  sec1 f32   : to_float(iv, 0)                                (control, expected exact)
  sec2 f32   : mul(pf, sv) with sv = concat(load_v<16>(sg) x4)   <- the current kernel
  sec3 f32   : mul(pf, sv) with sv = scalar-filled sbuf + load_v <- the old kernel

scale is packed as 1.0 everywhere, so sec2 and sec3 must both equal sec1.

  both sections exact      -> the multiply is fine on the packed pointer; the local array
                              in bisect2 was the only broken thing, and the remaining zero
                              lives in the brick's group/store path, not here.
  sec3 bad, sec2 exact     -> the scalar-store -> vector-load stack array IS the defect and
                              the concat fix is correct.
  both bad                 -> the multiply itself is broken against a packed operand.

Run:  ./run.sh probe_dequant_bisect3.py
"""
from pathlib import Path
import numpy as np

import aie.iron as iron
import bricklib
from verify_f2 import tile_pack
from verify_f2b import pack_int4_blocks, pack_int4

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

M, K, N = 8, 64, 64
mTiles, nTiles, sizeC = M // 4, N // 16, 64
NEL = M * N
NSEC = 4

rng = np.random.default_rng(17)
a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)
ref_i32 = a.astype(np.int32) @ b.astype(np.int32)


def tile_rowmajor(x, R=4, C=16):
    Rows, Cols = x.shape
    return x.reshape(Rows // R, R, Cols // C, C).transpose(0, 2, 1, 3).reshape(-1)


ref_tiled = tile_rowmajor(ref_i32).astype(np.int64)

b_pad = pack_int4_blocks(b, K, N, pack_int4)
b_bytes = int(b_pad.size)
scale = np.ones((N,), dtype=np.float32)          # 1.0 -> sec2/sec3 must equal sec1
wbuf = np.concatenate([b_pad, scale.view(np.int8)])

sym = "gi4dq_bisect3"
shim = f'''#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void {sym}(const int8_t* __restrict pA,
                      const int8_t* pWbuf,
                      int32_t* __restrict out) {{
  using MMUL = aie::mmul<4,16,16,int8_t,int4,accauto>;
  // pB and pScale alias ONE packed buffer (a real 3rd input exceeds the core tile's 2-in
  // DMA budget), so neither may be __restrict.
  const int4*  pB     = (const int4*)pWbuf;
  const float* pScale = (const float*)(pWbuf + {b_bytes});
  constexpr unsigned kSteps={K}/16, mTiles={M}/4, nTiles={N}/16;
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  int32_t* s0 = out;
  float*   s1 = (float*)(out + {NEL});
  float*   s2 = (float*)(out + 2*{NEL});
  float*   s3 = (float*)(out + 3*{NEL});
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

      const float* sg = pScale + ni*16;

      // sec2: vector-only replication (the current kernel).
      ::aie::vector<float,16> s16 = ::aie::load_v<16>(sg);
      ::aie::vector<float, MMUL::size_C> sv2 = ::aie::concat(s16,s16,s16,s16);
      ::aie::store_v(s2 + off, ::aie::mul(pf, sv2).template to_vector<float>());

      // sec3: scalar-filled stack array (the old kernel), same packed pointer.
      float sbuf[MMUL::size_C];
      for (unsigned r=0; r<4; ++r)
        for (unsigned c=0; c<16; ++c)
          sbuf[r*16+c] = sg[c];
      ::aie::vector<float, MMUL::size_C> sv3 = ::aie::load_v<MMUL::size_C>(sbuf);
      ::aie::store_v(s3 + off, ::aie::mul(pf, sv3).template to_vector<float>());
    }}
  }}
}}'''
(GEN / f"{sym}_shim.cc").write_text(shim)

design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc",
                                 [M * K, wbuf.size], NSEC * NEL,
                                 [np.int8, np.int8], np.int32, [])
a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
w_t = iron.tensor(np.ascontiguousarray(wbuf), dtype=np.int8, device="npu")
o_t = iron.zeros((NSEC * NEL,), dtype=np.int32, device="npu")
design(a_t, w_t, o_t)
raw = np.array(o_t.numpy().copy(), copy=True)

s0 = raw[:NEL].astype(np.int64)
s1 = raw[NEL:2*NEL].view(np.float32).astype(np.float64)
s2 = raw[2*NEL:3*NEL].view(np.float32).astype(np.float64)
s3 = raw[3*NEL:4*NEL].view(np.float32).astype(np.float64)
refd = ref_tiled.astype(np.float64)

secs = [("s0 int32 acc          ", s0 != ref_tiled),
        ("s1 to_float           ", s1 != refd),
        ("s2 mul w/ concat sv   ", s2 != refd),
        ("s3 mul w/ stack sbuf  ", s3 != refd)]

print(f"[bisect3 {M}x{K}x{N}] PACKED pScale at wbuf+{b_bytes}, scale=1.0, tiles={mTiles}x{nTiles}\n")
for name, bad in secs:
    print(f"  {name}: exact={not bad.any()}  mismatches={int(bad.sum())}/{NEL}")

print("\n  per-tile mismatch counts (s0 / s1 / s2 / s3):")
for mi in range(mTiles):
    for ni in range(nTiles):
        sl = slice((mi*nTiles+ni)*sizeC, (mi*nTiles+ni+1)*sizeC)
        c = [int(bad[sl].sum()) for _, bad in secs]
        print(f"    ({mi},{ni}): " + " / ".join(f"{x:3d}" for x in c)
              + ("   <== DIVERGES" if any(c) else ""))

s2_ok, s3_ok = not secs[2][1].any(), not secs[3][1].any()
if s2_ok and s3_ok:
    v = ("multiply is FINE on the packed pointer, both constructions. bisect2's damage came "
         "from its LOCAL pScale array. The remaining zero-C is elsewhere in the brick "
         "(group loop / accumulate / bf16 store) -- look there next, not at the multiply.")
elif s2_ok and not s3_ok:
    v = ("the scalar-filled stack array IS the defect and concat() is the correct fix; the "
         "multiply itself is sound.")
elif not s2_ok and not s3_ok:
    v = "the multiply is broken against a PACKED operand regardless of how sv is built."
else:
    v = "concat construction is broken while the stack array is fine -- unexpected; re-read."
print(f"\n  VERDICT: {v}")
