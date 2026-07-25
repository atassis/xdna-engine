#!/usr/bin/env python3
"""DEVICE BISECT for the int4 f32-epilogue miscompile (llvm-aie-int4-epilogue-f32-store-miscompile).

The blocker named by the prior session: "store both int32 AND f32 from ONE run to see which
tile diverges". This does exactly that, with NOTHING between the two stores but `to_float`:

    iv = acc.to_vector<int32_t>()      -> stored to out[0 : M*N]        (int32)
    pf = aie::to_float(iv, 0)          -> stored to out[M*N : 2*M*N]    (f32, bit-cast)

scale is dropped entirely (G == K here, numGroups == 1, scale == 1), so the f32 half is a pure
`to_float` of the int32 half. Therefore:
  - int32 half wrong at tile T  -> the mmul/accumulate path is at fault (NOT a Peano f32 bug).
  - int32 half exact everywhere but f32 half wrong at tile T -> the defect is precisely
    to_float -> f32-store, and T pins the miscompiled instruction. That is the upstream repro.

Shape 8x64x64 g64 = the shape the fault was originally observed at (wrong tile mi=1, ni=0).
"""
import numpy as np
import ml_dtypes  # noqa: F401  (registers bf16 with numpy; harness parity)

import aie.iron as iron
import bricklib
from pathlib import Path
from verify_f2 import tile_pack
from verify_f2b import pack_int4_blocks, pack_int4

GEN = Path(__file__).parent / "gen"
GEN.mkdir(exist_ok=True)

M, K, N = 8, 64, 64
mTiles, nTiles, sizeC = M // 4, N // 16, 64

rng = np.random.default_rng(17)          # same seed as verify_dequant_probe.py
a = rng.integers(-127, 128, (M, K), dtype=np.int64).astype(np.int8)
b = rng.integers(-7, 8, (K, N), dtype=np.int64).astype(np.int8)

# Exact integer reference, then laid out in the kernel's (mi,ni,4,16) tile-blocked order.
ref_i32 = (a.astype(np.int32) @ b.astype(np.int32))


def tile_rowmajor(x, R=4, C=16):
    Rows, Cols = x.shape
    return x.reshape(Rows // R, R, Cols // C, C).transpose(0, 2, 1, 3).reshape(-1)


ref_tiled = tile_rowmajor(ref_i32)

sym = "gi4dq_bisect"
shim = f'''#include <stdint.h>
#include <aie_api/aie.hpp>
extern "C" void {sym}(const int8_t* __restrict pA,
                      const int8_t* __restrict pB_only,
                      int32_t* __restrict out) {{
  using MMUL = aie::mmul<4,16,16,int8_t,int4,accauto>;
  const int4* pB = (const int4*)pB_only;
  constexpr unsigned M={M}, K={K}, N={N};
  constexpr unsigned kSteps=K/16, mTiles=M/4, nTiles=N/16;
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  int32_t* i32out = out;                    // [M*N]  raw int32 accumulator
  float*   f32out = (float*)(out + M*N);    // [M*N]  to_float of the SAME vector
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
      ::aie::vector<int32_t, MMUL::size_C> iv = acc.template to_vector<int32_t>();
      ::aie::store_v(i32out + (mi*nTiles+ni)*MMUL::size_C, iv);
      ::aie::vector<float, MMUL::size_C> pf = ::aie::to_float(iv, 0);
      ::aie::store_v(f32out + (mi*nTiles+ni)*MMUL::size_C, pf);
    }}
  }}
}}'''
(GEN / f"{sym}_shim.cc").write_text(shim)

b_pad = pack_int4_blocks(b, K, N, pack_int4)
design = bricklib._build_oneshot(sym, GEN / f"{sym}_shim.cc",
                                 [M * K, b_pad.size], 2 * M * N,
                                 [np.int8, np.int8], np.int32, [])
a_t = iron.tensor(np.ascontiguousarray(tile_pack(a, 4, 16)), dtype=np.int8, device="npu")
b_t = iron.tensor(np.ascontiguousarray(b_pad), dtype=np.int8, device="npu")
o_t = iron.zeros((2 * M * N,), dtype=np.int32, device="npu")
design(a_t, b_t, o_t)
raw = np.array(o_t.numpy().copy(), copy=True)

dev_i32 = raw[:M * N].astype(np.int64)
dev_f32 = raw[M * N:].view(np.float32).astype(np.float64)
ref64 = ref_tiled.astype(np.int64)

print(f"[bisect {M}x{K}x{N}] tiles={mTiles}x{nTiles} sizeC={sizeC}")
print(f"  int32 half : exact={np.array_equal(dev_i32, ref64)}  "
      f"mismatches={int((dev_i32 != ref64).sum())}/{M*N}")
f32_bad = dev_f32 != ref64.astype(np.float64)
print(f"  f32   half : exact={not f32_bad.any()}  mismatches={int(f32_bad.sum())}/{M*N}")

print("  per-tile (mi,ni): int32_bad / f32_bad element counts")
for mi in range(mTiles):
    for ni in range(nTiles):
        s = slice((mi * nTiles + ni) * sizeC, (mi * nTiles + ni + 1) * sizeC)
        ib = int((dev_i32[s] != ref64[s]).sum())
        fb = int(f32_bad[s].sum())
        flag = "   <== DIVERGES" if (ib or fb) else ""
        print(f"    ({mi},{ni}): int32_bad={ib:3d}  f32_bad={fb:3d}{flag}")

bad_tiles = [(mi, ni) for mi in range(mTiles) for ni in range(nTiles)
             if (dev_i32[(mi * nTiles + ni) * sizeC:(mi * nTiles + ni + 1) * sizeC]
                 != ref64[(mi * nTiles + ni) * sizeC:(mi * nTiles + ni + 1) * sizeC]).any()
             or f32_bad[(mi * nTiles + ni) * sizeC:(mi * nTiles + ni + 1) * sizeC].any()]
print(f"\n  VERDICT: bad tiles = {bad_tiles or 'NONE (bug did not reproduce)'}")
if bad_tiles:
    mi, ni = bad_tiles[0]
    s = slice((mi * nTiles + ni) * sizeC, (mi * nTiles + ni + 1) * sizeC)
    ib = (dev_i32[s] != ref64[s]).any()
    print(f"  first bad tile ({mi},{ni}): int32 {'WRONG -> mmul/accumulate path' if ib else 'EXACT'}"
          f" | f32 {'WRONG -> to_float/f32-store' if f32_bad[s].any() else 'exact'}")
    print(f"    ref  [:8] = {ref64[s][:8]}")
    print(f"    i32  [:8] = {dev_i32[s][:8]}")
    print(f"    f32  [:8] = {dev_f32[s][:8]}")
