#!/usr/bin/env python3
"""Does sin/cos survive the round trip through rope_lut.cc's LOCAL sin_buf/cos_buf?

probe_rope_sincos proves the GATHER is right: at pos=0 it reads sin==0 / cos==1 on all 64 lanes.
But it stores the fetched vectors STRAIGHT to the output buffer. The real kernel instead stores them
into two function-local arrays and loads them back in the apply loop, and that round trip has never
been isolated -- it is the one link between "gather correct" and "rows 0-1 corrupted".

This replicates the kernel's declaration and both accesses verbatim:

    alignas(::aie::vector_decl_align) bfloat16 sin_buf[kRotHalfPad];
    ...  ::aie::store_v(sin_buf + i, s);          # 16-lane store, sub-native for bf16 (native 32)
    ...  ::aie::load_v<kVec>(sin_buf + j)         # read back in the apply loop

and emits what the LOAD returns, per row, so a row-dependent fault is visible. At pos=0 every value
must be sin==0 / cos==1; anything else is the local array, not the gather.

Per-row output matters: the brick's damage is rows 0 and 1 of 16 and nothing else, so a probe that
only looks at one row can miss it entirely.
"""
import importlib.util

import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot

spec = importlib.util.spec_from_file_location(
    "g", "route_b_kernels/bricks/rope-lut/golden.py"
)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

M, ROT = 4, 128
HALF = ROT // 2
KVEC = 16
import pathlib

# Absolute: the compile runs out of ~/.npu/cache, so a repo-relative include is not found there.
BRICKS = str(pathlib.Path("route_b_kernels/bricks").resolve())

inv_freq = g.build_inv_freq(ROT).astype(np.float32)
POS_PAD = 16  # keep inv_freq 64-byte aligned; see probe_keys_vector_path
_pos = np.zeros(POS_PAD, dtype=np.int32)
cbuf = np.concatenate([_pos, inv_freq.view(np.int32)]).astype(np.int32)

SHIM = f"""#include <aie_api/aie.hpp>
#include <stdint.h>
#include "{BRICKS}/rope-lut/rope_lut_tables.inc"

constexpr unsigned kRotHalf = {HALF};
constexpr unsigned kFetchW = {KVEC};
constexpr unsigned kVec = {KVEC};
constexpr unsigned kRotHalfPad = ((kRotHalf + kFetchW - 1) / kFetchW) * kFetchW;
constexpr float kPi = 3.14159265358979323846f;

extern "C" void buf_roundtrip(int32_t *restrict cbuf, bfloat16 *restrict out) {{
  const ::aie::lut<4, bfloat16> sin_lut(256, kSinLutAb, kSinLutCd);
  const ::aie::lut<4, bfloat16> cos_lut(256, kCosLutAb, kCosLutCd);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> sin_look(sin_lut, 0, 128);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> cos_look(cos_lut, 0, 128);

  // EXACTLY the kernel's declaration.
  alignas(::aie::vector_decl_align) bfloat16 sin_buf[kRotHalfPad];
  alignas(::aie::vector_decl_align) bfloat16 cos_buf[kRotHalfPad];

  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  constexpr float kTwoPi = 2.0f * kPi;
  const int32_t *pos = cbuf;
  const float *inv_freq = (const float *)(cbuf + {POS_PAD});

  for (unsigned m = 0; m < {M}; ++m) {{
    const int32_t p = pos[m];
    ::aie::vector<float, kVec> posf =
        ::aie::mul(::aie::to_float(::aie::broadcast<int32, kVec>(p)), 1.0f);
    for (unsigned i = 0; i < kRotHalf; i += kVec) {{
      ::aie::vector<float, kVec> invf = ::aie::load_v<kVec>(inv_freq + i);
      ::aie::vector<float, kVec> theta = ::aie::mul(posf, invf);
      ::aie::vector<float, kVec> kwf = ::aie::mul(theta, 1.0f / kTwoPi);
      ::aie::vector<int32, kVec> k = ::aie::to_fixed<int32>(kwf);
      ::aie::vector<float, kVec> kf = ::aie::to_float(k);
      ::aie::vector<float, kVec> ktwopi = ::aie::mul(kf, kTwoPi);
      ::aie::vector<float, kVec> wrapped = ::aie::sub(theta, ktwopi);
      ::aie::vector<float, kVec> q = ::aie::mul(wrapped, 128.0f / kPi);
      ::aie::vector<int8, kVec> keys = ::aie::to_fixed<int8>(q);
      ::aie::store_v(sin_buf + i, sin_look.fetch(keys));
      ::aie::store_v(cos_buf + i, cos_look.fetch(keys));
    }}
    // Read the buffers back the way the apply loop does, and emit what we got.
    for (unsigned j = 0; j < kRotHalf; j += kVec) {{
      ::aie::store_v(out + m * 2 * kRotHalf + j, ::aie::load_v<kVec>(sin_buf + j));
      ::aie::store_v(out + m * 2 * kRotHalf + kRotHalf + j, ::aie::load_v<kVec>(cos_buf + j));
    }}
  }}
}}
"""


def main():
    p = GEN / "buf_roundtrip_shim.cc"
    p.write_text(SHIM)
    n_out = M * 2 * HALF
    design = _build_oneshot(
        "buf_roundtrip", p, [cbuf.size], n_out, [np.int32], ml_dtypes.bfloat16, []
    )
    ct = iron.tensor(np.ascontiguousarray(cbuf), dtype=np.int32, device="npu")
    ot = iron.zeros((n_out,), dtype=ml_dtypes.bfloat16, device="npu")
    design(ct, ot)
    dev = ot.numpy().astype(np.float32).reshape(M, 2, HALF)

    print("all rows have pos=0, so every lane must read sin==0 and cos==1\n")
    for m in range(M):
        s, c = dev[m][0], dev[m][1]
        ok = np.all(s == 0.0) and np.all(c == 1.0)
        print(f"row {m}: sin all zero={np.all(s == 0.0)}  cos all one={np.all(c == 1.0)}  {'OK' if ok else 'CORRUPT'}")
        if not ok:
            bs = np.nonzero(s != 0.0)[0]
            bc = np.nonzero(c != 1.0)[0]
            print(f"   sin bad lanes {bs[:16].tolist()} vals {s[bs][:8].tolist()}")
            print(f"   cos bad lanes {bc[:16].tolist()} vals {c[bc][:8].tolist()}")


if __name__ == "__main__":
    main()
