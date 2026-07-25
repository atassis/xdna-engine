#!/usr/bin/env python3
"""Dump rope-lut's sin/cos buffers instead of rotating, to split the kernel in half.

The gather itself is proven healthy (probe_gather_width: width-16 fetch + the documented
de-interleave restores an identity ramp exactly). The apply loop is proven irrelevant to
the failure (probe_rope_identity: pos=0, where the rotation degenerates to a copy, fails
the same way). So the remaining suspect is the middle: angle -> key -> gathered sin/cos.

This shim replicates the kernel's per-row setup verbatim and writes sin_buf/cos_buf out
instead of applying them. For pos=0 every key is 0, so the answer is known exactly:
sin == 0.0 and cos == 1.0 for every lane. Anything else localises the bug to the
quantize/gather stage; sin=0,cos=1 exonerates it and moves the hunt to the apply loop.
"""
import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot
import verify_rope_lut as V

D, ROT, M = V.D, V.ROT, V.M
HALF = ROT // 2

SHIM = f'''#include <aie_api/aie.hpp>
#include <stdint.h>
#include "{V.BRICKS}/rope-lut/rope_lut_tables.inc"

// Mirrors rope_lut.cc's constants and per-row setup exactly.
constexpr unsigned kRotHalf = {HALF};
constexpr unsigned kFetchW = 16;
constexpr unsigned kVec = 16;
constexpr float kPi = 3.14159265358979323846f;

extern "C" void rope_sincos(int32_t *restrict cbuf, bfloat16 *restrict out) {{
  const int32_t *pos = cbuf;
  const float *inv_freq = (const float *)(cbuf + {M});
  const ::aie::lut<4, bfloat16> sin_lut(256, kSinLutAb, kSinLutCd);
  const ::aie::lut<4, bfloat16> cos_lut(256, kCosLutAb, kCosLutCd);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> sin_look(sin_lut, 0, 128);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> cos_look(cos_lut, 0, 128);
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  constexpr float kTwoPi = 2.0f * kPi;

  // row 0 only; write [sin(kRotHalf) | cos(kRotHalf)] into out
  const int32_t p = pos[0];
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
    ::aie::vector<bfloat16, kFetchW> s = sin_look.fetch(keys);
    ::aie::vector<bfloat16, kFetchW> c = cos_look.fetch(keys);
    s = ::aie::concat(::aie::filter_even(s), ::aie::filter_odd(s));
    c = ::aie::concat(::aie::filter_even(c), ::aie::filter_odd(c));
    ::aie::store_v(out + i, s);
    ::aie::store_v(out + kRotHalf + i, c);
  }}
}}
'''


def run(pos_val):
    p = GEN / "rope_sincos_shim.cc"
    p.write_text(SHIM)
    g, _ = V.load_golden()
    inv_freq = g.build_inv_freq(ROT).astype(np.float32)
    pos = np.full(M, pos_val, dtype=np.int32)
    cbuf = np.concatenate([pos.view(np.int32), inv_freq.view(np.int32)]).astype(np.int32)
    n_out = 2 * HALF
    design = _build_oneshot("rope_sincos", p, [cbuf.size], n_out,
                            [np.int32], ml_dtypes.bfloat16, [])
    ct = iron.tensor(np.ascontiguousarray(cbuf), dtype=np.int32, device="npu")
    ot = iron.zeros((n_out,), dtype=ml_dtypes.bfloat16, device="npu")
    design(ct, ot)
    v = ot.numpy().astype(np.float32)
    return v[:HALF], v[HALF:]


def main():
    for pos_val in (0, 1):
        s, c = run(pos_val)
        print(f"\n### pos={pos_val}")
        np.set_printoptions(linewidth=200)
        for grp in range(0, HALF, 16):
            print(f"  lanes {grp:3d}-{grp+15:3d} sin {np.array2string(s[grp:grp+16], precision=3)}")
        print(f"  cos[:16] {np.array2string(c[:16], precision=3)}")
        print(f"  |sin| max {np.abs(s).max():.4e}   |cos| max {np.abs(c).max():.4e}")
        if pos_val == 0:
            print(f"  EXPECT sin==0 everywhere, cos==1 everywhere")
            print(f"  sin all zero: {bool(np.all(s == 0.0))}   cos all one: {bool(np.all(c == 1.0))}")


if __name__ == "__main__":
    main()
