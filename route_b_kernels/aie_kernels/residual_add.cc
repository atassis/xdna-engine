//===- residual_add.cc ----------------------------------------*- C++ -*-===//
//
// Device-side f32 scaled residual add (whole-block-resident fusion). Keeps the block
// residual `x = a + scale*b` (Macaron FFN: a=x, b=ff, scale=0.5; full residual: scale=1.0)
// ON-CHIP so the residual never round-trips to host. out[t,c] = a[t,c] + scale*b[t,c], f32.
//
// `scale` is BAKED at IRON-generation time (passed as a compile-time literal from core_body,
// like `cols`): an AIE2 tile has only 2 input DMA channels, both consumed by the row-tiled a
// and b, so there is no channel left for a runtime scale param -> one xclbin per scale value
// (s050 = 0.5, s100 = 1.0). f32 mul-by-0.5/1.0 + one add is near-exact (rel-L2 ~0 vs host).
//
// 2-input ABI: a (g3), b (g4), out (g5). One call: ONE row of `cols`. cols % N == 0.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

template <int N>
void residual_add_row(const float *restrict a, const float *restrict b,
                      float *restrict out, float scale, int32_t cols) {
  event0();
  // Round-to-nearest-even so the on-chip f32 add matches the host round-nearest add bit-for-bit
  // (default AIE rounding is truncation -> 1-ULP drift that accumulates over blocks; the WER-path
  // rounding rule, same as glu.cc / affine_cast.cc).
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const ::aie::vector<float, N> sv = ::aie::broadcast<float, N>(scale);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> av = ::aie::load_v<N>(a + i);
    ::aie::vector<float, N> bv = ::aie::load_v<N>(b + i);
    ::aie::vector<float, N> sb = ::aie::mul(bv, sv);  // scale*b
    ::aie::store_v(out + i, ::aie::add(av, sb));      // a + scale*b
  }
  event1();
}

extern "C" {
void residual_add_row(float *a, float *b, float *out, float scale, int32_t cols) {
  residual_add_row<16>(a, b, out, scale, cols);
}
}
