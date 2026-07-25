//===- relu2.cc -----------------------------------------------*- C++ -*-===//
//
// Device-side ReLU^2 activation brick (act group): out = max(x, 0)^2.
//
// Generic op-TYPE against the resident [tile, D] stream contract (mirrors the
// per-row core_body shape of affine_cast.cc / glu.cc / ln_2pass.cc in
// route_b_kernels/aie_kernels/): ONE call processes ONE row of `cols` f32
// elements, cols % N == 0. Not tied to any one model -- any block that wants
// a squared-ReLU nonlinearity (e.g. a Gemma-style MLP variant, or a conv/FFN
// activation swap) calls this brick the same way glu_row is called for GLU.
//
// max_cmp brick + square: clamp-to-zero (aie::max against a 0.0f broadcast,
// the same op_max lane primitive used elsewhere in aie_api) followed by one
// self-multiply. Both stages stay in f32 -- ReLU^2 grows unboundedly for
// x>1 (unlike GLU's bounded sigmoid), so narrowing to bf16 mid-pipe would
// need range analysis a generic brick shouldn't assume; leave that fold to
// a caller-supplied epilogue (matching mm_silu_epilogue.cc's separation of
// "generic act brick" vs "fused mm epilogue instance").
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

template <int N>
void relu2_row(const float *restrict input, float *restrict output, int32_t cols) {
  event0();
  const ::aie::vector<float, N> zero = ::aie::broadcast<float, N>(0.0f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> v = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> clamped = ::aie::max(v, zero); // max_cmp: max(x, 0)
    ::aie::vector<float, N> squared = ::aie::mul(clamped, clamped); // square
    ::aie::store_v(output + i, squared);
  }
  event1();
}

extern "C" {
void relu2_row(float *input, float *output, int32_t cols) {
  relu2_row<16>(input, output, cols);
}
}
