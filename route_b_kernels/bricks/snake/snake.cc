//===- snake.cc ------------------------------------------------*- C++ -*-===//
//
// Snake activation brick (route_b_kernels/bricks, group: activation):
//   y = x + sin^2(alpha * x) / alpha,  alpha per-CHANNEL, broadcast along time.
//
// The fish-audio DAC codec decoder applies Snake at every upsample stage, and S2-Pro and s1-mini
// share a byte-identical codec.pth, so this brick serves both. Reference semantics:
// s2.cpp/src/s2_codec.cpp:197-204.
//
// Composes the sin brick's vector-level `sin_v` rather than re-deriving a transcendental: one
// reusable primitive, one op-TYPE on top. No scratch buffer and no second pass over memory.
//
// One call processes ONE channel row of `t` elements with that channel's alpha. Keep `t` at 64: the
// sin brick measured exact at 64 floats per call and wrong past element 64 for larger single calls
// (it reads beyond the delivered tile), so the row is 64 wide and volume comes from the row count.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../sin/sin.cc"

namespace route_b_bricks {

template <int N>
void snake_core(const float *restrict input, float *restrict output, int32_t t, float alpha) {
  event0();
  const int chunks = t / N;
  const float inv_alpha = 1.0f / alpha;
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> x = ::aie::load_v<N>(input + i * N);
    ::aie::vector<float, N> ax = ::aie::mul(x, ::aie::broadcast<float, N>(alpha));
    ::aie::vector<float, N> s = sin_v<N>(ax);
    ::aie::vector<float, N> s2 = ::aie::mul(s, s);
    ::aie::vector<float, N> q = ::aie::mul(s2, ::aie::broadcast<float, N>(inv_alpha));
    ::aie::store_v(output + i * N, ::aie::add(x, q));
  }
  event1();
}

} // namespace route_b_bricks

extern "C" {

// Snake over ONE channel row of `t` f32 elements with that channel's `alpha` (> 0).
void snake_f32(float *input, float *output, int32_t t, float alpha) {
  route_b_bricks::snake_core<16>(input, output, t, alpha);
}

} // extern "C"
