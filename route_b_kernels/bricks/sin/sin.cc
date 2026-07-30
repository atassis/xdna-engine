//===- sin.cc --------------------------------------------------*- C++ -*-===//
//
// GENERIC vectorized sin brick (route_b_kernels/bricks, group: transcendental).
//
// WHY A POLYNOMIAL AND NOT A LUT. The measured KB verdict
// (kb/lut-primitive-choice-lookup-vs-linear-approx) is that aie::linear_approx beats a gather on sin
// by 3.5x-65x at matched memory. That verdict assumes the lut<4> ab/cd layout WORKS. In this repo it
// does not: the duplication layout is an OPEN question
// (log/2026-07/2026-07-25-rope-lut-root-cause-bank-granularity.md carries a same-day retraction and
// ends "The layout question is OPEN"), it already blocks the rope-lut brick, and
// _verify/probe_linear_approx_abcd.py measured our best guess still returning permuted entries.
// Snake needs ~1e-2 and the polynomial delivers ~2e-3, so the codec does not gate on that open
// question. Solving the layout stays a separate task -- this is a scoped detour, not a retreat.
//
// Argument fold, f32 (precision matters at large |x|: Snake's alpha*x reaches ~10 periods):
//   1. r = x - 2*pi*round(x / 2*pi)                  -> [-pi, pi], where round() is the f32
//                                                       (q + 2^23) - 2^23 identity, because
//                                                       aie::to_float on vector<int32,16> does not
//                                                       instantiate on aie2p.
// Then a 6-term odd Taylor polynomial, accurate to 4.5e-4 across all of [-pi, pi] -- so no second
// reduction stage and no conditional ops at all.
//
// Kept in f32 throughout: 4 multiplies + 3 adds per element is well short of the per-tile cycle
// budget that makes an all-f32 SiLU/GELU hang on this bf16-native unit (see mm_silu_epilogue's
// header), and it keeps the kernel bit-comparable to the numpy golden.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

template <int N>
void sin_core(const float *restrict input, float *restrict output, int32_t n) {
  event0();
  const int chunks = n / N;

  for (int i = 0; i < chunks; i++) {
    // REGISTER PRESSURE IS THE BINDING CONSTRAINT, not arithmetic. An earlier version kept every
    // Horner intermediate in its own named vector (~17 live) plus 9 hoisted broadcast constants; on
    // device the results came back ALIASED -- p3 returned r2's value, p2 returned r's, res returned
    // p4's, with zeros interleaved. That is spill codegen, not maths (and spills in this area are
    // what the pinned Peano carries llvm-aie #1155 for). So: one accumulator reassigned in place,
    // constants materialised at point of use, and nothing kept live that is already consumed.
    ::aie::vector<float, N> r = ::aie::load_v<N>(input + i * N);

    // fold to [-pi, pi]: r -= 2*pi*round(r/2*pi), rounding via the f32 (q + 2^23) - 2^23 identity
    // because aie::to_float on vector<int32,16> does not instantiate on aie2p.
    {
      ::aie::vector<float, N> k = ::aie::mul(r, ::aie::broadcast<float, N>(0.15915494309189535f));
      k = ::aie::add(k, ::aie::broadcast<float, N>(8388608.0f));
      k = ::aie::sub(k, ::aie::broadcast<float, N>(8388608.0f));
      ::aie::vector<float, N> kt = ::aie::mul(k, ::aie::broadcast<float, N>(6.28318530717958648f));
      r = ::aie::sub(r, kt);
    }

    // 6-term odd Taylor, Horner in r^2. Accurate to 4.5e-4 over all of [-pi, pi] (host-checked over
    // 2e5 points), so no second reduction stage and no conditional ops are needed.
    ::aie::vector<float, N> r2 = ::aie::mul(r, r);
    ::aie::vector<float, N> p = ::aie::mul(r2, ::aie::broadcast<float, N>(-2.50521083854417188e-08f));
    p = ::aie::add(p, ::aie::broadcast<float, N>(2.75573192239858907e-06f));
    p = ::aie::mul(p, r2);
    p = ::aie::add(p, ::aie::broadcast<float, N>(-1.98412698412698413e-04f));
    p = ::aie::mul(p, r2);
    p = ::aie::add(p, ::aie::broadcast<float, N>(8.33333333333333322e-03f));
    p = ::aie::mul(p, r2);
    p = ::aie::add(p, ::aie::broadcast<float, N>(-1.66666666666666657e-01f));
    p = ::aie::mul(p, r2);
    p = ::aie::add(p, ::aie::broadcast<float, N>(1.0f));
    p = ::aie::mul(p, r);
    ::aie::store_v(output + i * N, p);
  }
  event1();
}

} // namespace route_b_bricks

extern "C" {

// sin over `n` f32 elements (n must be a multiple of the vector width).
void sin_f32(float *input, float *output, int32_t n) {
  route_b_bricks::sin_core<16>(input, output, n);
}

} // extern "C"
