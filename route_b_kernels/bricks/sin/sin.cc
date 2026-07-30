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

  const ::aie::vector<float, N> two_pi = ::aie::broadcast<float, N>(6.28318530717958648f);
  const ::aie::vector<float, N> inv_two_pi = ::aie::broadcast<float, N>(0.15915494309189535f);
  const ::aie::vector<float, N> magic = ::aie::broadcast<float, N>(8388608.0f);  // 2^23
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> c1 = ::aie::broadcast<float, N>(-1.66666666666666657e-01f);
  const ::aie::vector<float, N> c2 = ::aie::broadcast<float, N>(8.33333333333333322e-03f);
  const ::aie::vector<float, N> c3 = ::aie::broadcast<float, N>(-1.98412698412698413e-04f);
  const ::aie::vector<float, N> c4 = ::aie::broadcast<float, N>(2.75573192239858907e-06f);
  const ::aie::vector<float, N> c5 = ::aie::broadcast<float, N>(-2.50521083854417188e-08f);

  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> x = ::aie::load_v<N>(input + i * N);

    // Fold to [-pi, pi]: r = x - 2*pi*round(x/2*pi). round() is the f32 (q + 2^23) - 2^23 identity,
    // because aie::to_float on a vector<int32,16> does not instantiate on aie2p (it fails inside
    // aie_api's own elementary.hpp, Fix2Float -> cast_to<uint16>).
    //
    // NO further range reduction and NO aie::select: a 6-term odd Taylor is accurate to 4.5e-4 over
    // the WHOLE of [-pi, pi] (checked on host over 2e5 points), so the usual reflect-into-[-pi/2,pi/2]
    // step buys nothing and would only add conditional ops whose mask polarity is one more thing to
    // get wrong. Branch-free, using only ops proven on this unit by layernorm.cc.
    //
    // aie::mul returns accum<accfloat,N>, not vector: it converts on vector initialisation but is
    // rejected as an argument to add/sub/store_v, so every product is materialised.
    ::aie::vector<float, N> q = ::aie::mul(x, inv_two_pi);
    ::aie::vector<float, N> k = ::aie::sub(::aie::add(q, magic), magic);
    ::aie::vector<float, N> kt = ::aie::mul(k, two_pi);
    ::aie::vector<float, N> r = ::aie::sub(x, kt);

    // odd Taylor, Horner in r^2. Every step is a fresh INITIALISATION from the accum, never an
    // assignment: initialisation is the only accum->vector form layernorm.cc uses, and it is the only
    // one confirmed correct on this unit. Reassigning a live vector from a mul() produced wrong
    // results here even though it compiled.
    ::aie::vector<float, N> r2 = ::aie::mul(r, r);
    ::aie::vector<float, N> m5 = ::aie::mul(c5, r2);
    ::aie::vector<float, N> p4 = ::aie::add(m5, c4);
    ::aie::vector<float, N> m4 = ::aie::mul(p4, r2);
    ::aie::vector<float, N> p3 = ::aie::add(m4, c3);
    ::aie::vector<float, N> m3 = ::aie::mul(p3, r2);
    ::aie::vector<float, N> p2 = ::aie::add(m3, c2);
    ::aie::vector<float, N> m2 = ::aie::mul(p2, r2);
    ::aie::vector<float, N> p1 = ::aie::add(m2, c1);
    ::aie::vector<float, N> m1 = ::aie::mul(p1, r2);
    ::aie::vector<float, N> p0 = ::aie::add(m1, one);
    ::aie::vector<float, N> res = ::aie::mul(p0, r);
    ::aie::store_v(output + i * N, res);
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
