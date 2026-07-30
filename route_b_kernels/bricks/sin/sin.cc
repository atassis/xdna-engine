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
//   2. wrap by +/-2*pi (safety net; stage 1 already lands in range)
//   3. reflect about +/-pi/2, since sin(pi - t) = sin(t)  -> [-pi/2, pi/2]
// Then the odd Taylor polynomial r*(1 + c1 r^2 + c2 r^4 + c3 r^6), exact to ~1e-7 on that interval,
// so the error budget is dominated by the f32 fold rather than by truncation order.
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
  const ::aie::vector<float, N> pi_v = ::aie::broadcast<float, N>(3.14159265358979324f);
  const ::aie::vector<float, N> neg_pi_v = ::aie::broadcast<float, N>(-3.14159265358979324f);
  const ::aie::vector<float, N> half_pi = ::aie::broadcast<float, N>(1.57079632679489662f);
  const ::aie::vector<float, N> neg_half_pi = ::aie::broadcast<float, N>(-1.57079632679489662f);
  const ::aie::vector<float, N> c1 = ::aie::broadcast<float, N>(-0.16666666666666666f);
  const ::aie::vector<float, N> c2 = ::aie::broadcast<float, N>(0.00833333333333333f);
  const ::aie::vector<float, N> c3 = ::aie::broadcast<float, N>(-0.00019841269841270f);
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> magic = ::aie::broadcast<float, N>(8388608.0f);  // 2^23

  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> x = ::aie::load_v<N>(input + i * N);

    // stage 1: drop whole periods. NOTE: no int round-trip -- aie::to_float on a vector<int32,16>
    // fails to instantiate inside aie_api's own elementary.hpp on aie2p (Fix2Float ->
    // cast_to<uint16>). Instead (q + 2^23) - 2^23 rounds q to nearest in pure f32, valid for
    // |q| < 2^22 which covers any argument this brick will see. IEEE-exact, so the compiler cannot
    // fold it away without -ffast-math (which this build does not use).
    // NOTE: aie::mul returns an accum<accfloat,N>, not a vector. It converts on initialisation of
    // a vector (as layernorm.cc relies on) but is NOT accepted as an argument to sub/add/store_v, so
    // every product is materialised into a named vector first.
    ::aie::vector<float, N> q = ::aie::mul(x, inv_two_pi);
    ::aie::vector<float, N> k = ::aie::sub(::aie::add(q, magic), magic);
    ::aie::vector<float, N> kt = ::aie::mul(k, two_pi);
    ::aie::vector<float, N> r = ::aie::sub(x, kt);

    // stage 2: wrap into [-pi, pi] (safety net; stage 1 already lands in range)
    r = ::aie::select(r, ::aie::sub(r, two_pi), ::aie::gt(r, pi_v));
    r = ::aie::select(r, ::aie::add(r, two_pi), ::aie::lt(r, neg_pi_v));

    // stage 3: reflect into [-pi/2, pi/2] using sin(pi - t) = sin(t)
    r = ::aie::select(r, ::aie::sub(pi_v, r), ::aie::gt(r, half_pi));
    r = ::aie::select(r, ::aie::sub(neg_pi_v, r), ::aie::lt(r, neg_half_pi));

    // odd Taylor polynomial, Horner in r^2
    ::aie::vector<float, N> r2 = ::aie::mul(r, r);
    ::aie::vector<float, N> t0 = ::aie::mul(c3, r2);
    ::aie::vector<float, N> p = ::aie::add(t0, c2);
    ::aie::vector<float, N> t1 = ::aie::mul(p, r2);
    p = ::aie::add(t1, c1);
    ::aie::vector<float, N> t2 = ::aie::mul(p, r2);
    p = ::aie::add(t2, one);
    ::aie::vector<float, N> res = ::aie::mul(p, r);
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
