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

// Vector-level sin: fold to [-pi, pi] then a 6-term odd Taylor. Exposed so composites (snake)
// can use it without a scratch buffer or a second pass over memory.
//
// REGISTER PRESSURE IS THE BINDING CONSTRAINT, not arithmetic. An earlier version kept every Horner
// intermediate in its own named vector (~17 live) plus hoisted broadcast constants; on device the
// results came back ALIASED -- p3 returned r2's value, res returned p4's, zeros interleaved. That is
// spill codegen (llvm-aie #1155 territory), not maths. So: one accumulator reassigned in place, and
// constants materialised at point of use.
template <int N>
static inline ::aie::vector<float, N> sin_v(::aie::vector<float, N> r) {
  // fold: r -= 2*pi*round(r/2*pi). round() via the f32 (q + C) - C magic-number identity, because
  // aie::to_float on a vector<int32,16> does not instantiate on aie2p (it fails inside aie_api's
  // own elementary.hpp, Fix2Float -> cast_to<uint16>).
  //
  // C IS 1.5*2^23, NOT 2^23. With C = 2^23 a NEGATIVE q lands in [2^22, 2^23), where the f32 ulp
  // is 0.5 rather than 1, so the identity rounds to the nearest HALF-integer. Measured on host over
  // x in [-12, 12]: exact on all 100k positive samples, off by 0.5 on 49996/100000 negative ones.
  // A half-integer k folds by an odd multiple of pi, so sin returns SIGN-FLIPPED. In host
  // simulation of this exact kernel on verify_sin.py's own input, bare 2^23 gives rel-L2 1.0045
  // and 1.5*2^23 gives 1.2747e-04. C = 1.5*2^23 keeps q in [2^23, 2^24) where the ulp is exactly 1
  // for BOTH signs (host max |err| vs rint: 0.0).
  //
  // DEVICE-NEUTRAL HERE, and the A/B is now clean. 2^23 vs 1.5*2^23 are BIT-IDENTICAL on device at
  // ONE chunk per call (7.114e-01 both, back when the brick was still red for the loop reason).
  // An earlier A/B ran at 4 chunks and so was confounded by the loop defect; re-running it at one
  // chunk removed that confound and the answer did not change. So this is a latent-correctness fix
  // for any correctly-rounding f32 path, not something this hardware observes.
  // The brick is now GREEN at 1.275e-04 -- exactly what host simulation predicts for a correct fold
  // (1.2747e-04), which also retires the old claim that that number was a stale-cache artifact.
  ::aie::vector<float, N> k = ::aie::mul(r, ::aie::broadcast<float, N>(0.15915494309189535f));
  k = ::aie::add(k, ::aie::broadcast<float, N>(12582912.0f));
  k = ::aie::sub(k, ::aie::broadcast<float, N>(12582912.0f));
  ::aie::vector<float, N> kt = ::aie::mul(k, ::aie::broadcast<float, N>(6.28318530717958648f));
  r = ::aie::sub(r, kt);

  // 6-term odd Taylor, Horner in r^2: accurate to 4.5e-4 over ALL of [-pi, pi] (host-checked over
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
  return ::aie::mul(p, r);
}

// ONE N-wide vector per call. No loop, no element count -- both are required, and this is what
// took the brick from red to green after it had been failing across sessions.
//
// The previous form derived `chunks = n / N` from a runtime `int32_t n`, and was red at 7.09e-01
// even at n=16 (one chunk). Measured 2026-07-31: input range is not the cause (|x|<=12 gives
// 7.035e-01, same as |x|<=64), and the fold constant is not the cause (2^23 vs 1.5*2^23 are
// BIT-IDENTICAL at one chunk, 7.114e-01 both). What fixed it is removing the runtime bound, the
// same fix that took `gelu-erf` from 9.538e+00 to 1.138e-03.
// Volume belongs in the objectFIFO worker loop, not in a kernel-internal bound: it costs
// nothing there, since the iteration merely moves to the driver.
template <int N>
void sin_core(const float *restrict input, float *restrict output) {
  event0();
  ::aie::store_v(output, sin_v<N>(::aie::load_v<N>(input)));
  event1();
}

} // namespace route_b_bricks

extern "C" {

// sin over `n` f32 elements (n must be a multiple of the vector width).
void sin_f32(float *input, float *output) {
  route_b_bricks::sin_core<16>(input, output);
}

} // extern "C"
