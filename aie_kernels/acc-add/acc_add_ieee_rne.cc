//===- acc_add_ieee_rne.cc -------------------------------------*- C++ -*-===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Binary32 addition, round-to-nearest ties-to-even, entirely in integer SIMD.
// Unlike native AIE float add, alignment retains guard/round/sticky bits.
// Independent of crRnd; gradual underflow, signed zero and overflow are retained.
// Infinities follow IEEE addition; any NaN or opposite infinities yield canonical
// quiet NaN 0x7fc00000. No payload/sign propagation or FP exception flags promised.
//
// Row ABI: three float pointers, positive cols divisible by 16. Buffers must not
// overlap. No float arithmetic or host runtime.
#include <stdint.h>
#if defined(__AIECC__)
#include <aie_api/aie.hpp>
#endif
namespace {
using U32 = uint32_t __attribute__((vector_size(64)));
__attribute__((always_inline)) inline U32 splat(uint32_t x) { const U32 zero = {}; return zero + x; }
#if defined(__AIECC__)
using Mask = aie::mask<16>; using Vec = aie::vector<uint32_t, 16>;
__attribute__((always_inline)) inline Mask less_than(U32 a, U32 b) { return aie::lt(Vec(a), Vec(b)); }
__attribute__((always_inline)) inline Mask equal(U32 a, U32 b) { return aie::eq(Vec(a), Vec(b)); }
__attribute__((always_inline)) inline U32 select(Mask m, U32 y, U32 n) { return aie::select(Vec(n), Vec(y), m).to_native(); }
#else
using Mask = U32;
__attribute__((always_inline)) inline Mask less_than(U32 a, U32 b) { return (U32)(a < b); }
__attribute__((always_inline)) inline Mask equal(U32 a, U32 b) { return (U32)(a == b); }
__attribute__((always_inline)) inline U32 select(Mask m, U32 y, U32 n) { return (m & y) | (~m & n); }
#endif
template <unsigned S> __attribute__((always_inline)) inline U32 shift_right(U32 x) {
#if defined(__AIECC__)
  return aie::logical_downshift(Vec(x), S).to_native();
#else
  return x >> S;
#endif
}
template <unsigned S> __attribute__((always_inline)) inline U32 shift_left(U32 x) {
#if defined(__AIECC__)
  return aie::upshift(Vec(x & (0xffffffffu >> S)), S).to_native();
#else
  return x << S;
#endif
}
template <unsigned S> __attribute__((always_inline)) inline U32 shift_right_jam(U32 x) {
  return shift_right<S>(x) | select(~equal(x & ((1u << S) - 1), splat(0)), splat(1), splat(0));
}
template <unsigned S> __attribute__((always_inline)) inline void align_stage(U32 &x, U32 d) {
  x = select(~equal(d & S, splat(0)), shift_right_jam<S>(x), x);
}
template <unsigned S> __attribute__((always_inline)) inline void normalize_stage(U32 &x, U32 &e) {
  const Mask m = less_than(x, splat(1u << (27 - S))) & less_than(splat(S), e);
  x = select(m, shift_left<S>(x), x); e = select(m, e - S, e);
}
__attribute__((always_inline)) inline U32 add_bits(U32 a, U32 b) {
  const U32 abs_a = a & 0x7fffffffu, abs_b = b & 0x7fffffffu;
  const Mask swap = less_than(abs_a, abs_b); const U32 hi = select(swap, b, a), lo = select(swap, a, b);
  const U32 sign = hi & 0x80000000u; const Mask opposite = ~equal((hi ^ lo) & 0x80000000u, splat(0));
  const U32 hi_exp = shift_right<23>(hi) & 255u, lo_exp = shift_right<23>(lo) & 255u;
  U32 exponent = select(~equal(hi_exp, splat(0)), hi_exp, splat(1));
  const U32 lo_effective_exp = select(~equal(lo_exp, splat(0)), lo_exp, splat(1));
  const U32 distance = exponent - lo_effective_exp;
  U32 hi_sig = shift_left<3>((hi & 0x7fffffu) | select(~equal(hi_exp, splat(0)), splat(0x800000), splat(0)));
  U32 lo_sig = shift_left<3>((lo & 0x7fffffu) | select(~equal(lo_exp, splat(0)), splat(0x800000), splat(0)));
  align_stage<1>(lo_sig, distance); align_stage<2>(lo_sig, distance); align_stage<4>(lo_sig, distance);
  align_stage<8>(lo_sig, distance); align_stage<16>(lo_sig, distance);
  lo_sig = select(~less_than(distance, splat(32)), select(~equal(lo_sig, splat(0)), splat(1), splat(0)), lo_sig);
  U32 sig = select(opposite, hi_sig - lo_sig, hi_sig + lo_sig); const Mask carry = ~less_than(sig, splat(1u << 27));
  sig = select(carry, shift_right_jam<1>(sig), sig); exponent += select(carry, splat(1), splat(0));
  normalize_stage<16>(sig, exponent); normalize_stage<8>(sig, exponent); normalize_stage<4>(sig, exponent);
  normalize_stage<2>(sig, exponent); normalize_stage<1>(sig, exponent);
  U32 rounded = shift_right<3>(sig); const U32 tail = sig & 7u;
  const Mask round_up = less_than(splat(4), tail) | (equal(tail, splat(4)) & ~equal(rounded & 1u, splat(0)));
  rounded += select(round_up, splat(1), splat(0)); const Mask round_carry = ~less_than(rounded, splat(1u << 24));
  rounded = select(round_carry, shift_right<1>(rounded), rounded); exponent += select(round_carry, splat(1), splat(0));
  const U32 normal_exp = select(~less_than(rounded, splat(1u << 23)), exponent, splat(0));
  U32 result = sign | shift_left<23>(normal_exp) | (rounded & 0x7fffffu);
  result = select(~less_than(exponent, splat(255)), sign | 0x7f800000u, result);
  result = select(equal(sig, splat(0)), (a & b) & 0x80000000u, result);
  const Mask is_inf = equal(abs_a, splat(0x7f800000)) | equal(abs_b, splat(0x7f800000)); result = select(is_inf, sign | 0x7f800000u, result);
  const Mask invalid = less_than(splat(0x7f800000), abs_a) | less_than(splat(0x7f800000), abs_b) |
      (equal(abs_a, splat(0x7f800000)) & equal(abs_b, splat(0x7f800000)) & opposite);
  return select(invalid, splat(0x7fc00000u), result);
}
} // namespace
extern "C" void acc_add_row_ieee_rne(const float *__restrict a, const float *__restrict b,
                                      float *__restrict out, int32_t cols) {
  for (int32_t i = 0; i < cols; i += 16) { U32 av, bv; __builtin_memcpy(&av, a + i, sizeof(av));
    __builtin_memcpy(&bv, b + i, sizeof(bv)); const U32 result = add_bits(av, bv);
    __builtin_memcpy(out + i, &result, sizeof(result)); }
}
