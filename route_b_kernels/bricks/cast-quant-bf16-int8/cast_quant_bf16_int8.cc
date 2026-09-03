//===- cast_quant_bf16_int8.cc --------------------------------*- C++ -*-===//
//
// Device-side format brick: bf16<->int8 symmetric quantize/dequantize (with a
// per-tensor scale) and bf16<->f32 widen/narrow. Generic op-TYPE against the
// resident [tile,D] stream contract (per-row core_body, cols % N == 0) -- NOT
// a per-model clone. Group: format.
//
// Quant convention (symmetric, per-tensor scale, matches a plain numpy
// reference -- see golden.py):
//   int8   = clamp(round(bf16 / scale), -128, 127)
//   bf16   = int8 * scale
//
// aie_api idioms used here are the SAME ones already proven in this tree:
//   * f32 -> bf16 narrow needs an accumulator: vector<float> has no
//     to_vector<bfloat16>; accum<accfloat> does (see cast_f32_bf16.cc).
//   * bf16 -> f32 widen also goes through an accumulator: accum::from_vector
//     accepts a bfloat16 vector directly (see mha_decode.cc,
//     `va.from_vector(aie::load_v<VL>(vrow ...))` widening bf16 -> f32) --
//     plain vector<bfloat16,N> has no standalone to_vector<float>() member.
//   * accum<accfloat,N>::to_vector<T>() is documented as "shift-round-
//     saturate" for the requested type/signedness -- reused here to go
//     straight from a scaled f32 accumulator to a saturated int8_t vector
//     (round-to-nearest + clamp to [-128,127] in one step, no separate
//     aie::min/aie::max needed).
//   * int8 -> f32 widen uses the elementary aie::to_float(vector<int8,N>)
//     conversion (aie.hpp group_fp_conversion).
//
// Four entry points, one row of `cols` elements each:
//   quantize_bf16_to_int8_row   : bf16 -> int8   (needs `scale`)
//   dequantize_int8_to_bf16_row : int8 -> bf16   (needs `scale`)
//   cast_bf16_to_f32_row        : bf16 -> f32
//   cast_f32_to_bf16_row        : f32  -> bf16
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// ---------------------------------------------------------------------------
// bf16 -> int8, symmetric per-tensor scale: q = round(x / scale), clamped to
// the int8 range by the accumulator's shift-round-saturate to_vector<int8_t>.
// ---------------------------------------------------------------------------
template <int N>
void quantize_bf16_to_int8_row(const bfloat16 *restrict input,
                               int8_t *restrict output, float scale,
                               int32_t cols) {
  event0();
  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const float inv_scale = 1.0f / scale;
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<bfloat16, N> v = ::aie::load_v<N>(input + i);
    ::aie::accum<accfloat, N> aw;
    aw.from_vector(v); // widen bf16 -> f32
    ::aie::vector<float, N> vf = aw.template to_vector<float>();
    ::aie::vector<float, N> scaled = ::aie::mul(vf, inv_scale); // x / scale
    // Round-to-nearest + saturate to signed int8 [-128,127]. Use to_fixed (the
    // float->int vector convert honoring the current rounding mode); the previous
    // accfloat accumulator -> to_vector<int8_t> path reinterpreted the float bits
    // as fixed-point and produced garbage (device rel-L2 2.377 vs the golden).
    ::aie::store_v(output + i, ::aie::to_fixed<int8_t>(scaled));
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

// ---------------------------------------------------------------------------
// int8 -> bf16, symmetric per-tensor scale: x = q * scale.
// ---------------------------------------------------------------------------
template <int N>
void dequantize_int8_to_bf16_row(const int8_t *restrict input,
                                 bfloat16 *restrict output, float scale,
                                 int32_t cols) {
  event0();
  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<int8_t, N> q = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> qf = ::aie::to_float(q);  // widen int8 -> f32
    ::aie::vector<float, N> vf = ::aie::mul(qf, scale); // q * scale
    ::aie::accum<accfloat, N> a;
    a.from_vector(vf);
    ::aie::store_v(output + i, a.template to_vector<bfloat16>()); // narrow -> bf16
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

// ---------------------------------------------------------------------------
// bf16 -> f32 widen (no scale; a direct format cast).
// ---------------------------------------------------------------------------
template <int N>
void cast_bf16_to_f32_row(const bfloat16 *restrict input,
                          float *restrict output, int32_t cols) {
  event0();
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<bfloat16, N> v = ::aie::load_v<N>(input + i);
    ::aie::accum<accfloat, N> a;
    a.from_vector(v); // widen bf16 -> f32
    ::aie::store_v(output + i, a.template to_vector<float>());
  }
  event1();
}

// ---------------------------------------------------------------------------
// f32 -> bf16 narrow (round-to-nearest-even via accumulator; no scale).
// ---------------------------------------------------------------------------
template <int N>
void cast_f32_to_bf16_row(const float *restrict input,
                          bfloat16 *restrict output, int32_t cols) {
  event0();
  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> v = ::aie::load_v<N>(input + i);
    ::aie::accum<accfloat, N> a;
    a.from_vector(v);
    ::aie::store_v(output + i, a.template to_vector<bfloat16>());
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void quantize_bf16_to_int8_row(bfloat16 *input, int8_t *output, float scale,
                               int32_t cols) {
  quantize_bf16_to_int8_row<16>(input, output, scale, cols);
}
void dequantize_int8_to_bf16_row(int8_t *input, bfloat16 *output, float scale,
                                 int32_t cols) {
  dequantize_int8_to_bf16_row<16>(input, output, scale, cols);
}
void cast_bf16_to_f32_row(bfloat16 *input, float *output, int32_t cols) {
  cast_bf16_to_f32_row<16>(input, output, cols);
}
void cast_f32_to_bf16_row(float *input, bfloat16 *output, int32_t cols) {
  cast_f32_to_bf16_row<16>(input, output, cols);
}
}
