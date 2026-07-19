//===- rope_lut.cc --------------------------------------------*- C++ -*-===//
//
// Generic RoPE (rotary position embedding) brick, applied as a Q/K prologue
// over a resident [tile,D] stream, using aie::parallel_lookup gather-LUT
// hardware for sin/cos instead of a per-element transcendental call.
//
// This is a generic op-TYPE (position x rotary), not a per-model kernel:
//   - ROPE_D    : total feature dim per row (head_dim)
//   - ROPE_ROT  : rotary width <= ROPE_D (PARTIAL rotary; dims
//                 [ROPE_ROT, ROPE_D) pass through untouched, in place)
//   - ROPE_M    : resident tile rows (tokens) this call processes
//   - ROPE_SCALE_INV : compile-time linear position-SCALING knob
//                 (effective_pos = pos * ROPE_SCALE_INV; default 1.0f is a
//                 no-op; NTK-linear schemes fold 1/scale_factor here)
//
// Split-half rotation (GPT-NeoX / Llama / Gemma style): pairs
// (i, i + ROPE_ROT/2) for i in [0, ROPE_ROT/2) rotate by
// theta[m,i] = pos[m] * ROPE_SCALE_INV * inv_freq[i]:
//
//   out[i]            = x[i]*cos(theta) - x[i+ROT/2]*sin(theta)
//   out[i+ROPE_ROT/2] = x[i]*sin(theta) + x[i+ROT/2]*cos(theta)
//
// inv_freq[] (length ROPE_ROT/2) is a FIXED function of (head_dim, theta
// base, NTK scaling config) -- it does not depend on the runtime position,
// so like norm_gemv_prologue.cc folding gamma host-side, it is folded
// host-side once at model-load time and passed in as a resident constant
// buffer (see golden.py for the reference generator). Only `pos[]` (the
// KV-cache write offset, decode M=1 or prefill M=tile) is a genuine
// runtime input.
//
// sin/cos values themselves come from a 256-entry bf16 gather-LUT indexed
// by an int8 key: key = round(wrapped_theta * 128/pi), key in [-128,127],
// bias=128 maps the full signed int8 range onto the full LUT (see
// aie::lut<4,bfloat16> layout requirement in aie_api's group_lut docs).
// The angle-wrap + quantization is plain float/int arithmetic (no libm)
// so the whole kernel needs no on-device sin/cos/exp/log call at all --
// the LUT read (SFU parallel-lookup) is the only place trig enters.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef ROPE_D
#define ROPE_D 128            // head_dim (feature width of one Q/K row)
#endif
#ifndef ROPE_ROT
#define ROPE_ROT ROPE_D       // rotary width; ROPE_ROT <= ROPE_D, must be even
#endif
#ifndef ROPE_M
#define ROPE_M 64             // resident tile rows this call processes
#endif
#ifndef ROPE_SCALE_INV
#define ROPE_SCALE_INV 1.0f   // 1/rope_scale_factor; 1.0 == no NTK scaling
#endif

static_assert(ROPE_ROT <= ROPE_D, "ROPE_ROT must be <= ROPE_D (partial rotary)");
static_assert(ROPE_ROT % 2 == 0, "ROPE_ROT must be even (paired split-half rotation)");

namespace {

constexpr unsigned kRotHalf = ROPE_ROT / 2;
// Fetch granularity of aie::parallel_lookup for a bf16-valued LUT (see
// output_lanes(sizeof(bfloat16)==2) == 512/8/2 == 32 in aie2/parallel_lookup.hpp).
constexpr unsigned kFetchW = 32;
constexpr unsigned kRotHalfPad = ((kRotHalf + kFetchW - 1) / kFetchW) * kFetchW;
constexpr unsigned kVec = 16;  // f32/bf16 elementwise apply width (matches norm_gemv_prologue.cc)
constexpr float kPi = 3.14159265358979323846f;  // avoid relying on libm M_PI macro

#include "rope_lut_tables.inc"

}  // namespace

extern "C" {

// qk        : [ROPE_M, ROPE_D] bf16, resident, rotated IN PLACE.
// pos       : [ROPE_M] int32, per-row absolute sequence position (runtime;
//             this is the one input that cannot be folded host-side).
// inv_freq  : [ROPE_ROT/2] float32, host-folded base^(-2i/ROPE_ROT) (optionally
//             pre-scaled for a fixed NTK config); resident constant buffer.
void rope_lut_prologue(bfloat16 *restrict qk, const int32_t *restrict pos,
                        const float *restrict inv_freq) {
  event0();

  const ::aie::lut<4, bfloat16> sin_lut(256, kSinLutAb, kSinLutCd);
  const ::aie::lut<4, bfloat16> cos_lut(256, kCosLutAb, kCosLutCd);
  // bias = elems>>1 = 128: the full signed int8 range indexes the whole LUT,
  // so no out-of-range branch is exercised at all (see check_index_coverage
  // in aie2/parallel_lookup.hpp -- this is the "all_space_used_" fast path).
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> sin_look(sin_lut, /*step_bits=*/0, /*bias=*/128);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> cos_look(cos_lut, /*step_bits=*/0, /*bias=*/128);

  int8_t key_buf[kRotHalfPad];
  bfloat16 sin_buf[kRotHalfPad];
  bfloat16 cos_buf[kRotHalfPad];

  for (unsigned m = 0; m < ROPE_M; ++m) {
    const float posf = (float)pos[m] * (float)ROPE_SCALE_INV;

    // --- build the quantized angle keys for this row (plain arithmetic) ---
    for (unsigned i = 0; i < kRotHalf; ++i) {
      const float theta = posf * inv_freq[i];
      // wrap to [-pi, pi): manual round-to-nearest (no libm roundf on device)
      const float k_wrap = theta * (1.0f / (2.0f * kPi));
      const int k = (int)(k_wrap + (k_wrap >= 0.0f ? 0.5f : -0.5f));
      const float wrapped = theta - (float)k * (2.0f * kPi);
      // quantize to int8 key in [-128,127], bias=128 (full-range LUT index)
      const float q = wrapped * (128.0f / kPi);
      int key = (int)(q + (q >= 0.0f ? 0.5f : -0.5f));
      if (key > 127) key = 127;
      if (key < -128) key = -128;
      key_buf[i] = (int8_t)key;
    }
    for (unsigned i = kRotHalf; i < kRotHalfPad; ++i) key_buf[i] = 0;  // pad, unused

    // --- gather sin/cos via the LUT hardware, kFetchW keys per fetch ---
    for (unsigned i = 0; i < kRotHalfPad; i += kFetchW) {
      ::aie::vector<int8, kFetchW> keys = ::aie::load_v<kFetchW>(key_buf + i);
      ::aie::vector<bfloat16, kFetchW> s = sin_look.fetch(keys);
      ::aie::vector<bfloat16, kFetchW> c = cos_look.fetch(keys);
      ::aie::store_v(sin_buf + i, s);
      ::aie::store_v(cos_buf + i, c);
    }

    // --- apply the rotation, vectorized over kVec lanes (f32 math, bf16 store,
    //     same pattern as norm_gemv_prologue.cc's accum round-trip) ---
    bfloat16 *row = qk + (size_t)m * ROPE_D;
    for (unsigned j = 0; j < kRotHalf; j += kVec) {
      ::aie::accum<accfloat, kVec> x1a; x1a.from_vector(::aie::load_v<kVec>(row + j), 0);
      ::aie::accum<accfloat, kVec> x2a; x2a.from_vector(::aie::load_v<kVec>(row + kRotHalf + j), 0);
      ::aie::accum<accfloat, kVec> sina; sina.from_vector(::aie::load_v<kVec>(sin_buf + j), 0);
      ::aie::accum<accfloat, kVec> cosa; cosa.from_vector(::aie::load_v<kVec>(cos_buf + j), 0);

      ::aie::vector<float, kVec> x1 = x1a.to_vector<float>();
      ::aie::vector<float, kVec> x2 = x2a.to_vector<float>();
      ::aie::vector<float, kVec> s = sina.to_vector<float>();
      ::aie::vector<float, kVec> c = cosa.to_vector<float>();

      ::aie::vector<float, kVec> out1 = ::aie::sub(::aie::mul(x1, c).template to_vector<float>(),
                                                    ::aie::mul(x2, s).template to_vector<float>());
      ::aie::vector<float, kVec> out2 = ::aie::add(::aie::mul(x1, s).template to_vector<float>(),
                                                    ::aie::mul(x2, c).template to_vector<float>());

      ::aie::accum<accfloat, kVec> out1a; out1a.from_vector(out1, 0);
      ::aie::accum<accfloat, kVec> out2a; out2a.from_vector(out2, 0);
      ::aie::store_v(row + j, out1a.template to_vector<bfloat16>());
      ::aie::store_v(row + kRotHalf + j, out2a.template to_vector<bfloat16>());
    }
    // dims [ROPE_ROT, ROPE_D) are untouched: already correct in place
    // (partial rotary pass-through).
  }

  event1();
}

}  // extern "C"
