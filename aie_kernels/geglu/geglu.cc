//===- geglu.cc -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GENERIC "act" brick: GeGLU (Gated GELU Linear Unit).
//
//   out[t, c] = GELU(gate[t, c]) * up[t, c]
//
// where the fused-fc1 output row is laid out as [up | gate]:
//   in[0 .. cols)         = up   ("value" half)
//   in[cols .. 2*cols)    = gate ("gate" half)
// i.e. the SAME [up|gate] concatenated-row convention as glu.cc's [a|g], just
// with GELU instead of sigmoid on the gate half. This is the op-TYPE a caller
// gets by choosing `act=gelu` in a GLU-family brick (silu -> glu.cc's
// sigmoid-gate SwiGLU, gelu -> this GeGLU) against the SAME resident
// [tile, D] row-stream contract -- NOT a per-model clone.
//
// GELU uses the tanh approximation (matches torch gelu(approximate="tanh") and
// mm_silu_epilogue.cc's mm_gelu_epilogue_f32o):
//   gelu(x) = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
//
// NUMERICS: mirrors glu.cc's higher-precision hybrid -- the tanh ARGUMENT and
// the cubic polynomial are computed in f32 (the un-rounded accumulator value),
// only the tanh OUTPUT is narrowed to bf16 (bounded in [-1,1], bf16 is
// accurate enough there), and the FINAL gelu*up multiply happens in f32. A
// full-f32 tanh is known (mm_silu_epilogue.cc note, 2026-06-28) to blow the
// on-chip cycle budget / hang; this hybrid is the proven-safe middle ground.
//
// One call: ONE row of `cols` (the per-row core_body contract, mirroring
// affine_cast.cc / ln_2pass.cc / glu.cc). 2*cols is the input row width;
// cols % N == 0.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>
#include <stdint.h>

// Generic GeGLU row op: out = gelu(gate) * up, over `cols` elements, N-wide.
// Parameterized purely on vector width N so it composes as a brick against
// any [tile, D] resident row-stream (D = cols here); no model-specific dims
// baked in.
template <int N>
void geglu_row(const float *restrict input, float *restrict output,
               int32_t cols) {
  event0();
  // Round-to-nearest-even on the bf16 narrowings (banked WER rule: bf16 casts
  // on the numerics-critical path must round-nearest, not truncate).
  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  const float *up = input;          // value half:  in[0..cols)
  const float *gate = input + cols; // gate  half:  in[cols..2*cols)

  const ::aie::vector<float, N> halff = ::aie::broadcast<float, N>(0.5f);
  const ::aie::vector<bfloat16, N> one = ::aie::broadcast<bfloat16, N>(1.0f);
  const ::aie::vector<bfloat16, N> c0 =
      ::aie::broadcast<bfloat16, N>(0.7978845608f); // sqrt(2/pi)
  const ::aie::vector<bfloat16, N> c1 =
      ::aie::broadcast<bfloat16, N>(0.044715f);

  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> upv = ::aie::load_v<N>(up + i);
    ::aie::vector<float, N> gv = ::aie::load_v<N>(gate + i);

    // Narrow gate to bf16 for the cubic term (bf16-native cheap path; mirrors
    // mm_gelu_epilogue_f32o's x^2/x^3 in bf16).
    ::aie::accum<accfloat, N> gacc;
    gacc.from_vector(gv);
    ::aie::vector<bfloat16, N> gb = gacc.template to_vector<bfloat16>();

    ::aie::vector<bfloat16, N> g2 = ::aie::mul(gb, gb);   // g^2
    ::aie::vector<bfloat16, N> g3 = ::aie::mul(g2, gb);   // g^3
    ::aie::vector<bfloat16, N> c1g3 = ::aie::mul(c1, g3); // 0.044715*g^3
    ::aie::vector<bfloat16, N> inner_b = ::aie::add(gb, c1g3); // g + 0.044715*g^3

    // Keep the tanh ARGUMENT scale-up in f32-derived precision: up-convert
    // inner_b back through f32 before the sqrt(2/pi) scale, then tanh -- the
    // tanh OUTPUT is bf16 (bounded), the cubic polynomial stays bf16-native
    // (cheap, matches the proven mm_gelu_epilogue_f32o budget).
    auto inner = ::aie::mul(c0, inner_b); // accum: sqrt(2/pi)*(g + 0.044715*g^3)
    ::aie::vector<bfloat16, N> t =
        ::aie::tanh<bfloat16>(inner.template to_vector<float>());
    ::aie::vector<bfloat16, N> t_p1 = ::aie::add(t, one); // 1 + tanh(...)

    // up-convert (1+tanh) to f32; final 0.5 * gate * up * (1+tanh) multiply
    // chain runs in f32 (the un-rounded gate/up values), matching glu.cc's
    // hybrid-precision final multiply.
    ::aie::accum<accfloat, N> tacc;
    tacc.from_vector(t_p1);
    ::aie::vector<float, N> t_p1f = tacc.template to_vector<float>();

    ::aie::vector<float, N> half_g = ::aie::mul(gv, halff); // 0.5*g
    ::aie::vector<float, N> gelu_g = ::aie::mul(half_g, t_p1f); // 0.5*g*(1+tanh)
    ::aie::vector<float, N> outv = ::aie::mul(upv, gelu_g);     // up * gelu(gate)
    ::aie::store_v(output + i, outv);
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void geglu_row(float *input, float *output, int32_t cols) {
  geglu_row<16>(input, output, cols);
}
}
