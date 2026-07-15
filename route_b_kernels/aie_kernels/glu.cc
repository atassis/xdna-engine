//===- glu.cc -------------------------------------------------*- C++ -*-===//
//
// Device-side GLU (Gated Linear Unit) for the Conformer conv module.
//
// out[t, c] = a[t, c] * sigmoid(g[t, c]),  where the pointwise_conv1 output row
// is [a | g] = [in(0..cols) | in(cols..2cols)] (a = "value" half, g = "gate" half).
// Consumes pw1's on-chip [T, 2*cols] f32 stream and emits [T, cols] f32 -- the
// GLU frontier step: the activation never touches host across pw1 -> GLU.
//
// One call: ONE row of `cols` (the per-row core_body contract, mirroring
// affine_cast.cc / ln_2pass.cc). 2*cols is the input row width; cols % N == 0.
//
// NUMERICS (banked, resident-rails-ffn WER gate): sigmoid via the bf16 tanh
// identity but with the tanh ARGUMENT kept in f32 and the FINAL a*sigmoid
// multiply in f32 -- the higher-precision hybrid from mm_silu_epilogue.cc's
// mm_silu_epilogue_f32o_hiprec (a full-f32 tanh blows the cycle budget and hangs;
// rounding a/x to bf16 before the multiply cost ~+0.3 WER). sigmoid stays bf16
// (bounded in [0,1], so bf16 is accurate enough).
//   sigmoid(g) = 0.5 * (1 + tanh(g/2))
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

template <int N>
void glu_row(const float *restrict input, float *restrict output, int32_t cols) {
  event0();
  // Round-to-nearest-even on the bf16 narrowings (banked: WER-path bf16 casts
  // MUST round-nearest; default truncation biases toward zero and regressed WER).
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const float *a = input;        // value half:  in[0..cols]
  const float *g = input + cols; // gate  half:  in[cols..2cols]
  const ::aie::vector<float, N> halff = ::aie::broadcast<float, N>(0.5f);
  const ::aie::vector<bfloat16, N> one = ::aie::broadcast<bfloat16, N>(1.0f);
  const ::aie::vector<bfloat16, N> halfb = ::aie::broadcast<bfloat16, N>(0.5f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> av = ::aie::load_v<N>(a + i);
    ::aie::vector<float, N> gv = ::aie::load_v<N>(g + i);
    // sigmoid(g): keep g/2 in f32 (un-rounded), only the tanh OUTPUT is bf16.
    ::aie::vector<float, N> half_g = ::aie::mul(gv, halff);
    ::aie::vector<bfloat16, N> tanh_half_g = ::aie::tanh<bfloat16>(half_g);
    ::aie::vector<bfloat16, N> tanh_p1 = ::aie::add(tanh_half_g, one);
    ::aie::vector<bfloat16, N> sig = ::aie::mul(tanh_p1, halfb); // bf16 sigmoid in [0,1]
    // up-convert sigmoid to f32; final multiply uses the UN-rounded f32 a.
    ::aie::accum<accfloat, N> sacc;
    sacc.from_vector(sig);
    ::aie::vector<float, N> sigf = sacc.template to_vector<float>();
    ::aie::vector<float, N> outv = ::aie::mul(av, sigf);
    ::aie::store_v(output + i, outv);
  }
  event1();
}

extern "C" {
void glu_row(float *input, float *output, int32_t cols) {
  glu_row<16>(input, output, cols);
}
}
