//===- affine_cast.cc -----------------------------------------*- C++ -*-===//
//
// Device-side affine + f32->bf16 cast (resident-rails LN affine seam).
//
// out = (in * gamma + beta) narrowed to bf16, per row over `cols`. Folds the LayerNorm
// learned affine (gamma,beta) onto the normalize-only ctxLN output so the modal fc1 sees
// affine_LN(x) directly and applies its on-chip SiLU with the UNMODIFIED weight (no
// modalid stream, no host bias/silu, no gamma-folded weight).
//
// gamma/beta are the SAME for every row, packed into ONE [2*cols] param buffer `gb` =
// [gamma(0..cols) | beta(cols..2cols)] and streamed on ONE DMA input channel -- an AIE2
// tile has only 2 input DMA channels, so x + gamma + beta as 3 separate inputs does NOT
// place (the 2-input-DMA wall). Broadcast to all cores, acquired once.
//
// One call: ONE row of `cols` (the ml/layernorm per-row core_body contract). cols % N == 0.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

template <int N>
void affine_cast_row(const float *restrict input, const float *restrict gb,
                     bfloat16 *restrict output, int32_t cols) {
  event0();
  // Round-to-nearest-even, matching the host AVX512 pack_f32_to_bf16 (default is truncation,
  // which biases toward zero and regressed WER vs the round-nearest host path).
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const float *gamma = gb;
  const float *beta = gb + cols;
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> v = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> g = ::aie::load_v<N>(gamma + i);
    ::aie::vector<float, N> b = ::aie::load_v<N>(beta + i);
    ::aie::vector<float, N> vg = ::aie::mul(v, g); // in*gamma (accum -> vector)
    ::aie::vector<float, N> y = ::aie::add(vg, b); // + beta
    ::aie::accum<accfloat, N> a;
    a.from_vector(y);
    ::aie::store_v(output + i, a.template to_vector<bfloat16>());
  }
  event1();
}

extern "C" {
void affine_cast_row(float *input, float *gb, bfloat16 *output, int32_t cols) {
  affine_cast_row<16>(input, gb, output, cols);
}
}
