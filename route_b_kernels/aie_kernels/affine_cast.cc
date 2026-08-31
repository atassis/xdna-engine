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
// Three wire formats, exactly ONE compiled per build (not multiple entry points), matching
// residual_add.cc's own dtype-arm discipline (mixed-precision-budget-sweep candidate #2):
//   default          input f32, gb f32, out bf16 (the shipped arm, unchanged)
//   AFFCAST_X_BF16   input bf16, gb f32, out bf16 -- narrows the inter-op [PAD_M,KRES]
//                    ctxLN->affcast STREAM (candidate #2's "2 MB f32 today" stream, the only
//                    large operand); gb stays f32.
//   AFFCAST_GB_BF16  input f32, gb bf16, out bf16 -- narrows gamma|beta instead. gb is a
//                    PER-OP PARAMETER, not a stream: the same rounded gamma[c]/beta[c] value
//                    is reused identically across all `rows`, so a narrowing error here is a
//                    per-column SYSTEMATIC bias repeated every row, not independent per-element
//                    noise the way x's rounding is -- a different numerics risk profile, gated
//                    on its own rel-L2/bit-exactness, not assumed to inherit x's result.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#if defined(AFFCAST_X_BF16)
template <int N>
void affine_cast_row(const bfloat16 *restrict input, const float *restrict gb,
                     bfloat16 *restrict output, int32_t cols) {
  event0();
  const auto saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const float *gamma = gb;
  const float *beta = gb + cols;
  for (int i = 0; i < cols; i += N) {
    // bf16 -> f32 is exact (bf16 is a truncated f32): widening input costs nothing
    // numerically, and the multiply-add runs at f32 exactly as the f32 arm does.
    ::aie::accum<accfloat, N> ia;
    ia.from_vector(::aie::load_v<N>(input + i), 0);
    ::aie::vector<float, N> v = ia.template to_vector<float>();
    ::aie::vector<float, N> g = ::aie::load_v<N>(gamma + i);
    ::aie::vector<float, N> b = ::aie::load_v<N>(beta + i);
    ::aie::vector<float, N> vg = ::aie::mul(v, g);
    ::aie::vector<float, N> y = ::aie::add(vg, b);
    ::aie::accum<accfloat, N> a;
    a.from_vector(y);
    ::aie::store_v(output + i, a.template to_vector<bfloat16>());
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void affine_cast_row(bfloat16 *input, float *gb, bfloat16 *output, int32_t cols) {
  affine_cast_row<16>(input, gb, output, cols);
}
}
#elif defined(AFFCAST_GB_BF16)
template <int N>
void affine_cast_row(const float *restrict input, const bfloat16 *restrict gb,
                     bfloat16 *restrict output, int32_t cols) {
  event0();
  const auto saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const bfloat16 *gamma = gb;
  const bfloat16 *beta = gb + cols;
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> v = ::aie::load_v<N>(input + i);
    // gamma/beta widen bf16 -> f32 exactly, same reasoning as the x arm, applied to the
    // OTHER operand: the narrowing that matters happened once on the host when gamma/beta
    // were packed, not here.
    ::aie::accum<accfloat, N> ga;
    ga.from_vector(::aie::load_v<N>(gamma + i), 0);
    ::aie::vector<float, N> g = ga.template to_vector<float>();
    ::aie::accum<accfloat, N> ba;
    ba.from_vector(::aie::load_v<N>(beta + i), 0);
    ::aie::vector<float, N> b = ba.template to_vector<float>();
    ::aie::vector<float, N> vg = ::aie::mul(v, g);
    ::aie::vector<float, N> y = ::aie::add(vg, b);
    ::aie::accum<accfloat, N> a;
    a.from_vector(y);
    ::aie::store_v(output + i, a.template to_vector<bfloat16>());
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void affine_cast_row(float *input, bfloat16 *gb, bfloat16 *output, int32_t cols) {
  affine_cast_row<16>(input, gb, output, cols);
}
}
#else
template <int N>
void affine_cast_row(const float *restrict input, const float *restrict gb,
                     bfloat16 *restrict output, int32_t cols) {
  event0();
  // Round-to-nearest-even, matching the host AVX512 pack_f32_to_bf16 (default is truncation,
  // which biases toward zero and regressed WER vs the round-nearest host path).
  // crRnd is a single sticky register shared by every kernel that runs on this core, so the mode
  // has to be handed back: leaving it set makes the NEXT kernel's narrowing depend on whether this
  // one happened to run first, which is invisible in every artifact we diff.
  const auto saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
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
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void affine_cast_row(float *input, float *gb, bfloat16 *output, int32_t cols) {
  affine_cast_row<16>(input, gb, output, cols);
}
}
#endif  // wire-format arm
