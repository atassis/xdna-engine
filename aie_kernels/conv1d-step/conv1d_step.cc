//===- conv1d_step.cc --------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// One decode step of a depthwise causal conv1d with a carried input history (op-type
// `conv1d{depthwise, causal, step}`: the Mamba / Gated DeltaNet short conv):
//
//   y[c]         = sum_{j<K-1} w[j][c] * hist[j][c] + w[K-1][c] * x[c]
//   hist_out[j]  = hist[j+1] for j < K-2,  hist_out[K-2] = x
//
// hist is oldest-first, w is tap-major so every tap is one C-wide vector. bf16 operands with an
// accfloat accumulator (aie2p has no native f32 vector MAC, see conv-1d/conv_1d_bf16.cc) and one
// bf16 round on store. No activation: the model's SiLU is its own op.
//
// C and K are compile-time so every loop bound is a literal
// (kb/kernel-internal-loops-miscompile-put-volume-in-the-worker). No static L1 state.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef CS_C
#define CS_C 256
#endif
#ifndef CS_K
#define CS_K 4
#endif

static constexpr unsigned L = 32;
static_assert(CS_C % L == 0, "channel count must be a multiple of the 32-lane bf16 vector");
static_assert(CS_K >= 2, "a one-tap conv has no history to carry");

template <unsigned C, unsigned K>
static inline void conv1d_step_core(const bfloat16 *__restrict x, const bfloat16 *__restrict hist,
                                    const bfloat16 *__restrict w, bfloat16 *__restrict y,
                                    bfloat16 *__restrict hist_out) {
  event0();
  const auto saved = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  for (unsigned c = 0; c < C; c += L) {
    aie::vector<bfloat16, L> xv = aie::load_v<L>(&x[c]);
    aie::accum<accfloat, L> acc = aie::mul(aie::load_v<L>(&w[(K - 1) * C + c]), xv);
    for (unsigned j = 0; j < K - 1; ++j)
      acc = aie::mac(acc, aie::load_v<L>(&w[j * C + c]), aie::load_v<L>(&hist[j * C + c]));
    aie::store_v(&y[c], acc.template to_vector<bfloat16>());
    for (unsigned j = 0; j + 1 < K - 1; ++j)
      aie::store_v(&hist_out[j * C + c], aie::load_v<L>(&hist[(j + 1) * C + c]));
    aie::store_v(&hist_out[(K - 2) * C + c], xv);
  }
  ::aie::set_rounding(saved);
  event1();
}

extern "C" {

void conv1d_step(bfloat16 *x, bfloat16 *hist, bfloat16 *w, bfloat16 *y, bfloat16 *hist_out) {
  conv1d_step_core<CS_C, CS_K>(x, hist, w, y, hist_out);
}
}
