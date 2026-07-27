//===- ln_affine_cast.cc --------------------------------------*- C++ -*-===//
//
// FUSED normalize-only two-pass LayerNorm + affine + f32->bf16 cast, in ONE kernel (one dispatch /
// one hw-context). Replaces the ctxLN (ln_2pass) -> affine_cast two-kernel chain, deleting the
// inter-kernel shape-reload on every LN site (FFN + conv). Output = affine_LN(x) as bf16, ready as
// the modal fc1's A input -- byte-identical intent to running ln_2pass then affine_cast.
//
//   per row of `cols`:
//     mean = Sx / cols ; var = S(x-mean)^2 / cols (two-pass, centered) ; inv = 1/sqrt(var+eps)
//     out  = ((x - mean) * inv) * gamma + beta   -> bf16
//
// gamma/beta are the SAME for every row, packed into ONE [2*cols] param buffer gb =
// [gamma(0..cols) | beta(cols..2cols)] on ONE DMA input channel (x + gb = 2 inputs, the AIE2
// compute-tile input-DMA limit). cols % N == 0.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

template <int N>
void ln_affine_cast_row(const float *restrict input, const float *restrict gb,
                        bfloat16 *restrict output, int32_t cols) {
  event0();
  // NOTE: set_rounding(conv_even) is applied ONLY around the affine+cast write pass (below), NOT here.
  // The ctxLN->affcast chain we replace runs the LN reductions (sum/var/invsqrt) under the AIE DEFAULT
  // rounding (ctxLN never sets a mode) and only affcast uses conv_even for its bf16 narrowing. Setting
  // conv_even up front changed the reduction rounding -> diverged from the chain -> WER regressed
  // 8.2->8.8 (compounded over 24 layers). Keep the reductions bit-matched to ctxLN.
  constexpr float epsilon = 1e-5f;
  const int chunks = cols / N;
  const float *gamma = gb;
  const float *beta = gb + cols;

  // pass 1: mean = Sx / cols
  ::aie::vector<float, N> sum_v = ::aie::zeros<float, N>();
  for (int i = 0; i < chunks; i++)
    sum_v = ::aie::add(sum_v, ::aie::load_v<N>(input + i * N));
  float mean = ::aie::reduce_add(sum_v) / float(cols);
  ::aie::vector<float, N> mean_v = ::aie::broadcast<float, N>(mean);

  // pass 2: var = S(x-mean)^2 / cols  (centered two-pass -- matches the host reference)
  ::aie::vector<float, N> var_v = ::aie::zeros<float, N>();
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<float, N> sq = ::aie::mul(d, d);
    var_v = ::aie::add(var_v, sq);
  }
  float var = ::aie::reduce_add(var_v) / float(cols);
  float inv_std = ::aie::invsqrt(var + epsilon);
  ::aie::vector<float, N> inv_v = ::aie::broadcast<float, N>(inv_std);

  // write: out = ((x - mean) * inv) * gamma + beta -> bf16. conv_even ONLY here (the bf16 narrowing +
  // affine), matching affcast; the reductions above ran under the default mode, matching ctxLN.
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<float, N> norm = ::aie::mul(d, inv_v);            // normalized f32
    ::aie::vector<float, N> g = ::aie::load_v<N>(gamma + i * N);
    ::aie::vector<float, N> b = ::aie::load_v<N>(beta + i * N);
    ::aie::vector<float, N> ng = ::aie::mul(norm, g);              // norm * gamma
    ::aie::vector<float, N> y = ::aie::add(ng, b);                 // + beta
    ::aie::accum<accfloat, N> a;
    a.from_vector(y);
    ::aie::store_v(output + i * N, a.template to_vector<bfloat16>());
  }
  event1();
}

extern "C" {
void ln_affine_cast_row(float *input, float *gb, bfloat16 *output, int32_t cols) {
  ln_affine_cast_row<16>(input, gb, output, cols);
}
}
