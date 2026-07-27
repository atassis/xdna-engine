//===- ln_affine_f32.cc ---------------------------------------*- C++ -*-===//
//
// FUSED normalize-only two-pass LayerNorm + affine, f32 OUT, in ONE kernel. The f32-out twin of
// ln_affine_cast.cc: identical math, but it STORES f32 instead of narrowing to bf16.
//
// This is the BLOCK-EXIT LN for the block-to-block resident stream. The block boundary is f32 in the
// host reference, and the next block's entry LN consumes f32, so emitting bf16 here would both break
// the dtype contract and round the residual stream once per block (24x per clip). Emitting f32 keeps
// the boundary exactly as the reference has it.
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
void ln_affine_f32_row(const float *restrict input, const float *restrict gb,
                       float *restrict output, int32_t cols) {
  event0();
  // NOTE: this kernel sets NO rounding mode at all. Its bf16-out sibling (ln_affine_cast.cc) needs
  // conv_even around its narrowing store, and its header records that hoisting conv_even above the
  // reductions changed their rounding and regressed WER 8.2->8.8 over 24 layers. Here there is no
  // narrowing, so the whole kernel stays in f32 under the AIE default mode -- which is also what the
  // host reference layernorm does.
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

  // write: out = ((x - mean) * inv) * gamma + beta -> f32. NO rounding-mode change: there is no
  // narrowing here, so the conv_even that ln_affine_cast needs for its bf16 store has nothing to do.
  // Reductions and affine both stay in f32 under the default mode.
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<float, N> norm = ::aie::mul(d, inv_v);            // normalized f32
    ::aie::vector<float, N> g = ::aie::load_v<N>(gamma + i * N);
    ::aie::vector<float, N> b = ::aie::load_v<N>(beta + i * N);
    ::aie::vector<float, N> ng = ::aie::mul(norm, g);              // norm * gamma
    ::aie::vector<float, N> y = ::aie::add(ng, b);                 // + beta
    ::aie::store_v(output + i * N, y);
  }
  event1();
}

extern "C" {
void ln_affine_f32_row(float *input, float *gb, float *output, int32_t cols) {
  ln_affine_f32_row<8>(input, gb, output, cols);
}
}
