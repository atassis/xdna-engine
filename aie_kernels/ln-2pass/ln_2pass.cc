//===- ln_2pass.cc --------------------------------------------*- C++ -*-===//
//
// STEP C / Step D — per-row f32 two-pass LayerNorm (NORMALIZE-ONLY) for the encoder ctxLN.
//
// Matches the HOST reference exactly (npu-asr-host/src/lib.rs `layer_norm_normalize`):
//   per row of `cols` values:
//     mean = Σx / cols
//     var  = Σ(x - mean)² / cols          (TWO-PASS, exact-mean-centered — NOT E[x²]-mean²)
//     inv  = 1 / sqrt(var + eps)           (eps = 1e-5)
//     out  = (x - mean) * inv              (affine γ,β applied on the host for the 4 affine sites)
//
// WHY two-pass f32 and not the shipped kernels (internal notes §3, the load-bearing accuracy note):
//   * `layer_norm` (bf16) uses var = E[x²]-mean² with a bf16 sum → catastrophic cancellation,
//     measured 5.77% block rel (docs/08). REJECTED.
//   * `layer_norm_welford` is f32-stable BUT reduces over the ROWS axis (per-column stats); the
//     encoder LN reduces over the `cols` (D=768) axis per row. WRONG AXIS. REJECTED.
//   This kernel is f32, reduces over `cols` per row (correct axis), centered two-pass (stable).
//
// One call normalizes ONE row of `cols` elements (the ml/layernorm per-row core_body contract).
// f32 in / f32 out (docs/05 "never re-expand"). cols must be a multiple of N (16).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

template <int N>
void layer_norm_2pass_f32(const float *restrict input, float *restrict output,
                          int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const int chunks = cols / N;

  // pass 1: mean = Σx / cols
  ::aie::vector<float, N> sum_v = ::aie::zeros<float, N>();
  for (int i = 0; i < chunks; i++) {
    sum_v = ::aie::add(sum_v, ::aie::load_v<N>(input + i * N));
  }
  float mean = ::aie::reduce_add(sum_v) / float(cols);
  ::aie::vector<float, N> mean_v = ::aie::broadcast<float, N>(mean);

  // pass 2: var = Σ(x - mean)² / cols   (centered). aie::mul returns an accum, so assign it to
  // an intermediate vector first (the accum->vector conversion), exactly as layer_norm.cc does.
  ::aie::vector<float, N> var_v = ::aie::zeros<float, N>();
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<float, N> sq = ::aie::mul(d, d);
    var_v = ::aie::add(var_v, sq);
  }
  float var = ::aie::reduce_add(var_v) / float(cols);
  float inv_std = ::aie::invsqrt(var + epsilon);
  ::aie::vector<float, N> inv_v = ::aie::broadcast<float, N>(inv_std);

  // write: out = (x - mean) * inv
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<float, N> out_v = ::aie::mul(d, inv_v);
    ::aie::store_v(output + i * N, out_v);
  }
  event1();
}

extern "C" {
void layer_norm_2pass_f32(float *input, float *output, int32_t cols) {
  layer_norm_2pass_f32<16>(input, output, cols);
}
}
