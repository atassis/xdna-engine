//===- layernorm.cc --------------------------------------------*- C++ -*-===//
//
// GENERIC "norm{LN|RMS}" brick (route_b_kernels/bricks, group: norm).
//
// Model studied: route_b_kernels/aie_kernels/ln_2pass.cc (the encoder ctxLN 2-pass f32 kernel).
// This brick generalizes that kernel into a parameterized op-TYPE against the resident [tile,D]
// stream contract, per the CLAUDE.md rail: "Fixes must be GENERIC primitives ... never per-model
// hand-fused kernels". It is NOT a clone of ln_2pass.cc for one model's shape/affine convention --
// it is the reusable primitive that ln_2pass.cc's normalize-only, host-affine-folded behavior is
// ONE instantiation of.
//
// One call normalizes ONE row ("tile") of `cols` (= D) elements. f32 accumulation in / f32 out
// (docs/05 "never re-expand" -- callers cast to bf16 at the DMA boundary, not inside this brick).
// `cols` must be a multiple of the vector width N (default 16, matches ln_2pass.cc).
//
// Two op-TYPES selected at compile time via the `UseMean` template bool (the "{LN|RMS}" half of
// norm{LN|RMS}):
//   UseMean = true   -> LayerNorm:  mean = Sigma(x)/cols ; var = Sigma((x-mean)^2)/cols  (2-pass,
//                        exact-mean-centered -- NOT the E[x^2]-mean^2 shortcut, which ln_2pass.cc's
//                        header documents as catastrophically cancelling in bf16, measured 5.77%
//                        block rel error on the shipped bf16 `layer_norm` kernel. REJECTED here too.)
//   UseMean = false  -> RMSNorm:    mean := 0 ; var = Sigma(x^2)/cols  (single accumulation pass,
//                        no mean subtraction -- the standard RMSNorm definition.)
// Both share the same invsqrt-based normalize tail: out = (x - mean) * inv_std, with mean == 0
// under RMSNorm so the subtract costs nothing extra to keep the code path unified.
//
// A second template bool `Affine` selects whether gamma/beta are applied ON-DEVICE
// (out = out*gamma + beta) or left to the caller (ln_2pass.cc's model folds affine into the
// adjacent matmul weights on the host and calls this brick normalize-only, i.e. Affine=false).
// Both are legitimate instantiations of the same generic primitive -- the brick does not hardcode
// either choice.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

// Generic 2-pass norm{LN|RMS} core. `epsilon` and `cols` are runtime params (a MODEL is data: the
// block schedule picks epsilon/shape, the brick stays parameterized only on vector width + op-TYPE
// flags). `gamma`/`beta` are ignored (and may be nullptr) when Affine == false.
template <int N, bool UseMean, bool Affine>
void layernorm_core(const float *restrict input, float *restrict output, int32_t cols,
                     float epsilon, const float *restrict gamma, const float *restrict beta) {
  event0();
  const int chunks = cols / N;

  // pass 1: mean (LN) or skip (RMS: mean stays the zero vector).
  ::aie::vector<float, N> mean_v = ::aie::zeros<float, N>();
  if constexpr (UseMean) {
    ::aie::vector<float, N> sum_v = ::aie::zeros<float, N>();
    for (int i = 0; i < chunks; i++) {
      sum_v = ::aie::add(sum_v, ::aie::load_v<N>(input + i * N));
    }
    float mean = ::aie::reduce_add(sum_v) / float(cols);
    mean_v = ::aie::broadcast<float, N>(mean);
  }

  // pass 2: var = Sigma((x - mean)^2) / cols  (mean == 0 under RMS, so this is Sigma(x^2)/cols).
  ::aie::vector<float, N> var_v = ::aie::zeros<float, N>();
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<float, N> sq = ::aie::mul(d, d);
    var_v = ::aie::add(var_v, sq);
  }
  float var = ::aie::reduce_add(var_v) / float(cols);
  float inv_std = ::aie::invsqrt(var + epsilon);
  ::aie::vector<float, N> inv_v = ::aie::broadcast<float, N>(inv_std);

  // write: out = (x - mean) * inv, optionally affine-scaled/shifted in the same pass.
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(input + i * N), mean_v);
    ::aie::vector<float, N> out_v = ::aie::mul(d, inv_v);
    if constexpr (Affine) {
      ::aie::vector<float, N> g = ::aie::load_v<N>(gamma + i * N);
      ::aie::vector<float, N> b = ::aie::load_v<N>(beta + i * N);
      out_v = ::aie::add(::aie::mul(out_v, g), b);
    }
    ::aie::store_v(output + i * N, out_v);
  }
  event1();
}

} // namespace route_b_bricks

extern "C" {

// LayerNorm, normalize-only (drop-in for ln_2pass.cc's contract: host folds gamma/beta into the
// adjacent matmul, e.g. the encoder ctxLN's shipped usage).
void layernorm_ln_normalize_f32(float *input, float *output, int32_t cols) {
  route_b_bricks::layernorm_core<16, /*UseMean=*/true, /*Affine=*/false>(
      input, output, cols, 1e-5f, nullptr, nullptr);
}

// LayerNorm, on-device affine (out = (x-mean)*inv*gamma + beta). Generic instantiation for models
// that don't fold affine into an adjacent weight matrix.
void layernorm_ln_affine_f32(float *input, float *output, int32_t cols, float *gamma,
                              float *beta) {
  route_b_bricks::layernorm_core<16, /*UseMean=*/true, /*Affine=*/true>(
      input, output, cols, 1e-5f, gamma, beta);
}

// RMSNorm, normalize-only (mean=0, var=E[x^2]). Common in modern decoder-only LLM blocks (the
// decode-north-star Gemma-270M path's norm{RMS} instance).
void layernorm_rms_normalize_f32(float *input, float *output, int32_t cols) {
  route_b_bricks::layernorm_core<16, /*UseMean=*/false, /*Affine=*/false>(
      input, output, cols, 1e-5f, nullptr, nullptr);
}

// RMSNorm, on-device affine (weight-only scale is the RMSNorm convention; beta accepted for
// generality but callers typically pass a zero vector or nullptr-equivalent all-zeros buffer).
void layernorm_rms_affine_f32(float *input, float *output, int32_t cols, float *gamma,
                               float *beta) {
  route_b_bricks::layernorm_core<16, /*UseMean=*/false, /*Affine=*/true>(
      input, output, cols, 1e-5f, gamma, beta);
}

} // extern "C"
