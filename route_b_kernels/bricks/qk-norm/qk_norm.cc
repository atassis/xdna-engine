//===- qk_norm.cc ---------------------------------------------*- C++ -*-===//
//
// BRICK: qk-norm  [group: norm]
//
// Generic RMSNorm op-TYPE applied to Q/K vectors pre-attention, against the
// resident [tile,D] stream contract: `rows` resident rows (tokens x heads,
// flattened by the caller) each of `cols` (= head_dim D) elements, contiguous.
// One dispatch normalizes the WHOLE resident tile (not one row) — this is the
// "process the resident [tile,D] block" shape the frontier build wants, not a
// per-model/per-shape clone.
//
//   per row of `cols` values (single-pass, NOT centered — this is RMSNorm, not
//   LayerNorm: no mean subtraction, matches ln_2pass.cc's two-pass LN only in
//   the reduce-over-`cols`-per-row axis convention):
//     ms      = Σ x²  / cols
//     inv_rms = 1 / sqrt(ms + eps)          (eps = 1e-6, the common Q/K-norm eps;
//                                             override via -DQKNORM_EPS if a model
//                                             needs LayerNorm's 1e-5 instead)
//     out[j]  = x[j] * inv_rms * gamma[j]    (gamma optional: pass nullptr for
//                                             the weight-free / gamma==1 variant)
//
// gamma is a single [cols] learned-scale vector SHARED across all `rows` in the
// tile (the standard Q/K-RMSNorm shape: one scale per head_dim, broadcast over
// every token/head row) — NOT a per-row weight. This is the same "shared
// per-column vector, broadcast over the resident rows" shape as norm_gemv_prologue.cc's
// γ-folding note, kept here as an explicit runtime input instead of host-folded,
// since qk-norm sits pre-attention (no downstream GEMV to fold γ into).
//
// f32 in / f32 out (docs/05 "never re-expand"; the encoder/decode resident
// stream is f32 at this seam per ln_2pass.cc's contract). `cols` must be a
// multiple of N (16). `rows` >= 1; rows==1 degenerates to the single-vector
// case (e.g. decode M=1 Q/K norm).
//
// Parameterized by:
//   - N (template, vector width, default 16 — the aie2p f32 SIMD width used by
//     every sibling kernel in this dir)
//   - cols (runtime, = head_dim D; e.g. 64/128 for typical Q/K head dims)
//   - rows (runtime, = resident tile row count; e.g. num_heads or num_heads*tile_tokens)
//   - gamma present/absent (runtime nullptr check) — generic op-TYPE, not two clones
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef QKNORM_EPS
#define QKNORM_EPS 1e-6f
#endif

template <int N>
void qk_norm_f32(const float *restrict input, const float *restrict gamma,
                  float *restrict output, int32_t rows, int32_t cols) {
  event0();
  constexpr float epsilon = QKNORM_EPS;
  const int chunks = cols / N;

  for (int r = 0; r < rows; r++) {
    const float *in_row = input + (size_t)r * cols;
    float *out_row = output + (size_t)r * cols;

    // pass 1: ms = Σx² / cols
    ::aie::vector<float, N> ssq_v = ::aie::zeros<float, N>();
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<float, N> x = ::aie::load_v<N>(in_row + i * N);
      ::aie::vector<float, N> sq = ::aie::mul(x, x);
      ssq_v = ::aie::add(ssq_v, sq);
    }
    float ms = ::aie::reduce_add(ssq_v) / float(cols);
    float inv_rms = ::aie::invsqrt(ms + epsilon);
    ::aie::vector<float, N> inv_v = ::aie::broadcast<float, N>(inv_rms);

    // pass 2: out = x * inv_rms [* gamma]
    if (gamma != nullptr) {
      for (int i = 0; i < chunks; i++) {
        ::aie::vector<float, N> x = ::aie::load_v<N>(in_row + i * N);
        ::aie::vector<float, N> g = ::aie::load_v<N>(gamma + i * N);
        ::aie::vector<float, N> scaled = ::aie::mul(x, inv_v);
        ::aie::vector<float, N> out_v = ::aie::mul(scaled, g);
        ::aie::store_v(out_row + i * N, out_v);
      }
    } else {
      for (int i = 0; i < chunks; i++) {
        ::aie::vector<float, N> x = ::aie::load_v<N>(in_row + i * N);
        ::aie::vector<float, N> out_v = ::aie::mul(x, inv_v);
        ::aie::store_v(out_row + i * N, out_v);
      }
    }
  }
  event1();
}

extern "C" {
// Full RMSNorm with learned per-column scale gamma[cols], broadcast over rows.
void qk_norm_f32_gamma(float *input, float *gamma, float *output,
                        int32_t rows, int32_t cols) {
  qk_norm_f32<16>(input, gamma, output, rows, cols);
}

// Weight-free variant (gamma == 1 everywhere) — e.g. a model whose Q/K-norm
// has no learned scale.
void qk_norm_f32_nogamma(float *input, float *output, int32_t rows,
                          int32_t cols) {
  qk_norm_f32<16>(input, nullptr, output, rows, cols);
}
}
