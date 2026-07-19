//===- rmsnorm.cc ---------------------------------------------*- C++ -*-===//
//
// GENERIC AIE2P brick: RMSNorm (op-TYPE `norm{RMS}`, see docs/aie2p-brick-catalog.md
// and the norm{LN|RMS} rail in build-methodology). Affine, single-pass, no re-centering
// (that's the LN-vs-RMS distinction: RMSNorm skips the mean-subtraction entirely and
// normalizes by the root-mean-square of x directly).
//
//   per row of `cols` values:
//     ms     = Σ(x²) / cols                      (mean of squares, NOT variance)
//     inv    = 1 / sqrt(ms + eps)                (SFU invsqrt, same primitive as ln_2pass.cc)
//     out    = (x * inv) * gamma + beta          (affine; beta optional -- pass nullptr to skip)
//
// This is the standard RMSNorm formulation (Zhang & Sennrich 2019): no mean-centering step,
// one pass over the row (vs. LayerNorm's two-pass mean-then-variance). gamma is required
// (RMSNorm's scale term); beta is an OPTIONAL bias for callers that fuse the model's
// residual-affine into this brick (nullptr => output has no additive bias, matching the
// bias-free RMSNorm most LLMs (Gemma, Llama) actually ship).
//
// Contract: resident [tile, D] stream. One call normalizes ONE row of `cols` elements
// (matches the ml/layernorm per-row core_body contract already established by ln_2pass.cc).
// f32 in / f32 out. `cols` must be a multiple of N (vector width, default 16).
// gamma/beta are `cols`-length per-column affine params (broadcast across rows by the caller
// loop, not baked into this per-row core).
//
// Generic op-TYPE, NOT a per-model clone: no assumption about which model (Gemma/Llama/etc.)
// or which layer calls it -- D, eps, and the presence/absence of beta are all caller-supplied.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

template <int N>
void rmsnorm_f32(const float *restrict input, const float *restrict gamma,
                  const float *restrict beta, float *restrict output,
                  int32_t cols, float eps) {
  event0();
  const int chunks = cols / N;

  // single pass: ms = Σ(x²) / cols   (no mean-centering -- that's the LN/RMS distinction)
  ::aie::vector<float, N> sumsq_v = ::aie::zeros<float, N>();
  for (int i = 0; i < chunks; i++) {
    ::aie::vector<float, N> x = ::aie::load_v<N>(input + i * N);
    ::aie::vector<float, N> sq = ::aie::mul(x, x);
    sumsq_v = ::aie::add(sumsq_v, sq);
  }
  float ms = ::aie::reduce_add(sumsq_v) / float(cols);
  float inv_rms = ::aie::invsqrt(ms + eps);
  ::aie::vector<float, N> inv_v = ::aie::broadcast<float, N>(inv_rms);

  // write: out = (x * inv) * gamma [+ beta]
  if (beta != nullptr) {
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<float, N> x = ::aie::load_v<N>(input + i * N);
      ::aie::vector<float, N> g = ::aie::load_v<N>(gamma + i * N);
      ::aie::vector<float, N> b = ::aie::load_v<N>(beta + i * N);
      ::aie::vector<float, N> normed = ::aie::mul(x, inv_v);
      ::aie::vector<float, N> scaled = ::aie::mul(normed, g);
      ::aie::vector<float, N> out_v = ::aie::add(scaled, b);
      ::aie::store_v(output + i * N, out_v);
    }
  } else {
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<float, N> x = ::aie::load_v<N>(input + i * N);
      ::aie::vector<float, N> g = ::aie::load_v<N>(gamma + i * N);
      ::aie::vector<float, N> normed = ::aie::mul(x, inv_v);
      ::aie::vector<float, N> out_v = ::aie::mul(normed, g);
      ::aie::store_v(output + i * N, out_v);
    }
  }
  event1();
}

extern "C" {
// Default eps = 1e-6f (common RMSNorm default, e.g. Gemma/Llama); pass a different
// eps via the templated core above if a model needs 1e-5 (LN's convention) instead.
// beta = nullptr for the bias-free RMSNorm variant most LLM stacks actually ship.
void rmsnorm_f32(float *input, float *gamma, float *beta, float *output,
                  int32_t cols) {
  rmsnorm_f32<16>(input, gamma, beta, output, cols, 1e-6f);
}

void rmsnorm_f32_nobeta(float *input, float *gamma, float *output,
                         int32_t cols) {
  rmsnorm_f32<16>(input, gamma, nullptr, output, cols, 1e-6f);
}
}
