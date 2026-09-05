//===- norm_gemv_prologue.cc ----------------------------------*- C++ -*-===//
//
// Fused Norm+GEMV decode prologue. Normalizes the resident [m,K] A tile IN PLACE
// (bf16) before the GEMV, so the whole fused op is one dispatch.
//
// Decode is M=1 padded to M=64 (row 0 = query, rows 1..m-1 in this core's tile are
// ZERO). So the whole-tile sum == row-0 sum, and mean/var use K (the real-row
// length), NOT m*K. The reduction is a plain linear sweep over the L1 buffer
// (order-independent → no tiled-layout gather); the normalize is a uniform scalar
// op applied to every element (zero rows → harmless garbage in ignored output rows).
//
// RMS:  inv_rms = 1/sqrt(Σx²/K + eps);            a := inv_rms · x
// LN :  mean=Σx/K; var=Σx²/K − mean²; inv=1/sqrt(var+eps);  a := inv·(x − mean)
// (γ is folded into W'' host-side; β/bias via host-add or K-aug — not here.)
//
// f32 reduction + f32 normalize math (the bf16-long-reduction lesson); bf16 store
// (the matmul input dtype). Compile-time NORM_rms / NORM_ln.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef EPI_MK
#define EPI_MK 12288   // m * K  (whole resident A tile element count)
#endif
#ifndef EPI_K
#define EPI_K 768      // K = real query row length (the mean/var divisor)
#endif

extern "C" {
void norm_prologue(bfloat16 *restrict a) {
  event0();
  constexpr int V = 16;
  constexpr float epsilon = 1e-5f;

  // --- pass 1: linear Σx (LN) and Σx² over the whole tile, f32 accum ---
  ::aie::accum<accfloat, V> sum_acc;  sum_acc.from_vector(::aie::zeros<float, V>(), 0);
  ::aie::accum<accfloat, V> ssq_acc;  ssq_acc.from_vector(::aie::zeros<float, V>(), 0);
  for (int i = 0; i < EPI_MK; i += V) {
    ::aie::vector<bfloat16, V> x = ::aie::load_v<V>(a + i);
    sum_acc = ::aie::add(sum_acc, x);                 // bf16 lane added into f32 accum
    ::aie::vector<float, V> sq = ::aie::mul_square(x); // f32 squares (as in rms_norm.cc)
    ssq_acc = ::aie::add(ssq_acc, sq);
  }
  float ssq = ::aie::reduce_add(ssq_acc.to_vector<float>());

#ifdef NORM_ln
  float sum = ::aie::reduce_add(sum_acc.to_vector<float>());
  float mean = sum / (float)EPI_K;
  float var = ssq / (float)EPI_K - mean * mean;
  float inv = ::aie::invsqrt(var + epsilon);
  ::aie::vector<float, V> meanv = ::aie::broadcast<float, V>(mean);
  ::aie::vector<float, V> invv = ::aie::broadcast<float, V>(inv);
  for (int i = 0; i < EPI_MK; i += V) {
    ::aie::accum<accfloat, V> xa;
    xa.from_vector(::aie::load_v<V>(a + i), 0);        // bf16 -> f32 accum
    ::aie::vector<float, V> d = ::aie::sub(xa.to_vector<float>(), meanv);
    ::aie::vector<float, V> y = ::aie::mul(d, invv);
    ::aie::accum<accfloat, V> ya; ya.from_vector(y, 0);
    ::aie::store_v(a + i, ya.template to_vector<bfloat16>());
  }
#else  // NORM_rms
  float ms = ssq / (float)EPI_K;
  float inv = ::aie::invsqrt(ms + epsilon);
  ::aie::vector<float, V> invv = ::aie::broadcast<float, V>(inv);
  for (int i = 0; i < EPI_MK; i += V) {
    ::aie::accum<accfloat, V> xa;
    xa.from_vector(::aie::load_v<V>(a + i), 0);
    ::aie::vector<float, V> y = ::aie::mul(xa.to_vector<float>(), invv);
    ::aie::accum<accfloat, V> ya; ya.from_vector(y, 0);
    ::aie::store_v(a + i, ya.template to_vector<bfloat16>());
  }
#endif
  event1();
}
}
