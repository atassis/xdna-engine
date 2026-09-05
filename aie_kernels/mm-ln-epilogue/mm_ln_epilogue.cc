//===- mm_ln_epilogue.cc --------------------------------------*- C++ -*-===//
//
// Fused LayerNorm epilogue for the M-STATIONARY GEMM (Phase 1.2 spike).
//
// The matmul writes its C accumulator in the aie::mmul TILED layout (r x t
// sub-tiles); the C ObjectFifo's dims_to_stream un-tiles it to row-major AFTER
// the kernel. So element (logical row i, col j) of a [DIM_M, DIM_N] output tile
// sits in the f32 accumulator at:
//     off(i,j) = (i/R)*(R*DIM_N) + (i%R)*T + (j/T)*(R*T) + (j%T)
// i.e. logical row i = DIM_N/T contiguous T-chunks at
//     base(i,tj) = (i/R)*(R*DIM_N) + (i%R)*T + tj*(R*T),  tj in [0, DIM_N/T).
//
// A row REDUCTION (LayerNorm) is NOT layout-independent (unlike elementwise
// SiLU), so we reduce/normalize/write in these tiled coordinates, leaving the
// output in the same tiled layout for the C ObjectFifo to un-tile. Per row:
//   mean = Σx/DIM_N;  var = Σ(x-mean)²/DIM_N;  out = (x-mean)/sqrt(var+1e-5)
// NORMALIZE-ONLY (gamma=1/beta=0); affine is a follow-on.  Two-pass f32 (stable).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 16
#endif
#ifndef DIM_N
#define DIM_N 64
#endif
#ifndef DIM_R
#define DIM_R 4
#endif
#ifndef DIM_T
#define DIM_T 8
#endif

template <int M, int N, int R, int T>
static inline void ln_tiled_f32_bf16(const float *restrict acc, bfloat16 *restrict out) {
  constexpr int V = T;                 // one mmul sub-tile column-chunk (T=8) per vector
  constexpr int ntiles = N / T;
  constexpr float epsilon = 1e-5f;

  for (int i = 0; i < M; i++) {
    const int rowbase = (i / R) * (R * N) + (i % R) * T;

    // pass 1: mean = Σx / N
    ::aie::vector<float, V> sum_v = ::aie::zeros<float, V>();
    for (int tj = 0; tj < ntiles; tj++)
      sum_v = ::aie::add(sum_v, ::aie::load_v<V>(acc + rowbase + tj * R * T));
    float mean = ::aie::reduce_add(sum_v) / float(N);
    ::aie::vector<float, V> mean_v = ::aie::broadcast<float, V>(mean);

    // pass 2: var = Σ(x - mean)² / N
    ::aie::vector<float, V> var_v = ::aie::zeros<float, V>();
    for (int tj = 0; tj < ntiles; tj++) {
      ::aie::vector<float, V> d = ::aie::sub(::aie::load_v<V>(acc + rowbase + tj * R * T), mean_v);
      ::aie::vector<float, V> sq = ::aie::mul(d, d);
      var_v = ::aie::add(var_v, sq);
    }
    float var = ::aie::reduce_add(var_v) / float(N);
    float inv = ::aie::invsqrt(var + epsilon);
    ::aie::vector<float, V> inv_v = ::aie::broadcast<float, V>(inv);

    // write (x - mean) * inv back to the SAME tiled positions, f32 -> bf16
    for (int tj = 0; tj < ntiles; tj++) {
      ::aie::vector<float, V> d = ::aie::sub(::aie::load_v<V>(acc + rowbase + tj * R * T), mean_v);
      ::aie::vector<float, V> y = ::aie::mul(d, inv_v);
      ::aie::accum<accfloat, V> ya;
      ya.from_vector(y);
      ::aie::store_v(out + rowbase + tj * R * T, ya.template to_vector<bfloat16>());
    }
  }
}

extern "C" {
// acc: [DIM_M, DIM_N] f32 in aie::mmul tiled layout; out: same layout, bf16
void mm_ln_epilogue_f32_bf16(float *acc, bfloat16 *out) {
  event0();
  ln_tiled_f32_bf16<DIM_M, DIM_N, DIM_R, DIM_T>(acc, out);
  event1();
}
}
