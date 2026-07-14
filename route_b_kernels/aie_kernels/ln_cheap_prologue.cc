//===- ln_cheap_prologue.cc -----------------------------------*- C++ -*-===//
//
// CHEAP single-pass LayerNorm PROLOGUE for the resident whole_array matmul (feat/r1-ln-cheap).
//
// The THIRD path (after the 2-pass on-chip reduction died on L1 capacity -> 9 ms/dispatch, and the
// epilogue-correction died on catastrophic cancellation): the HOST computes the cheap per-row stats
// (mu, inv_std) -- the SAME reduction the host LayerNorm already does -- and the NPU does ONLY a
// SINGLE-PASS affine-normalize of each streamed A block, IN PLACE, before the mmul consumes it.
//
//   a := (x - mu[row]) * inv[row]        (one mul-add per element, f32 math, bf16 store)
//
// NO on-chip reduction, NO second A-stream: this runs INSIDE the existing single A DMA, one
// elementwise pass -- the input-side analog of the SiLU epilogue (mm_silu_epilogue.cc). Its on-chip
// cost is ~one A-tile touch (~0.1 ms class), NOT the 9 ms the re-streaming 2-pass cost.
//
// STATS DELIVERY: the compute tile has only 2 input DMA channels (A, B), both taken, so (mu, inv)
// can NOT arrive on a 3rd channel. They ride IN-BAND on the A stream: the generator prepends ONE
// stats "k-block" that the core peels into core-local mu[PRO_M]/inv[PRO_M] buffers (ln_cheap_load),
// then normalizes k-blocks 1..K/k with ln_cheap_apply. mu MUST be f32-precise (delivering mu in bf16
// re-introduces the |mean|-scaled cancellation -- see route_b_kernels/ctx_ln/ln_cheap_study.py, table
// A2: bf16-mean K-aug hits rel 16.6). Deliver mu as double-bf16 (mu_hi+mu_lo two-sum) or reinterpret
// f32 bytes; inv tolerates bf16 (it scales the already-centered O(std) residual).
//
// AFFINE IS FREE host-side (validated on feat/r1-ln-prologue): gamma folds into the weight
// (W' = diag(gamma) W) and beta into an additive output bias (b' = beta @ W). So this is
// normalize-ONLY.  LAYOUT: the A block sits in aie::mmul-BLOCKED order (the A_l2l1 dims_to_stream
// [(m/r,r*k),(k/s,s),(r,k),(s,1)]); a per-ROW op is NOT layout-independent, so we index the tiled
// coords exactly as mm_ln_epilogue.cc does over N. Element (row i, col c) of the [m,k] block is at
//   off(i,c) = (i/r)*(r*k) + (i%r)*s + (c/s)*(r*s) + (c%s).
//
// f32 reduction/normalize math, bf16 store (the matmul input dtype) -- the accuracy lesson of
// ln_2pass.cc / norm_gemv_prologue.cc.  NOT DEVICE-VALIDATED (numpy-only; see the study + verdict).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef PRO_M
#define PRO_M 64      // rows per A-tile (m)
#endif
#ifndef PRO_K
#define PRO_K 32      // cols per streamed k-block (k)
#endif
#ifndef PRO_R
#define PRO_R 8       // mmul sub-tile r
#endif
#ifndef PRO_S
#define PRO_S 8       // mmul sub-tile s
#endif

extern "C" {

// Peel the prepended stats block into core-local mu[PRO_M], inv[PRO_M] (f32). The generator lays the
// stats block out row-major-per-core: mu_hi[PRO_M] | mu_lo[PRO_M] | inv[PRO_M] as bf16 lanes, so mu
// recovers ~f32 precision via the two-sum (mu = mu_hi + mu_lo) and inv rides one bf16 lane.
void ln_cheap_load(const bfloat16 *restrict stats, float *restrict mu, float *restrict inv) {
  event0();
  constexpr int V = 16;
  static_assert(PRO_M % V == 0, "PRO_M must be a multiple of 16");
  for (int i = 0; i < PRO_M; i += V) {
    ::aie::accum<accfloat, V> hi;
    hi.from_vector(::aie::load_v<V>(stats + i), 0);
    ::aie::accum<accfloat, V> lo;
    lo.from_vector(::aie::load_v<V>(stats + PRO_M + i), 0);
    ::aie::vector<float, V> muv =
        ::aie::add(hi.template to_vector<float>(), lo.template to_vector<float>());
    ::aie::store_v(mu + i, muv);
    ::aie::accum<accfloat, V> iv;
    iv.from_vector(::aie::load_v<V>(stats + 2 * PRO_M + i), 0);
    ::aie::store_v(inv + i, iv.template to_vector<float>());
  }
  event1();
}

// Normalize ONE streamed [PRO_M, PRO_K] mmul-blocked A block IN PLACE: a := (x - mu[row]) * inv[row].
// Called K/k times per output tile (once per streamed k-block), between DMA-acquire and matmul.
void ln_cheap_apply(bfloat16 *restrict a, const float *restrict mu, const float *restrict inv) {
  event0();
  constexpr int V = PRO_S;          // one mmul sub-tile row-chunk (s contiguous) per vector
  constexpr int nch = PRO_K / PRO_S; // k/s chunks per row in this block
  for (int i = 0; i < PRO_M; i++) {
    const int base = (i / PRO_R) * (PRO_R * PRO_K) + (i % PRO_R) * PRO_S;
    ::aie::vector<float, V> muv = ::aie::broadcast<float, V>(mu[i]);
    ::aie::vector<float, V> ivv = ::aie::broadcast<float, V>(inv[i]);
    for (int cj = 0; cj < nch; cj++) {
      ::aie::vector<bfloat16, V> xb = ::aie::load_v<V>(a + base + cj * PRO_R * PRO_S);
      ::aie::accum<accfloat, V> xa;
      xa.from_vector(xb, 0);          // bf16 -> f32 accum domain
      ::aie::vector<float, V> d = ::aie::sub(xa.template to_vector<float>(), muv);
      ::aie::vector<float, V> y = ::aie::mul(d, ivv);
      ::aie::accum<accfloat, V> ya;
      ya.from_vector(y, 0);
      ::aie::store_v(a + base + cj * PRO_R * PRO_S, ya.template to_vector<bfloat16>()); // bf16 store
    }
  }
  event1();
}

} // extern "C"
