//===- mm_ln_prologue.cc --------------------------------------*- C++ -*-===//
//
// On-chip LayerNorm PROLOGUE for the resident whole_array matmul (feat/r1-ln-prologue).
// Normalizes the streamed A-tile over the FULL contraction dim K, per row, in f32, BEFORE the
// mmul consumes it -- so the host sends RAW x (no host LayerNorm) and the idle NPU array does
// the normalize. This is the M=T ENCODER analog of norm_gemv_prologue.cc (which was M=1 DECODE,
// whole [m,K] A resident in one L1 tile). Here K is DMA-BLOCKED (K_div_k=32 k-blocks of k=32),
// and the full-K A row for a core (m=64,K=1024 bf16 = 128 KB) does NOT fit L1 (64 KB), so the
// full row is NEVER co-resident. The prologue therefore runs as a TWO-PASS over a re-streamed A:
//
//   pass 1 (stats):   for each streamed k-block, ACCUMULATE per-row Sx and Sxx into core-local
//                     f32 buffers ssum[m], ssq[m]  (ln_prologue_stats, called K_div_k times).
//   finalize (once):  mu[i]=Sx/K;  var=Sxx/K - mu^2;  inv[i]=1/sqrt(var+eps)  (ln_prologue_finalize).
//   pass 2 (apply):   for each re-streamed k-block, a := (x - mu[row])*inv[row], IN PLACE, then the
//                     mmul reduces the normalized block into the C accumulator (ln_prologue_apply).
//
// f32 reduction + f32 normalize math, bf16 store (the matmul input dtype) -- the accuracy lesson
// of ln_2pass.cc / norm_gemv_prologue.cc: the E[x^2]-mean^2 var is safe HERE because the sums are
// f32 accumulators (accfloat), NOT the bf16 running-sum the shipped `layer_norm` used (docs/08's
// 5.77% cancellation, REJECTED). Mirrors norm_gemv_prologue's f32 one-pass-stats idiom exactly.
// If the WER gate ever fails on a high-|mean| layer, upgrade to the CENTERED two-pass (an extra
// mean-only A-pass before the var-pass, +1 A stream) -- see BUILD-AND-BENCH.md.
//
// LAYOUT: the A block in L1 is in aie::mmul-BLOCKED order (the A_l2l1 dims_to_stream
// [(m/r,r*k),(k/s,s),(r,k),(s,1)]). A per-row reduction over k is NOT layout-independent (unlike
// the elementwise SiLU epilogue), so we reduce/normalize in the SAME tiled coords the microkernel
// consumes -- exactly like mm_ln_epilogue.cc does over N. Element (row i, col c) of the [m,k] block
// sits at:  off(i,c) = (i/r)*(r*k) + (i%r)*s + (c/s)*(r*s) + (c%s).  So row i = k/s chunks of s
// contiguous elements at base (i/r)*(r*k)+(i%r)*s, stride r*s between chunks (validated bit-exact
// in route_b_kernels/ctx_ln/ln_prologue_validate.py, V3).
//
// AFFINE IS FREE (folded host-side, validated V2): gamma folds into the weight (W' = diag(g).W,
// cached like the plain weight) and beta becomes an additive output bias b' = b @ W. So this
// kernel is normalize-ONLY; the host supplies W'/b'. See rust/npu-parakeet/src/npu.rs.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// Per-block tile dims (== the matmul's DIM_M / DIM_K and the mmul sub-tile r,s). One kernel call
// sees ONE k-block [PRO_M, PRO_K]; the full contraction length (the mean/var divisor) is PRO_KFULL.
#ifndef PRO_M
#define PRO_M 64
#endif
#ifndef PRO_K
#define PRO_K 32
#endif
#ifndef PRO_R
#define PRO_R 8
#endif
#ifndef PRO_S
#define PRO_S 8
#endif
#ifndef PRO_KFULL
#define PRO_KFULL 1024
#endif

template <int M, int K, int R, int S>
static inline void ln_prologue_stats_t(const bfloat16 *restrict a, float *restrict ssum,
                                       float *restrict ssq) {
  constexpr int V = S;            // one mmul sub-tile row-chunk (s contiguous) per vector
  constexpr int nch = K / S;      // k/s chunks per row in this block
  for (int i = 0; i < M; i++) {
    const int base = (i / R) * (R * K) + (i % R) * S;
    ::aie::vector<float, V> sv = ::aie::zeros<float, V>();
    ::aie::vector<float, V> qv = ::aie::zeros<float, V>();
    for (int cj = 0; cj < nch; cj++) {
      ::aie::vector<bfloat16, V> xb = ::aie::load_v<V>(a + base + cj * R * S);
      ::aie::accum<accfloat, V> xa; xa.from_vector(xb, 0);   // bf16 -> f32 accum domain
      ::aie::vector<float, V> xf = xa.template to_vector<float>();
      sv = ::aie::add(sv, xf);
      ::aie::vector<float, V> sq = ::aie::mul(xf, xf);        // mul -> accum -> vector
      qv = ::aie::add(qv, sq);
    }
    ssum[i] += ::aie::reduce_add(sv);   // ACCUMULATE across the K_div_k streamed blocks
    ssq[i] += ::aie::reduce_add(qv);
  }
}

template <int M>
static inline void ln_prologue_finalize_t(const float *restrict ssum, const float *restrict ssq,
                                          float *restrict mu, float *restrict inv, int kfull,
                                          int prologue_on) {
  // RTP-GATED (rtp[1]): the LN prologue is a MODE on the SINGLE resident array program (the core
  // always runs the two-pass structure, exactly as the modal epilogue is a mode on rtp[0]). When
  // the dispatch is a NON-LN GEMM (q/k/v/out/ff.l2/pw2), prologue_on==0 forces mu=0, inv=1, so
  // ln_prologue_apply becomes a := (x - 0)*1 = x -- an IDENTITY, leaving the raw A untouched for a
  // plain matmul. When prologue_on==1 (fc1/pw1) we compute the real per-row stats. This is what lets
  // ONE resident xclbin serve both LN and non-LN dispatches with ZERO hw-context switches.
  constexpr float epsilon = 1e-5f;
  const float invk = 1.0f / (float)kfull;
  for (int i = 0; i < M; i++) {
    if (prologue_on) {
      float m = ssum[i] * invk;
      float var = ssq[i] * invk - m * m;   // f32 E[x^2]-mean^2 (safe: f32 accum, not bf16 sum)
      mu[i] = m;
      inv[i] = ::aie::invsqrt(var + epsilon);
    } else {
      mu[i] = 0.0f;
      inv[i] = 1.0f;
    }
  }
}

template <int M, int K, int R, int S>
static inline void ln_prologue_apply_t(bfloat16 *restrict a, const float *restrict mu,
                                       const float *restrict inv) {
  constexpr int V = S;
  constexpr int nch = K / S;
  for (int i = 0; i < M; i++) {
    const int base = (i / R) * (R * K) + (i % R) * S;
    ::aie::vector<float, V> muv = ::aie::broadcast<float, V>(mu[i]);
    ::aie::vector<float, V> ivv = ::aie::broadcast<float, V>(inv[i]);
    for (int cj = 0; cj < nch; cj++) {
      ::aie::vector<bfloat16, V> xb = ::aie::load_v<V>(a + base + cj * R * S);
      ::aie::accum<accfloat, V> xa; xa.from_vector(xb, 0);
      ::aie::vector<float, V> d = ::aie::sub(xa.template to_vector<float>(), muv);
      ::aie::vector<float, V> y = ::aie::mul(d, ivv);
      ::aie::accum<accfloat, V> ya; ya.from_vector(y, 0);
      ::aie::store_v(a + base + cj * R * S, ya.template to_vector<bfloat16>());  // bf16 store
    }
  }
}

extern "C" {

// zero the core-local [PRO_M] stat accumulators before pass 1 (called once per output tile).
void ln_prologue_zero(float *ssum, float *ssq) {
  event0();
  constexpr int V = 16;
  ::aie::vector<float, V> z = ::aie::zeros<float, V>();
  for (int i = 0; i < PRO_M; i += V) {
    ::aie::store_v(ssum + i, z);
    ::aie::store_v(ssq + i, z);
  }
  event1();
}

// pass 1: accumulate per-row Sx, Sxx over ONE streamed [PRO_M, PRO_K] mmul-blocked A block.
// ssum/ssq are core-local [PRO_M] f32 buffers, zeroed by the core before the K loop.
void ln_prologue_stats(bfloat16 *a, float *ssum, float *ssq) {
  event0();
  ln_prologue_stats_t<PRO_M, PRO_K, PRO_R, PRO_S>(a, ssum, ssq);
  event1();
}

// finalize: turn accumulated Sx/Sxx into per-row mu, inv (called once, between the two passes).
// rtp[1] gates the prologue on (fc1/pw1) vs off (identity, every other GEMM) -- see the template.
void ln_prologue_finalize(float *ssum, float *ssq, float *mu, float *inv, int32_t *rtp) {
  event0();
  ln_prologue_finalize_t<PRO_M>(ssum, ssq, mu, inv, PRO_KFULL, rtp[1]);
  event1();
}

// pass 2: normalize ONE re-streamed [PRO_M, PRO_K] block IN PLACE, a := (x - mu[row]) * inv[row].
void ln_prologue_apply(bfloat16 *a, float *mu, float *inv) {
  event0();
  ln_prologue_apply_t<PRO_M, PRO_K, PRO_R, PRO_S>(a, mu, inv);
  event1();
}

} // extern "C"
