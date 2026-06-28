//===- mv_bf16_gelu.cc - bf16 cascade-FFN GEMV micro-kernels + GELU --------===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Cascade-FFN Phase 0 (Task 3) micro-kernel object. Dense bf16, no packing.
// Base: bf16_cascade/mv_bf16.cc (matvec/zero/partial_plus_r, r=32,
// set_rounding(conv_even)); _store / _b_offset variant pattern mirrored in
// bf16 from int4_awq/mv_int4_bf16.cc (Q/S/Z packing dropped). GELU(tanh)
// epilogue ported from patches/iron-gemv-gelu-epilogue.patch.
//
// TWO-SHAPE RESOLUTION (see STRUCTURE.md A.3 / B.1 / B.5)
// -------------------------------------------------------
// The Whisper FFN cascade needs the matvec at TWO different compile-time
// shapes on the same herd, so a single -DDIM_M/-DDIM_K compile cannot serve
// both. Instead of building two renamed .o, we instantiate the templated
// matvec at its two FIXED shapes under two distinctly-named extern "C"
// symbols in ONE .cc -> ONE mv_bf16_gelu.o links into the whole herd:
//
//   fc1 : out slab DIM_M=384 rows, reduction DIM_K=768  (Wfc1_slab[384,768] @ x_norm[768])
//   fc2 : out      DIM_M=768 rows, reduction DIM_K=384  (Wfc2_slab[768,384] @ h_ty[384], one K-chunk/core)
//
// Why one .o (not two renamed .o): the cascade herd is a single link_with
// target; one object keeps the kernel-cache key stable and avoids a second
// compile + symbol-rename dance. The shape lives in the symbol name, not in a
// -D macro, so this file compiles WITHOUT any -DDIM_M/-DDIM_K (the verify
// command passes none). Adding distinct shapes later = add a named wrapper.
//
// The elementwise helpers (zero / partial_plus_r / gelu) take a RUNTIME length
// n so one symbol each serves both the 384 (fc1) and 768 (fc2) output slabs --
// same runtime-n convention the shipped gelu_tile_bf16 epilogue already uses.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

// ---------------------------------------------------------------------------
// matvec: c[0..m] (+)= a[m,k] @ b[k], bf16, accfloat accumulate, inner r=32.
// ---------------------------------------------------------------------------

// Accumulating: c[row] += dot(a_row, b). Caller zeroes c first (or K is one
// chunk and a prior _store seeded it).
template <unsigned m, unsigned k, unsigned r>
void matvec_vectorized_impl(const bfloat16 *__restrict a,
                            const bfloat16 *__restrict b,
                            bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned row = 0; row < m; row++) {
    aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
    const bfloat16 *a_row = a + row * k;
    for (unsigned i = 0; i < k; i += r) {
      aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a_row + i);
      aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b + i);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    float partial = aie::reduce_add(acc.template to_vector<float>());
    c[row] = static_cast<bfloat16>(static_cast<float>(c[row]) + partial);
  }
}

// Overwriting: c[row] = dot(a_row, b). Folds the per-tile zero into the
// epilogue (saves a separate zero call) when this core writes the first/only
// K-chunk -- which is the case for BOTH our shapes (fc1 K=768 in one chunk;
// fc2 each core owns one 384 K-chunk, the cross-core reduction rides the
// cascade, not the kernel).
template <unsigned m, unsigned k, unsigned r>
void matvec_vectorized_store_impl(const bfloat16 *__restrict a,
                                  const bfloat16 *__restrict b,
                                  bfloat16 *__restrict c) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned row = 0; row < m; row++) {
    aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
    const bfloat16 *a_row = a + row * k;
    for (unsigned i = 0; i < k; i += r) {
      aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a_row + i);
      aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b + i);
      acc = aie::mac(acc, a_vec, b_vec);
    }
    float partial = aie::reduce_add(acc.template to_vector<float>());
    c[row] = static_cast<bfloat16>(partial);
  }
}

// ---------------------------------------------------------------------------
// M-TILE GEMM (Phase 1): one weight tile [NW,WROW] reused across m_act activation
// rows. out[ar, col_off + wr] = dot(w[wr,:K], act[ar,:K]) (+ bias) for ar in
// [0,m_act), wr in [0,NW). The weight tile stays L1-resident across all m_act
// rows (the LPDDR-reuse win vs the M=1 GEMV); act/out are [m_act, *] row-major.
// row_stride is the OUTPUT row width (fc1: M_SLAB=384; fc2: D=768).
//
// BIAS via K-AUGMENTED WEIGHT COLUMN (ADD_BIAS): the fc1 weight tile is padded to
// WROW = K + R columns with col K = bias[oc] and cols K+1..WROW = 0 (host-baked).
// After the dot we MAC the bias block w[wr, K:K+R] (= [bias,0,..,0]) against an
// all-ones vector into the accumulator, so reduce_add yields dot + bias. This adds
// the bias with a VECTOR mac (NO scalar (float)bfloat16 -- that path is the
// unreliable-on-aie2p one that bit the LayerNorm; see layernorm_rows_impl). It
// also keeps bias OFF the inL2L1 weight stream -> the w1 objectFIFO is a clean
// depth-2 double-buffer (multiplexing bias there inflated its lock to init=3 and
// the producer overran the in-flight tile at slow M_TILE>1 -- the M_TILE>1 bug;
// see internal notes mtile-ffn-phase1-ln-fix). fc2 uses WROW=K, ADD_BIAS=false.
// m_act/col_off/row_stride are RUNTIME so one symbol serves any M_TILE.
// ---------------------------------------------------------------------------
template <unsigned NW, unsigned K, unsigned R, unsigned WROW, bool ADD_BIAS>
void gemm_tile_impl(uint32_t m_act, uint32_t row_stride, uint32_t col_off,
                    const bfloat16 *__restrict w, const bfloat16 *__restrict act,
                    bfloat16 *__restrict out) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, R> ones = aie::broadcast<bfloat16, R>(static_cast<bfloat16>(1.0f));
  for (uint32_t ar = 0; ar < m_act; ar++) {
    const bfloat16 *a_row = act + ar * K;
    bfloat16 *o_row = out + ar * row_stride + col_off;
    for (unsigned wr = 0; wr < NW; wr++) {
      aie::accum<accfloat, R> acc = aie::zeros<accfloat, R>();
      const bfloat16 *w_row = w + wr * WROW;
      for (unsigned i = 0; i < K; i += R) {
        aie::vector<bfloat16, R> w_vec = aie::load_v<R>(w_row + i);
        aie::vector<bfloat16, R> a_vec = aie::load_v<R>(a_row + i);
        acc = aie::mac(acc, w_vec, a_vec);
      }
      if (ADD_BIAS) {
        // bias block w_row[K:K+R] = [bias, 0, ..., 0]; mac vs ones adds bias to
        // lane 0 of acc (vector op, no scalar bf16<->f32) -> reduce includes bias.
        aie::vector<bfloat16, R> b_vec = aie::load_v<R>(w_row + K);
        acc = aie::mac(acc, b_vec, ones);
      }
      o_row[wr] = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
  }
}

// ---------------------------------------------------------------------------
// elementwise helpers (runtime length n -> one symbol serves both shapes).
// ---------------------------------------------------------------------------

// x[ar*n + j] += bias[j] for ar in [0,m_act), j in [0,n) -- bias[n] broadcast
// across all m_act rows. In-place, vectorized over j (n a multiple of 16).
// Used for +bias_fc1 (n=M_SLAB, before GELU) and the cascade-HEAD +b_fc2
// (n=D). Two extern symbols below (slab/d) so each MLIR call has a single
// memref type (one C symbol cannot back two distinct memref decls).
static inline void add_bias_bcast_impl(uint32_t m_act, uint32_t n,
                                       bfloat16 *__restrict x,
                                       const bfloat16 *__restrict bias) {
  for (uint32_t ar = 0; ar < m_act; ar++) {
    bfloat16 *x_row = x + ar * n;
    for (uint32_t j = 0; j < n; j += 16) {
      aie::vector<bfloat16, 16> xv = aie::load_v<16>(x_row + j);
      aie::vector<bfloat16, 16> bv = aie::load_v<16>(bias + j);
      aie::store_v(x_row + j, aie::add(xv, bv));
    }
  }
}

// Non-affine LayerNorm per row (affine gamma/beta folded host-side into
// Wfc1/bias_fc1, per gen_ffn): xnorm[ar,:] = (x[ar,:] - mean) * rstd.
// x/xnorm are [m_act, n] row-major bf16, n a multiple of 16.
//
// VECTORIZED, mirroring the proven old MLIR LN numerics (which passed rel-L2
// 0.0217). A prior SCALAR-f32 version was wrong on-device (rel-L2 0.165): the
// per-element scalar `(float)bfloat16` conversion path is unreliable on aie2p
// (the old MLIR LN converted via VECTOR extf, never scalar). Here: sum via f32
// vector accumulate; sum-of-squares by squaring in BF16 (bf16 vector mul, then
// f32 accumulate -- f32 VECTOR mul is unsupported on aie2p); mean/var/rstd in
// SCALAR f32 (legal -- the old LN used scalar mean*mean); normalize in bf16.
// NOT __restrict: the generator calls this in-place (x == xnorm aliased).
static void layernorm_rows_impl(uint32_t m_act, uint32_t n, float eps,
                                const bfloat16 *x, bfloat16 *xnorm) {
  const float inv_n = 1.0f / static_cast<float>(n);
  for (uint32_t ar = 0; ar < m_act; ar++) {
    const bfloat16 *xr = x + ar * n;
    bfloat16 *outr = xnorm + ar * n;
    // sum and sum-of-squares via mac into f32 accumulators (bf16 inputs, f32
    // accumulate -- the only legal MAC form on aie2p; cf. the matvec kernel).
    // acc_sum lane i += xb[i]*1 ; acc_sq lane i += xb[i]*xb[i].
    aie::accum<accfloat, 16> acc_sum = aie::zeros<accfloat, 16>();
    aie::accum<accfloat, 16> acc_sq = aie::zeros<accfloat, 16>();
    aie::vector<bfloat16, 16> ones = aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(1.0f));
    for (uint32_t j = 0; j < n; j += 16) {
      aie::vector<bfloat16, 16> xb = aie::load_v<16>(xr + j);
      acc_sum = aie::mac(acc_sum, xb, ones);
      acc_sq = aie::mac(acc_sq, xb, xb);
    }
    float sum = aie::reduce_add(acc_sum.to_vector<float>());
    float sumsq = aie::reduce_add(acc_sq.to_vector<float>());
    float mean = sum * inv_n;
    float var = sumsq * inv_n - mean * mean;             // scalar f32 (legal)
    float rstd = 1.0f / aie::sqrt(var + eps);
    aie::vector<bfloat16, 16> vmean = aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(mean));
    aie::vector<bfloat16, 16> vrstd = aie::broadcast<bfloat16, 16>(static_cast<bfloat16>(rstd));
    for (uint32_t j = 0; j < n; j += 16) {
      aie::vector<bfloat16, 16> xb = aie::load_v<16>(xr + j);
      aie::vector<bfloat16, 16> cen = aie::sub(xb, vmean);
      aie::vector<bfloat16, 16> nrm = aie::mul(cen, vrstd);  // accum -> bf16 vector
      aie::store_v(outr + j, nrm);
    }
  }
}

static inline void zero_impl(bfloat16 *__restrict c, uint32_t n) {
  for (uint32_t i = 0; i < n; i++)
    c[i] = static_cast<bfloat16>(0.0f);
}

// d[i] = partial[i] + r_full[offset + i]   (residual / cascade-head inject)
static inline void partial_plus_r_impl(const bfloat16 *__restrict partial,
                                       const bfloat16 *__restrict r_full,
                                       int offset, bfloat16 *__restrict d,
                                       uint32_t n) {
  for (uint32_t i = 0; i < n; i++)
    d[i] = static_cast<bfloat16>(static_cast<float>(partial[i]) +
                                 static_cast<float>(r_full[offset + i]));
}

// GELU (tanh approx) -- same math as patches/iron-gemv-gelu-epilogue.patch
// (aie_kernels/aie2p/gelu.cc). In-place over n bf16 elements (n a multiple of
// 16). MUST be applied ONCE over the FULL m_output slab AFTER the matvec
// inner-loop, NOT per per-call matvec tile -- a per-tile m_input can be < 16
// and overrun the 16-wide vector (the ru-2.05 bug; STRUCTURE.md B.2).
static inline void gelu_inplace_bf16(bfloat16 *__restrict v, int32_t n) {
  const bfloat16 k0_5 = 0.5f, k1 = 1.0f, sqrt_2_over_pi = 0.79788456f,
                 kBeta = 0.044715f;
  auto v05 = aie::broadcast<bfloat16, 16>(k0_5);
  auto v1 = aie::broadcast<bfloat16, 16>(k1);
  auto vs2opi = aie::broadcast<bfloat16, 16>(sqrt_2_over_pi);
  auto vBeta = aie::broadcast<bfloat16, 16>(kBeta);
  auto it = aie::begin_restrict_vector<16>(v);
  for (int i = 0; i < n; i += 16) {
    aie::vector<bfloat16, 16> x = *it;
    aie::vector<bfloat16, 16> x2 = aie::mul(x, x);
    aie::vector<bfloat16, 16> x3 = aie::mul(x, x2);
    aie::vector<bfloat16, 16> x3_beta = aie::mul(x3, vBeta);
    aie::vector<bfloat16, 16> inner = aie::add(x, x3_beta);
    auto inner1 = aie::mul(inner, vs2opi);
    auto tanh_out = aie::tanh<bfloat16>(inner1.to_vector<float>());
    aie::vector<bfloat16, 16> one_plus_tanh = aie::add(tanh_out, v1);
    aie::vector<bfloat16, 16> mul_v05 = aie::mul(v05, one_plus_tanh);
    auto result = aie::mul(x, mul_v05);
    *it++ = result.to_vector<bfloat16>();
  }
}

// L1-CAPACITY (build-proven, Task 4): each whole slab (Wfc1 [384,768] or Wfc2
// [768,384]) is 576 KB bf16, but AIE2P L1 is only 64 KB/core, so the WHOLE-slab
// _store entries below cannot keep their weights resident (aiecc: "allocated
// buffers exceeded available memory"). The generator instead streams [M_INPUT,K]
// weight tiles L3->L1 and calls the tiled entry once per tile, writing M_INPUT
// output rows. M_INPUT=8 matches the int4 reference M_TILE=8: tile L1 =
// [8,768]bf16 = 12 KB (fits); 384/8=48 fc1 iters, 768/8=96 fc2 iters per core.
// 384 and 768 are both divisible by 8. GELU/zero/partial_plus_r are unchanged --
// GELU still runs ONCE over the full 384 h_ty slab after the fc1 tiling loop
// (the ru-2.05 rule holds). The whole-slab entries are kept for back-compat.
#define M_INPUT 8

extern "C" {

// --- tiled (L3-streamed weights): write M_INPUT output rows per call ---
// fc1 tile: h[tile*8 .. +8] = Wfc1_tile[8,768] @ x_norm[768]   (overwrite)
void matvec_fc1_tile_bf16_store(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  matvec_vectorized_store_impl<M_INPUT, 768, 32>(a, b, c);
}
// fc2 tile: partial[tile*8 .. +8] = Wfc2_tile[8,384] @ h_ty[384]  (overwrite)
void matvec_fc2_tile_bf16_store(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  matvec_vectorized_store_impl<M_INPUT, 384, 32>(a, b, c);
}

// --- M-TILE GEMM tiles (Phase 1): weight tile reused across m_act act rows ---
// fc1: h_slab[ar, col_off + 0..8] = Wfc1_tile[8,800] @ xnorm[ar, :768] + bias.
//      The fc1 weight tile is K-AUGMENTED to 800 cols (col 768 = bias, 769:800 =
//      0); the kernel folds the bias in with a vector mac (ADD_BIAS) -> bias is
//      NOT streamed separately (keeps the w1 objectFIFO a clean depth-2 buffer).
//      h_slab row width = M_SLAB = 384; col_off = tile * 8.
#define FC1_K 768
#define FC1_WROW 800   // FC1_K + 32 (one R-block for the bias; 32-mult -> aligned)
void gemm_fc1_tile_bf16(uint32_t m_act, uint32_t col_off, bfloat16 *w,
                        bfloat16 *act, bfloat16 *h) {
  gemm_tile_impl<M_INPUT, FC1_K, 32, FC1_WROW, /*ADD_BIAS=*/true>(
      m_act, /*row_stride=M_SLAB*/ 384, col_off, w, act, h);
}
// fc2: partial[ar, col_off + 0..8] = Wfc2_tile[8,384] @ h_slab[ar, :384]
//      (partial row width = D = 768; col_off = tile * 8). No bias here (the
//      cascade-HEAD adds b_fc2); ADD_BIAS = false. The w2 [8,384] tile lives in
//      the first 384 cols of the SHARED [8,FC1_WROW] weight buffer (row stride
//      FC1_WROW), so WROW = FC1_WROW (not 384) -- the dot still spans only K=384.
void gemm_fc2_tile_bf16(uint32_t m_act, uint32_t col_off, bfloat16 *w,
                        bfloat16 *act, bfloat16 *partial) {
  gemm_tile_impl<M_INPUT, 384, 32, FC1_WROW, /*ADD_BIAS=*/false>(
      m_act, /*row_stride=D*/ 768, col_off, w, act, partial);
}

// bias broadcast-add across m_act rows, in-place. Two symbols (single memref
// type each at the MLIR callsite): _slab for h[m_act,M_SLAB]+bias_fc1[M_SLAB],
// _d for partial[m_act,D]+b_fc2[D].
void add_bias_bcast_slab_bf16(uint32_t m_act, uint32_t n, bfloat16 *x, bfloat16 *bias) {
  add_bias_bcast_impl(m_act, n, x, bias);
}
void add_bias_bcast_d_bf16(uint32_t m_act, uint32_t n, bfloat16 *x, bfloat16 *bias) {
  add_bias_bcast_impl(m_act, n, x, bias);
}

// Per-row non-affine LayerNorm: xnorm[m_act,n] = (x - mean)*rstd.
void layernorm_rows_bf16(uint32_t m_act, uint32_t n, float eps, bfloat16 *x,
                         bfloat16 *xnorm) {
  layernorm_rows_impl(m_act, n, eps, x, xnorm);
}

// --- fc1: Wfc1_slab[384,768] @ x_norm[768] -> h_ty[384] (one K-chunk/core) ---
// _store seeds h_ty (overwrite); accumulate variant provided for completeness.
// NOTE: whole-slab residency exceeds 64 KB L1 -- use the tiled entries above on
// device; these are retained for back-compat / small-shape callers only.
void matvec_fc1_bf16_store(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  matvec_vectorized_store_impl<384, 768, 32>(a, b, c);
}
void matvec_fc1_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  matvec_vectorized_impl<384, 768, 32>(a, b, c);
}

// --- fc2: Wfc2_slab[768,384] @ h_ty[384] -> partial_ty[768] (this core's
// K-chunk). _store writes the per-core partial; the cross-core K-reduction is
// the cascade add (MLIR), not the kernel. ---
void matvec_fc2_bf16_store(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  matvec_vectorized_store_impl<768, 384, 32>(a, b, c);
}
void matvec_fc2_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  matvec_vectorized_impl<768, 384, 32>(a, b, c);
}
// Accumulate with a base offset into b: lets a caller keep one big B buffer and
// tile fc2's K via scf.for if it is ever chunked finer than one core. Not used
// by the single-chunk Phase 0 path but kept (cheap) for Task 4 flexibility.
void matvec_fc2_bf16_b_offset(bfloat16 *a, bfloat16 *b, int b_offset,
                              bfloat16 *c) {
  matvec_vectorized_impl<768, 384, 32>(a, b + b_offset, c);
}

// --- shape-agnostic elementwise helpers (runtime n) ---

// c[0..n] = 0
void zero_vectorized_bf16(uint32_t n, bfloat16 *c) { zero_impl(c, n); }

// d[i] = partial[i] + r_full[offset + i], i in [0,n)
void partial_plus_r_bf16(uint32_t n, bfloat16 *partial, bfloat16 *r_full,
                         int offset, bfloat16 *d) {
  partial_plus_r_impl(partial, r_full, offset, d, n);
}

// GELU(tanh) in-place over the full m_output slab c[0..n) (n a multiple of 16).
void gelu_tile_bf16(uint32_t n, bfloat16 *c) {
  gelu_inplace_bf16(c, static_cast<int32_t>(n));
}

} // extern "C"
