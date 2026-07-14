//===- relpos_mha.cc ------------------------------------------*- C++ -*-===//
//
// FastConformer (NeMo Transformer-XL) RELATIVE-POSITION multi-head self-attention
// -- the rel-pos-SPECIFIC compute bricks (the hardest Parakeet node, A4).
//
// Parakeet-tdt-0.6b-v3 encoder: d_model=1024, H=8, head_dim DK=128, T = encoder
// frames (compute-bound, M=T regime -> the COMPUTE bricks win). This kernel
// covers the pieces that are NOT plain GEMM; the heavy GEMMs (Q/K/V/pos/out
// projections + AC/BD/ctx) are mmul tiles reused from IRON aie2p/mm.cc and are
// NOT re-authored here (see the gen wiring / golden for the dataflow).
//
// MATH (per head h, host ref = rust/npu-parakeet/src/{ops.rs,encoder.rs} +
//       scripts/parakeet_ref_encoder.py mhsa, already rel<=3e-5 vs ONNX):
//   q,k,v = x @ {Wq,Wk,Wv}                         [T, DK]   (mmul)
//   p     = pos_enc @ Wpos                         [P=2T-1, DK] (mmul)
//   qu = q + pos_bias_u[h]   (broadcast over T)              <- BRICK add_pos_bias
//   qv = q + pos_bias_v[h]
//   AC = qu @ k^T                                  [T, T]    (mmul)
//   BD = qv @ p^T                                  [T, P]    (mmul)
//   BD_shifted = rel_shift(BD)                      [T, T]    <- BRICK: STRIDED RELAYOUT
//   scores = (AC + BD_shifted) / sqrt(DK)
//   attn   = softmax_keys(scores)                  [T, T]    <- BRICK: vectorized exp2
//   ctx    = attn @ v                              [T, DK]   (mmul)
//   out    = merge_heads(ctx) @ Wout               [T, D]    (mmul)
//
// THE TWO NEW BRICKS authored here (everything else is mmul):
//
//  (1) rel_shift AS A STRIDED RELAYOUT (NOT a recompute). The NeMo rel_shift
//      (pad-1, reshape, drop-row, slice) is algebraically the index identity
//          BD_shifted[i, j] = BD[i, (T-1) - i + j]      (derived + golden-checked)
//      i.e. output row i is a CONTIGUOUS length-T window of BD row i starting at
//      column (T-1-i). Laid out row-major [T, P] this is a 2-D strided slice:
//          base offset = (T-1),  row stride = (P-1) = 2T-2,  col stride = 1.
//      So there is ZERO arithmetic: in this kernel it is a per-row pointer offset
//      (BD + i*P + (T-1-i)); in the resident dataflow it is a single dma_bd n-D
//      strided descriptor over the BD buffer (the MOVEMENT brick). No shuffle, no
//      pad buffer, no recompute. This is the load-bearing rel-pos insight.
//
//  (2) softmax over keys via VECTORIZED exp2 (COMPUTE/SFU brick). e^x =
//      2^(x*log2e); aie::exp2 is the only on-chip exp and runs 16 lanes/issue.
//      We sweep the key dimension in VL-lane vectors (max -> exp2 -> sum -> inv),
//      unlike the Whisper M=1 mha_decode.cc which issues a 16-lane SFU per SCALAR
//      key. Reductions stay f32 (accfloat); probs drain to bf16 to feed the ctx
//      mmul directly.
//
// PRECISION: AC/BD arrive f32 (mmul accfloat drained to f32 -- the host keeps
// f32 through the score assembly). softmax max/sum reduce in f32; the only bf16
// hop is exp2's output (same hw constraint as IRON softmax.cc) + the probs that
// feed the ctx GEMM. The pos_bias add is bf16 (it feeds the bf16 AC/BD mmul).
//
// STATUS: CANDIDATE (A4, low-confidence-by-design). Peano CPU compile-checked +
// numpy golden (rel-L2 gate). On-device dataflow wiring (objectFIFO + the BD
// strided dma_bd + the mmul tiles) is the expected interactive rework.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

// e^x = 2^(x*log2e). No libm on aie2p.
static constexpr float LOG2E = 1.4426950408889634f;

// float-domain vector width (512-bit reg / 32-bit = 16 lanes).
static constexpr int VL = 16;

// Max supported T for the per-row f32 score scratch (2 KB at 512). The encoder
// runs one tile per head; bump if T_enc exceeds this (build-time).
#ifndef RELPOS_TMAX
#define RELPOS_TMAX 512
#endif

// head_dim (Parakeet = 128). Compile-time so the bias add fully vectorizes.
#ifndef RELPOS_DK
#define RELPOS_DK 128
#endif
static constexpr int DK = RELPOS_DK;
static_assert(DK % VL == 0, "DK must be a multiple of the vector width");

// Encoder frame count T baked at build time for the STANDALONE step-1 kernel
// (relpos_scores_softmax_bake below). Parakeet block 0 = 32 frames (P = 2T-1).
// Single-tile design: AC[T,T]+BD[T,P]+probs[T,T] must fit L1, which caps this
// standalone variant (T=32 uses ~14 KB; large T needs the row-tiled resident
// block, not this de-risk kernel). Override with -DRELPOS_T at compile time to
// match the IRON generator's -T.
#ifndef RELPOS_T
#define RELPOS_T 32
#endif

// ---------------------------------------------------------------------------
// BRICK 1 -- broadcast pos_bias add.  q_in [T, DK] (row-major, one head) plus
// bias [DK] -> q_out [T, DK].  Vectorized over DK; called twice/head (u, v).
// bf16 in/out (feeds the bf16 AC/BD mmul).
// ---------------------------------------------------------------------------
extern "C" void relpos_add_bias(bfloat16 *restrict q_in, bfloat16 *restrict bias,
                                bfloat16 *restrict q_out, int32_t T) {
  event0();
  for (int i = 0; i < T; i++) {
    const bfloat16 *qr = q_in + i * DK;
    bfloat16 *orow = q_out + i * DK;
    for (int c = 0; c < DK; c += VL) {
      aie::vector<bfloat16, VL> qv = aie::load_v<VL>(qr + c);
      aie::vector<bfloat16, VL> bv = aie::load_v<VL>(bias + c);
      aie::store_v(orow + c, aie::add(qv, bv));
    }
  }
  event1();
}

// scalar exp via the only hw exp (vector exp2 -> bf16); used only for the
// ragged key tail (T not a multiple of VL). The hot path is fully vectorized.
static inline float exp2_scalar(float x) {
  aie::vector<float, VL> v = aie::broadcast<float, VL>(x);
  aie::vector<bfloat16, VL> e = aie::exp2<bfloat16>(v);
  return (float)e.get(0);
}

// ---------------------------------------------------------------------------
// BRICK 2 -- rel_shift (strided relayout) FUSED with the score add/scale and the
// vectorized-exp2 softmax over keys.
//   AC   : [T, T] row-major f32   (qu @ k^T, mmul-drained to f32)
//   BD   : [T, P] row-major f32   (qv @ p^T, P = 2T-1)
//   probs: [T, T] row-major bf16  (out: softmax weights, feed the ctx mmul)
//   inv_scale = 1/sqrt(DK)
// Per row i:  scores[j] = (AC[i,j] + BD[i, (T-1)-i+j]) * inv_scale
//             probs[i]  = softmax_j(scores)
// The rel_shift is the pointer offset `BD + i*P + (T-1-i)` -- a strided read, no
// recompute (see header note 1).
// ---------------------------------------------------------------------------
extern "C" void relpos_scores_softmax(float *restrict AC, float *restrict BD,
                                      bfloat16 *restrict probs, int32_t T,
                                      int32_t P, float inv_scale) {
  event0();
  alignas(aie::vector_decl_align) static float srow[RELPOS_TMAX];

  aie::vector<float, VL> inv_scale_v = aie::broadcast<float, VL>(inv_scale);
  aie::vector<float, VL> log2e_v = aie::broadcast<float, VL>(LOG2E);

  for (int i = 0; i < T; i++) {
    const float *ac_row = AC + i * T;
    // STRIDED RELAYOUT: rel_shift -> contiguous length-T window of BD row i.
    const float *bd_row = BD + (int)i * P + (T - 1 - i);
    bfloat16 *prob_row = probs + i * T;

    // -- pass 1: scores = (AC + BD_shifted) * inv_scale ; row max (f32) --
    float rowmax = -3.0e38f;
    int j = 0;
    for (; j + VL <= T; j += VL) {
      // ac_row/prob_row have row stride T (or P), which is NOT a multiple of VL for
      // real T (e.g. T=172 -> 172%16=12), so the per-row base is unaligned. Aligned
      // load_v/store_v truncate to 128b and corrupt -> use unaligned everywhere the
      // stride is T/P (same root cause as the bd_row rel_shift load above).
      aie::vector<float, VL> a = aie::load_unaligned_v<VL>(ac_row + j);
      // bd_row = BD + i*P + (T-1-i): the rel_shift base is NEVER VL-aligned (the
      // (T-1-i) shift is not a multiple of 16), so this MUST be an unaligned load.
      // aie::load_v is an ALIGNED load -> on aie2p it truncates the address to the
      // 128b boundary and returns shifted/garbage BD (masked when BD<<AC, e.g. real
      // block-0 after rescale; exposed the moment BD ~ AC, e.g. synth / spread).
      aie::vector<float, VL> b = aie::load_unaligned_v<VL>(bd_row + j);
      aie::vector<float, VL> s =
          aie::mul(aie::add(a, b), inv_scale_v).to_vector<float>();
      aie::store_v(srow + j, s);
      float cm = aie::reduce_max(s);
      if (cm > rowmax) rowmax = cm;
    }
    for (; j < T; j++) {
      float s = (ac_row[j] + bd_row[j]) * inv_scale;
      srow[j] = s;
      if (s > rowmax) rowmax = s;
    }

    // -- pass 2: exp2((s - max) * log2e) -> bf16 probs ; running f32 sum --
    aie::vector<float, VL> maxv = aie::broadcast<float, VL>(rowmax);
    aie::accum<accfloat, VL> sumacc = aie::zeros<accfloat, VL>();
    j = 0;
    for (; j + VL <= T; j += VL) {
      aie::vector<float, VL> s = aie::load_v<VL>(srow + j);
      aie::vector<float, VL> d = aie::sub(s, maxv);
      aie::vector<float, VL> sl = aie::mul(d, log2e_v).to_vector<float>();
      aie::vector<bfloat16, VL> e = aie::exp2<bfloat16>(sl);
      aie::store_unaligned_v(prob_row + j, e);
      sumacc = aie::add(sumacc, e); // widen bf16 -> f32 accumulate
    }
    float sum = aie::reduce_add(sumacc.to_vector<float>());
    for (; j < T; j++) {
      float e = exp2_scalar((srow[j] - rowmax) * LOG2E);
      prob_row[j] = (bfloat16)e;
      sum += e;
    }

    // -- pass 3: normalize probs *= 1/sum  (aie::inv reciprocal brick) --
    bfloat16 inv_sum = (bfloat16)aie::inv(sum);
    aie::vector<bfloat16, VL> inv_sum_v = aie::broadcast<bfloat16, VL>(inv_sum);
    j = 0;
    for (; j + VL <= T; j += VL) {
      aie::vector<bfloat16, VL> e = aie::load_unaligned_v<VL>(prob_row + j);
      aie::store_unaligned_v(prob_row + j,
                             aie::mul(e, inv_sum_v).to_vector<bfloat16>());
    }
    for (; j < T; j++) {
      prob_row[j] = (bfloat16)((float)prob_row[j] * (float)inv_sum);
    }
  }
  event1();
}

// ---------------------------------------------------------------------------
// STEP-1 STANDALONE ENTRY -- zero-scalar-arg wrapper over relpos_scores_softmax
// with T, P and inv_scale baked at build time. This is what the IRON generator
// (relpos_scores_softmax_iron.py) declares as its core Kernel, so the (AC, BD,
// probs) 3-buffer ABI carries no runtime scalars (mirrors dwconv1d_k9_bf16).
// It proves the two hard rel-pos bricks -- the zero-arithmetic strided-relayout
// rel_shift and the vectorized-exp2 softmax -- as one dataflow WITHOUT any matmul.
//   AC   : [T, T] row-major f32
//   BD   : [T, P] row-major f32   (P = 2T-1)
//   probs: [T, T] row-major bf16  (softmax over keys of rel_shift(BD)+AC, scaled)
// ---------------------------------------------------------------------------
extern "C" void relpos_scores_softmax_bake(float *restrict AC,
                                           float *restrict BD,
                                           bfloat16 *restrict probs) {
  constexpr int T = RELPOS_T;
  constexpr int P = 2 * T - 1;
  // inv_scale = 1/sqrt(DK). DK is compile-time (Parakeet = 128); the literal is
  // guarded so a DK change is caught at build instead of silently mis-scaling.
  static_assert(DK == 128,
                "baked inv_scale is 1/sqrt(128); regenerate for a different DK");
  constexpr float inv_scale = 0.08838834764831843f; // 1/sqrt(128)
  relpos_scores_softmax(AC, BD, probs, T, P, inv_scale);
}

// ---------------------------------------------------------------------------
// STEP-2 BRICK 3 -- the AC score matmul AC = qu @ k^T, ROW-MAJOR f32 out.
//   qu : [T, DK] row-major bf16  (q + pos_bias_u, one head)
//   k  : [T, DK] row-major bf16  (one head; k^T is IMPLICIT -- AC[i,j] is the dot
//                                 of qu row i with k row j, so no transpose buffer
//                                 and no strided load are needed)
//   AC : [T, T]  row-major f32   (RESIDENT L1 accumulator; handed straight to the
//                                 softmax epilogue below, never DMA'd to host)
// bf16 inputs, f32 (accfloat) accumulate + f32 lane-reduce -- the SAME numeric
// path as the aie::mmul tile (bf16*bf16 -> accfloat), mirrored from the proven
// q.K dot in mha_decode.cc (L128). Producing AC ROW-MAJOR (not the mmul-blocked
// r*t layout) is deliberate: the softmax epilogue does a per-row rel_shift +
// softmax-over-keys, which needs row-major AC; the aie::mmul microkernel would
// emit a blocked tile that then needs an extra L1 de-block pass. For the tiny
// [T,DK]x[DK,T] score matmul (T=32, DK=128) this dot-product tile is the clean
// single-core resident form; the aie::mmul-blocked variant is the perf follow-up.
// DK is a compile-time multiple of VL (Parakeet = 128 = 8 VL), so the reduction
// fully vectorizes with no ragged tail.
// ---------------------------------------------------------------------------
// General row-major score matmul: out[i,j] = dot(A row i, B row j), A is [T,DK],
// B is [N,DK] (B^T implicit -- no transpose buffer, no strided load). N=T gives
// AC=qu@k^T; N=P gives BD=qv@p^T (step 3). bf16*bf16 -> f32 accfloat, f32 out.
static inline void relpos_dot_matmul(const bfloat16 *restrict A,
                                     const bfloat16 *restrict B,
                                     float *restrict out, int T, int N) {
  static_assert(DK % VL == 0, "DK must be a multiple of the vector width");
  for (int i = 0; i < T; i++) {
    const bfloat16 *a_row = A + i * DK;
    float *o_row = out + i * N;
    for (int j = 0; j < N; j++) {
      const bfloat16 *b_row = B + j * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL) {
        aie::vector<bfloat16, VL> av = aie::load_v<VL>(a_row + d);
        aie::vector<bfloat16, VL> bv = aie::load_v<VL>(b_row + d);
        acc = aie::mac(acc, av, bv); // bf16*bf16 accumulated in f32
      }
      o_row[j] = aie::reduce_add(acc.to_vector<float>());
    }
  }
}

// STEP-2 AC matmul: qu @ k^T -> [T,T]. Thin wrapper over the general dot-matmul.
extern "C" void relpos_ac_matmul(bfloat16 *restrict qu, bfloat16 *restrict k,
                                 float *restrict AC, int32_t T) {
  event0();
  relpos_dot_matmul(qu, k, AC, T, T);
  event1();
}

// ---------------------------------------------------------------------------
// STEP-2 COMPOSED ENTRY -- the FIRST resident-block test: the AC score matmul
// feeds the scores->softmax brick with the f32 score tile staying RESIDENT in L1
// (never round-tripping to host). This mirrors the modal generator's
// matmul -> f32 acc in L1 -> epilogue shape; here the "epilogue" is
// rel_shift(BD) + scale + softmax over keys instead of SiLU.
//   qk   : [2*T, DK] row-major bf16  (PACKED: qu = qk[0:T], k = qk[T:2T]). qu and
//          k share ONE input buffer so the core stays within the NPU2 compute
//          tile's 2 input-DMA-channel budget (qk + BD), the same 2-input
//          discipline the modal design uses (bias folded, not a 3rd stream).
//   BD   : [T, P] row-major f32      (host-fed, P = 2T-1; unchanged from step 1)
//   probs: [T, T] row-major bf16     (softmax over keys of rel_shift(BD)+AC, scaled)
// The AC[T,T] f32 tile lives in a static L1 scratch (g_ac): written by the matmul,
// read IN PLACE by relpos_scores_softmax. That resident L1 hand-off IS the thesis
// this step exists to prove.  T is baked (must match the generator -T / -DRELPOS_T).
// ---------------------------------------------------------------------------
extern "C" void relpos_ac_scores_softmax_bake(bfloat16 *restrict qk,
                                              float *restrict BD,
                                              bfloat16 *restrict probs) {
  constexpr int T = RELPOS_T;
  constexpr int P = 2 * T - 1;
  static_assert(DK == 128,
                "baked inv_scale is 1/sqrt(128); regenerate for a different DK");
  constexpr float inv_scale = 0.08838834764831843f; // 1/sqrt(128)

  // RESIDENT L1 score tile: the matmul writes it, the softmax reads it -- no host
  // hop. Sized by the BAKED T (T=32 -> 4 KB), not RELPOS_TMAX (a T*T f32 tile at
  // 512 would be 1 MB -- far past L1; this de-risk variant is small-T by design).
  alignas(aie::vector_decl_align) static float g_ac[T * T];

  bfloat16 *qu = qk;         // [T, DK]
  bfloat16 *k = qk + T * DK; // [T, DK]
  relpos_ac_matmul(qu, k, g_ac, T);
  relpos_scores_softmax(g_ac, BD, probs, T, P, inv_scale);
}

// ---------------------------------------------------------------------------
// STEP-3 COMPOSED ENTRY -- BOTH score matmuls resident. Adds BD = qv @ p^T on
// chip (step 2 host-fed it), so AC and BD both live in L1 and the softmax brick
// consumes both with NO host score buffer at all. The rel_shift is the strided
// read INSIDE relpos_scores_softmax (BD + i*P + (T-1-i)), so the resident g_bd is
// laid out [T,P] row-major exactly as the host BD was.
//   qk  : [2*T, DK] bf16  PACKED (qu = qk[0:T], k = qk[T:2T])   -- input DMA ch 1
//   qvp : [(T+P), DK] bf16 PACKED (qv = qvp[0:T], p = qvp[T:T+P]) -- input DMA ch 2
//   probs: [T, T] bf16 out
// Two packed inputs keep the core within the NPU2 2-input-DMA-channel budget (the
// pos_bias adds qu=q+u / qv=q+v are folded host-side into the packed buffers).
// g_ac[T*T] + g_bd[T*P] both resident f32 (T=32: 4KB + ~8KB). Single-tile, small-T.
// ---------------------------------------------------------------------------
extern "C" void relpos_qkp_scores_softmax_bake(bfloat16 *restrict qk,
                                               bfloat16 *restrict qvp,
                                               bfloat16 *restrict probs) {
  constexpr int T = RELPOS_T;
  constexpr int P = 2 * T - 1;
  static_assert(DK == 128,
                "baked inv_scale is 1/sqrt(128); regenerate for a different DK");
  constexpr float inv_scale = 0.08838834764831843f; // 1/sqrt(128)

  alignas(aie::vector_decl_align) static float g_ac[T * T];
  alignas(aie::vector_decl_align) static float g_bd[T * P];

  bfloat16 *qu = qk;          // [T, DK]
  bfloat16 *k = qk + T * DK;  // [T, DK]
  bfloat16 *qv = qvp;         // [T, DK]
  bfloat16 *p = qvp + T * DK; // [P, DK]
  relpos_dot_matmul(qu, k, g_ac, T, T); // AC = qu @ k^T  [T,T]
  relpos_dot_matmul(qv, p, g_bd, T, P); // BD = qv @ p^T  [T,P]
  relpos_scores_softmax(g_ac, g_bd, probs, T, P, inv_scale);
}

// ---------------------------------------------------------------------------
// STEP-4 BRICK 4 -- the context matmul ctx = probs @ V (the AV half). Weighted-
// sum form: ctx[i, :] = sum_j probs[i,j] * V[j, :] -- for each output row, a
// running DK-wide f32 accumulation of the value rows weighted by the attention
// probabilities. This is the natural resident form (probs row + V rows both
// row-major, no transpose); the contraction is over the T keys (not DK like the
// score matmuls). bf16 in, f32 accumulate, bf16 out (ctx feeds the bf16 out proj).
//   probs : [T, T]  row-major bf16  (softmax output; resident in the full block)
//   V     : [T, DK] row-major bf16  (one head)
//   ctx   : [T, DK] row-major bf16
// ---------------------------------------------------------------------------
static inline void relpos_ctx_matmul(const bfloat16 *restrict probs,
                                     const bfloat16 *restrict V,
                                     bfloat16 *restrict ctx, int T) {
  static_assert(DK % VL == 0, "DK must be a multiple of the vector width");
  for (int i = 0; i < T; i++) {
    const bfloat16 *p_row = probs + i * T;
    bfloat16 *ctx_row = ctx + i * DK;
    // One accumulator strip per VL-wide slice of the DK output; accumulate the
    // weighted value rows over the T keys, then narrow to bf16.
    for (int d = 0; d < DK; d += VL) {
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int j = 0; j < T; j++) {
        aie::vector<bfloat16, VL> wv = aie::broadcast<bfloat16, VL>(p_row[j]);
        aie::vector<bfloat16, VL> vv = aie::load_v<VL>(V + j * DK + d);
        acc = aie::mac(acc, wv, vv); // bf16*bf16 -> f32
      }
      aie::store_v(ctx_row + d, acc.to_vector<bfloat16>());
    }
  }
}

// STEP-4 STANDALONE ENTRY -- the AV matmul in isolation (host-fed probs + V),
// mirroring step 1 (host-fed scores). Validates the context brick before it is
// composed into the full resident block (step 5, where probs come resident from
// the softmax). T is baked (must match the generator -T / -DRELPOS_T).
//   probs : [T, T]  row-major bf16
//   V     : [T, DK] row-major bf16
//   ctx   : [T, DK] row-major bf16
extern "C" void relpos_ctx_bake(bfloat16 *restrict probs, bfloat16 *restrict V,
                                bfloat16 *restrict ctx) {
  event0();
  relpos_ctx_matmul(probs, V, ctx, RELPOS_T);
  event1();
}

// ---------------------------------------------------------------------------
// STEP-5 FULL COMPOSED ENTRY -- the ENTIRE per-head MHA node in one dispatch:
// AC + BD matmuls -> rel_shift + softmax -> ctx matmul, all resident. k, p AND v
// are all resident (softmax needs every key, ctx needs every value), packed into
// two input buffers to stay in the 2 input-DMA-channel budget.
//   qkv : [3*T, DK] bf16  PACKED (qu = qkv[0:T], k = qkv[T:2T], V = qkv[2T:3T])
//   qvp : [(T+P), DK] bf16 PACKED (qv = qvp[0:T], p = qvp[T:T+P])
//   ctx : [T, DK] bf16 out
// NOTE: single-tile, small-T only. At T=172 the resident k/p/v ([172,128] + [343,128]
// + [172,128] bf16 = ~176 KB) exceed L1 (64 KB) -- the full-T block MUST stage k/p/v in
// MemTile (512 KB) and tile the query rows. This entry proves the COMPOSITION; the
// row-tiled MemTile design is the scaling step.
// ---------------------------------------------------------------------------
extern "C" void relpos_full_bake(bfloat16 *restrict qkv, bfloat16 *restrict qvp,
                                 bfloat16 *restrict ctx) {
  constexpr int T = RELPOS_T;
  constexpr int P = 2 * T - 1;
  static_assert(DK == 128, "baked inv_scale is 1/sqrt(128)");
  constexpr float inv_scale = 0.08838834764831843f;
  alignas(aie::vector_decl_align) static float g_ac[T * T];
  alignas(aie::vector_decl_align) static float g_bd[T * P];
  alignas(aie::vector_decl_align) static bfloat16 g_probs[T * T];
  bfloat16 *qu = qkv;              // [T, DK]
  bfloat16 *k = qkv + T * DK;      // [T, DK]
  bfloat16 *V = qkv + 2 * T * DK;  // [T, DK]
  bfloat16 *qv = qvp;              // [T, DK]
  bfloat16 *p = qvp + T * DK;      // [P, DK]
  relpos_dot_matmul(qu, k, g_ac, T, T);                   // AC = qu @ k^T
  relpos_dot_matmul(qv, p, g_bd, T, P);                   // BD = qv @ p^T
  relpos_scores_softmax(g_ac, g_bd, g_probs, T, P, inv_scale); // probs
  relpos_ctx_matmul(g_probs, V, ctx, T);                  // ctx = probs @ V
}

// ===========================================================================
// STEP-6 ROW-TILED, MemTile-STAGED resident MHA block (T up to 172, one head).
//
// WHY: relpos_full_bake (step 5) needs k[T,DK], p[P,DK] AND V[T,DK] ALL resident
// in L1 (softmax reads every key, ctx every value). At the real T'=172 that is
// ~172 KB (k 43 + p 86 + V 43) >> the 64 KB L1 -- and p ALONE (86 KB) already
// overflows L1. So single-tile is impossible for real T; this step removes the
// wall by (a) STAGING k/p/V once in the 512 KB MemTile (L2) and (b) processing
// the T query rows in TILES of Tq so only per-tile [Tq,*] working set lives in L1.
//
// The per-QUERY-row computation is INDEPENDENT (row i of the output depends on
// qu[i], qv[i], ALL k, ALL p, ALL V -- softmax is per-row over the keys), so
// tiling the query rows changes NOTHING numerically PROVIDED the rel_shift window
// uses the GLOBAL query index. That is the one load-bearing correctness subtlety:
//
//   single-tile row i:      BD_shifted[i,j] = BD[i, (T-1) - i + j]
//   tiled  local row il in a tile whose global base is q0 (i_global = q0 + il):
//                           BD_shifted[il,j] = BD_tile[il, (T-1) - (q0+il) + j]
//
// i.e. the strided rel_shift base is (T-1) - (q0 + il), NOT (T-1) - il. Get the
// GLOBAL index into that offset and the tiled block is bit-identical to the
// single tile (golden: scripts/parakeet_relpos_mha_golden.py, G6/G7).
//
// L1 BUDGET (per query tile): AC_tile[Tq,T] f32 + BD_tile[Tq,P] f32 + probs[Tq,T]
// bf16 + one streamed k/p/V block. At Tq=8, T=172: g_ac 5.5 KB + g_bd 11 KB +
// g_probs 2.75 KB = ~19 KB of resident score scratch (vs 172 KB for full k/p/V) --
// the score/prob tiles shrink Tq/T-fold, and k/p/V are STREAMED (never fully
// resident). The MemTile holds the 172 KB k/p/V staging (fits the 512 KB L2), and
// the L2->L1 DMA is REPLAYED once per query tile (IRON ObjectFifo repeat_count),
// so k/p/V are fetched from host DDR ONCE, not per tile.
// ===========================================================================

// Encoder frame count T baked for the row-tiled block. Real Parakeet blocks run
// T' up to 172 (P = 2T-1 = 343). Override with -DRELPOS_T at compile time.
// (RELPOS_T also feeds the small-T standalone/composed de-risk entries above.)

// Query-tile row count Tq for the row-tiled block. Chosen so the per-tile score
// scratch (AC_tile[Tq,T] + BD_tile[Tq,P] f32) fits L1 with margin. Tq need NOT
// divide T -- the driver handles a ragged final tile (tq = min(Tq, T-q0)).
#ifndef RELPOS_TQ
#define RELPOS_TQ 8
#endif

// ---------------------------------------------------------------------------
// BRICK 2 (row-tiled) -- rel_shift + score add/scale + exp2 softmax, GENERALIZED
// to a query-tile row count Tq DISTINCT from the key count T, with the rel_shift
// window keyed on the GLOBAL query index (base row q0). Identical numerics to
// relpos_scores_softmax; the ONLY change is Tq rows and the (T-1)-(q0+il) offset.
//   AC   : [Tq, T] row-major f32  (qu_tile @ k^T)
//   BD   : [Tq, P] row-major f32  (qv_tile @ p^T, P = 2T-1)
//   probs: [Tq, T] row-major bf16 (out)
//   q0   : global row index of this tile's first row (i_global = q0 + il)
// Per local row il:  scores[j] = (AC[il,j] + BD[il, (T-1)-(q0+il)+j]) * inv_scale
// ---------------------------------------------------------------------------
static inline void relpos_scores_softmax_rows(float *restrict AC,
                                              float *restrict BD,
                                              bfloat16 *restrict probs, int Tq,
                                              int T, int P, int q0,
                                              float inv_scale) {
  alignas(aie::vector_decl_align) static float srow[RELPOS_TMAX];
  aie::vector<float, VL> inv_scale_v = aie::broadcast<float, VL>(inv_scale);
  aie::vector<float, VL> log2e_v = aie::broadcast<float, VL>(LOG2E);

  for (int il = 0; il < Tq; il++) {
    const float *ac_row = AC + il * T;
    // GLOBAL-INDEX rel_shift: contiguous length-T window of BD row il starting at
    // column (T-1) - (q0 + il). q0=0 (single tile) recovers the (T-1-il) form.
    const float *bd_row = BD + il * P + (T - 1 - (q0 + il));
    bfloat16 *prob_row = probs + il * T;

    // pass 1: scores = (AC + BD_shifted) * inv_scale ; row max (f32)
    float rowmax = -3.0e38f;
    int j = 0;
    for (; j + VL <= T; j += VL) {
      // ac_row/prob_row have row stride T (or P), which is NOT a multiple of VL for
      // real T (e.g. T=172 -> 172%16=12), so the per-row base is unaligned. Aligned
      // load_v/store_v truncate to 128b and corrupt -> use unaligned everywhere the
      // stride is T/P (same root cause as the bd_row rel_shift load above).
      aie::vector<float, VL> a = aie::load_unaligned_v<VL>(ac_row + j);
      // bd_row = BD + i*P + (T-1-i): the rel_shift base is NEVER VL-aligned (the
      // (T-1-i) shift is not a multiple of 16), so this MUST be an unaligned load.
      // aie::load_v is an ALIGNED load -> on aie2p it truncates the address to the
      // 128b boundary and returns shifted/garbage BD (masked when BD<<AC, e.g. real
      // block-0 after rescale; exposed the moment BD ~ AC, e.g. synth / spread).
      aie::vector<float, VL> b = aie::load_unaligned_v<VL>(bd_row + j);
      aie::vector<float, VL> s =
          aie::mul(aie::add(a, b), inv_scale_v).to_vector<float>();
      aie::store_v(srow + j, s);
      float cm = aie::reduce_max(s);
      if (cm > rowmax) rowmax = cm;
    }
    for (; j < T; j++) {
      float s = (ac_row[j] + bd_row[j]) * inv_scale;
      srow[j] = s;
      if (s > rowmax) rowmax = s;
    }

    // pass 2: exp2((s - max) * log2e) -> bf16 probs ; running f32 sum
    aie::vector<float, VL> maxv = aie::broadcast<float, VL>(rowmax);
    aie::accum<accfloat, VL> sumacc = aie::zeros<accfloat, VL>();
    j = 0;
    for (; j + VL <= T; j += VL) {
      aie::vector<float, VL> s = aie::load_v<VL>(srow + j);
      aie::vector<float, VL> d = aie::sub(s, maxv);
      aie::vector<float, VL> sl = aie::mul(d, log2e_v).to_vector<float>();
      aie::vector<bfloat16, VL> e = aie::exp2<bfloat16>(sl);
      aie::store_unaligned_v(prob_row + j, e);
      sumacc = aie::add(sumacc, e);
    }
    float sum = aie::reduce_add(sumacc.to_vector<float>());
    for (; j < T; j++) {
      float e = exp2_scalar((srow[j] - rowmax) * LOG2E);
      prob_row[j] = (bfloat16)e;
      sum += e;
    }

    // pass 3: normalize probs *= 1/sum
    bfloat16 inv_sum = (bfloat16)aie::inv(sum);
    aie::vector<bfloat16, VL> inv_sum_v = aie::broadcast<bfloat16, VL>(inv_sum);
    j = 0;
    for (; j + VL <= T; j += VL) {
      aie::vector<bfloat16, VL> e = aie::load_unaligned_v<VL>(prob_row + j);
      aie::store_unaligned_v(prob_row + j,
                             aie::mul(e, inv_sum_v).to_vector<bfloat16>());
    }
    for (; j < T; j++) {
      prob_row[j] = (bfloat16)((float)prob_row[j] * (float)inv_sum);
    }
  }
}

// BRICK 4 (row-tiled) -- ctx = probs @ V, GENERALIZED to a query-tile row count
// Tq distinct from the key count T. ctx[il,:] = sum_j probs[il,j] * V[j,:].
//   probs: [Tq, T]  row-major bf16   V: [T, DK] row-major bf16
//   ctx  : [Tq, DK] row-major bf16
static inline void relpos_ctx_matmul_rows(const bfloat16 *restrict probs,
                                          const bfloat16 *restrict V,
                                          bfloat16 *restrict ctx, int Tq, int T) {
  static_assert(DK % VL == 0, "DK must be a multiple of the vector width");
  for (int i = 0; i < Tq; i++) {
    const bfloat16 *p_row = probs + i * T;
    bfloat16 *ctx_row = ctx + i * DK;
    for (int d = 0; d < DK; d += VL) {
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int j = 0; j < T; j++) {
        aie::vector<bfloat16, VL> wv = aie::broadcast<bfloat16, VL>(p_row[j]);
        aie::vector<bfloat16, VL> vv = aie::load_v<VL>(V + j * DK + d);
        acc = aie::mac(acc, wv, vv);
      }
      aie::store_v(ctx_row + d, acc.to_vector<bfloat16>());
    }
  }
}

// ---------------------------------------------------------------------------
// STEP-6 ROW-TILED DRIVER -- one head, T query rows processed in Tq-row tiles,
// reading the resident k/p/V. This is the on-chip ARITHMETIC of the row-tiled
// block; it is what the numpy row-tiled golden (G6/G7) mirrors. Reuses the four
// device-validated bricks (dot_matmul x2, softmax_rows, ctx_matmul_rows) on the
// [Tq,*] tiles, with the GLOBAL-index rel_shift (q0) -- the load-bearing new bit.
//
//   quv : [2*T, DK]     bf16  PACKED (qu = quv[0:T], qv = quv[T:2T])
//   kpv : [(2*T+P), DK] bf16  PACKED (k = kpv[0:T], p = kpv[T:T+P], V = kpv[T+P:])
//   ctx : [T, DK]       bf16  (out)
// TWO packed inputs keep the core within the NPU2 compute tile's 2 input-DMA-
// channel budget (same discipline as steps 3/5); the pos_bias adds (qu=q+u,
// qv=q+v) are folded host-side. kpv is the RESIDENT staging (k/p/V); quv is the
// per-block query stream.
//
// Score/prob scratch is sized by Tq (NOT T): g_ac[Tq*T] + g_bd[Tq*P] + g_probs
// [Tq*T]. This monolithic bake takes FULL k/p/V pointers (kpv resident in L1), so
// on its own it caps at the T where kpv still fits L1 (bring-up / arithmetic gate
// -- but note the Tq-shrunk score scratch already raises that ceiling vs step-5
// relpos_full_bake). Reaching T=172 needs k/p/V STREAMED in key-blocks from the
// 512 KB MemTile (p alone is 86 KB > L1), which the row-tiled generator stages via
// ObjectFifo repeat_count; that block-streamed variant is the device-gate step
// (see relpos_rowtiled_iron.py + BUILD-AND-BENCH.md). The query-row loop and its
// GLOBAL-index rel_shift are bit-identical either way (golden G6/G7).
// ---------------------------------------------------------------------------
extern "C" void relpos_rowtiled_bake(bfloat16 *restrict quv,
                                     bfloat16 *restrict kpv,
                                     bfloat16 *restrict ctx) {
  constexpr int T = RELPOS_T;
  constexpr int P = 2 * T - 1;
  constexpr int Tq = RELPOS_TQ;
  static_assert(DK == 128,
                "baked inv_scale is 1/sqrt(128); regenerate for a different DK");
  constexpr float inv_scale = 0.08838834764831843f; // 1/sqrt(128)

  alignas(aie::vector_decl_align) static float g_ac[Tq * T];
  alignas(aie::vector_decl_align) static float g_bd[Tq * P];
  alignas(aie::vector_decl_align) static bfloat16 g_probs[Tq * T];

  bfloat16 *qu = quv;                  // [T, DK]
  bfloat16 *qv = quv + T * DK;         // [T, DK]
  bfloat16 *k = kpv;                   // [T, DK]
  bfloat16 *p = kpv + T * DK;          // [P, DK]
  bfloat16 *V = kpv + (T + P) * DK;    // [T, DK]

  event0();
  for (int q0 = 0; q0 < T; q0 += Tq) {
    int tq = (T - q0 < Tq) ? (T - q0) : Tq; // ragged final tile
    const bfloat16 *qu_t = qu + q0 * DK;     // [tq, DK]
    const bfloat16 *qv_t = qv + q0 * DK;     // [tq, DK]
    bfloat16 *ctx_t = ctx + q0 * DK;         // [tq, DK]
    relpos_dot_matmul(qu_t, k, g_ac, tq, T); // AC_tile = qu_t @ k^T  [tq,T]
    relpos_dot_matmul(qv_t, p, g_bd, tq, P); // BD_tile = qv_t @ p^T  [tq,P]
    relpos_scores_softmax_rows(g_ac, g_bd, g_probs, tq, T, P, q0, inv_scale);
    relpos_ctx_matmul_rows(g_probs, V, ctx_t, tq, T); // ctx_tile = probs @ V
  }
  event1();
}

// ===========================================================================
// STEP-7 KEY-BLOCK-STREAMED bricks + monolithic reference driver.
//
// WHY: step-6 relpos_rowtiled_bake still takes FULL k/p/V pointers (kpv resident
// in L1), so it caps at the T where kpv fits L1. p alone is 86 KB > 64 KB L1 at
// T=172. The dataflow fix (relpos_rowtiled_stream_iron.py) STREAMS k/p/V from the
// 512 KB MemTile in KB-row key-blocks, so L1 only ever holds ONE key-block plus
// the Tq-sized score/prob/ctx scratch. That streaming core must consume k/p/V a
// block at a time and ASSEMBLE the full [Tq,*] score rows across blocks before the
// softmax (design (a): the score rows fit L1, the INPUT k/p/V do not).
//
// This section provides the per-block compute bricks the streaming core calls,
// and a MONOLITHIC driver (relpos_kpvstream_bake) that walks query-tiles x
// key-blocks calling exactly those bricks with the SAME packed (quv, kpv) ABI as
// step-6. The monolithic driver is buildable/device-gateable at the T where kpv
// fits L1 (same runner, STEP=7): it de-risks the BLOCK-DECOMPOSED ARITHMETIC
// (column-slice AC/BD fill + f32-running ctx accumulate across V-blocks) on
// silicon, independent of the MemTile dataflow wiring. Once the arithmetic is
// gated here, relpos_rowtiled_stream_iron.py only has to prove the DATAFLOW
// (the 2-channel ObjectFifo topology + repeat_count replay) at T=172.
//
// NUMERICS vs the proven step-6 bricks:
//  * AC/BD: relpos_dot_block fills a COLUMN SLICE [Tq, kb] of the row-major score
//    row at column j0. Each output element is a single full-DK dot (DK is NOT
//    blocked) -> BIT-IDENTICAL to relpos_dot_matmul; only the key dim is tiled and
//    each key column is independent.
//  * ctx: relpos_ctx_block accumulates each V-block's partial into a RESIDENT f32
//    ctx buffer (g_ctxf[Tq*DK]), narrowed to bf16 once at the end. The proven
//    relpos_ctx_matmul_rows keeps ONE accfloat over all T keys; block-summing in
//    f32 re-associates the reduction but stays in f32 the whole way (the bf16 hop
//    is only the final narrow, same as the proven brick), so the delta is far
//    below the ctx rel-L2 tolerance (golden 3.8e-3). This is the only numeric
//    difference from step-6 and it is a strict precision non-regression.
// ===========================================================================

// Key-block row count baked for the streamed block bricks. T=172 = 4*43, so
// KB=43 blocks k and V with NO padding; p (P=343) is 7 full blocks + a 42-row
// ragged tail (the driver/core pass the real kb per block). Override -DRELPOS_KB.
#ifndef RELPOS_KB
#define RELPOS_KB 43
#endif

// COLUMN-SLICE dot-matmul: out[il, j0+jj] = dot(A[il,:DK], Bblk[jj,:DK]) for
// il in [0,Tq), jj in [0,kb). out is row-major [Tq, ncol] (ncol = T for AC,
// P for BD); this fills the kb columns [j0, j0+kb) of every row. A is the
// resident query tile ([Tq,DK]); Bblk is ONE streamed key/pos block ([kb,DK]).
static inline void relpos_dot_block(const bfloat16 *restrict A,
                                    const bfloat16 *restrict Bblk,
                                    float *restrict out, int Tq, int kb, int j0,
                                    int ncol) {
  static_assert(DK % VL == 0, "DK must be a multiple of the vector width");
  for (int il = 0; il < Tq; il++) {
    const bfloat16 *a_row = A + il * DK;
    float *o_row = out + il * ncol + j0;
    for (int jj = 0; jj < kb; jj++) {
      const bfloat16 *b_row = Bblk + jj * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL) {
        aie::vector<bfloat16, VL> av = aie::load_v<VL>(a_row + d);
        aie::vector<bfloat16, VL> bv = aie::load_v<VL>(b_row + d);
        acc = aie::mac(acc, av, bv);
      }
      o_row[jj] = aie::reduce_add(acc.to_vector<float>());
    }
  }
}

// Zero the resident f32 ctx accumulator [Tq, DK] at the start of a query tile.
static inline void relpos_ctx_zero(float *restrict ctxf, int Tq) {
  aie::vector<float, VL> z = aie::zeros<float, VL>();
  for (int il = 0; il < Tq; il++)
    for (int d = 0; d < DK; d += VL)
      aie::store_v(ctxf + il * DK + d, z);
}

// ctx BLOCK accumulate: ctxf[il,:DK] += sum_{jj<kb} probs[il, j0+jj] * Vblk[jj,:].
// probs is the resident [Tq,T] softmax output; Vblk is ONE streamed value block
// ([kb,DK]); ctxf is the resident f32 running accumulator. Called once per
// V-block; the running f32 buffer is the cross-block carry the streaming core
// cannot hold in a register (separate kernel calls).
static inline void relpos_ctx_block(const bfloat16 *restrict probs,
                                    const bfloat16 *restrict Vblk,
                                    float *restrict ctxf, int Tq, int T, int kb,
                                    int j0) {
  static_assert(DK % VL == 0, "DK must be a multiple of the vector width");
  for (int il = 0; il < Tq; il++) {
    const bfloat16 *p_row = probs + il * T + j0;
    float *c_row = ctxf + il * DK;
    for (int d = 0; d < DK; d += VL) {
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int jj = 0; jj < kb; jj++) {
        aie::vector<bfloat16, VL> wv = aie::broadcast<bfloat16, VL>(p_row[jj]);
        aie::vector<bfloat16, VL> vv = aie::load_v<VL>(Vblk + jj * DK + d);
        acc = aie::mac(acc, wv, vv); // bf16*bf16 -> f32 block partial
      }
      // add block partial to the running f32 ctx (stays in f32 across blocks).
      aie::vector<float, VL> part = acc.to_vector<float>();
      aie::vector<float, VL> run = aie::load_v<VL>(c_row + d);
      aie::store_v(c_row + d, aie::add(run, part));
    }
  }
}

// Narrow the resident f32 ctx accumulator [Tq,DK] -> bf16 ctx tile (final hop,
// same single bf16 narrow as the proven ctx brick). ctx is the output tile.
static inline void relpos_ctx_narrow(const float *restrict ctxf,
                                     bfloat16 *restrict ctx, int Tq) {
  for (int il = 0; il < Tq; il++) {
    for (int d = 0; d < DK; d += VL) {
      aie::vector<float, VL> v = aie::load_v<VL>(ctxf + il * DK + d);
      aie::accum<accfloat, VL> a;
      a.from_vector(v); // f32 vector -> accum -> bf16 narrow (proven idiom)
      aie::store_v(ctx + il * DK + d, a.to_vector<bfloat16>());
    }
  }
}

// ---------------------------------------------------------------------------
// STEP-7 MONOLITHIC REFERENCE DRIVER -- walks query-tiles x key-blocks calling
// the streamed block bricks, with FULL k/p/V pointers (kpv resident, same packed
// ABI as step-6 relpos_rowtiled_bake). This reproduces the EXACT accumulation
// order the streaming core will execute (AC/BD filled column-slice per k/p block;
// ctx accumulated in a resident f32 buffer across V-blocks), so gating it on the
// existing runner (STEP=7, at the T where kpv fits L1) validates the block bricks
// on silicon. The block LOOPS here are what the streaming core replaces with
// ObjectFifo acquire/release over MemTile-streamed blocks (same brick calls).
//   quv : [2*T, DK]     bf16  PACKED (qu = quv[0:T], qv = quv[T:2T])
//   kpv : [(2*T+P), DK] bf16  PACKED (k = kpv[0:T], p = kpv[T:T+P], V = kpv[T+P:])
//   ctx : [T, DK]       bf16  (out)
// ---------------------------------------------------------------------------
extern "C" void relpos_kpvstream_bake(bfloat16 *restrict quv,
                                      bfloat16 *restrict kpv,
                                      bfloat16 *restrict ctx) {
  constexpr int T = RELPOS_T;
  constexpr int P = 2 * T - 1;
  constexpr int Tq = RELPOS_TQ;
  constexpr int KB = RELPOS_KB;
  static_assert(DK == 128,
                "baked inv_scale is 1/sqrt(128); regenerate for a different DK");
  constexpr float inv_scale = 0.08838834764831843f; // 1/sqrt(128)

  alignas(aie::vector_decl_align) static float g_ac[Tq * T];
  alignas(aie::vector_decl_align) static float g_bd[Tq * P];
  alignas(aie::vector_decl_align) static bfloat16 g_probs[Tq * T];
  alignas(aie::vector_decl_align) static float g_ctxf[Tq * DK];

  bfloat16 *qu = quv;               // [T, DK]
  bfloat16 *qv = quv + T * DK;      // [T, DK]
  bfloat16 *k = kpv;                // [T, DK]
  bfloat16 *p = kpv + T * DK;       // [P, DK]
  bfloat16 *V = kpv + (T + P) * DK; // [T, DK]

  event0();
  for (int q0 = 0; q0 < T; q0 += Tq) {
    int tq = (T - q0 < Tq) ? (T - q0) : Tq; // ragged final query tile
    const bfloat16 *qu_t = qu + q0 * DK;    // [tq, DK]
    const bfloat16 *qv_t = qv + q0 * DK;    // [tq, DK]
    bfloat16 *ctx_t = ctx + q0 * DK;        // [tq, DK]

    // -- phase K: stream k in KB-row blocks, fill AC[:, j0:j0+kb] --
    for (int j0 = 0; j0 < T; j0 += KB) {
      int kb = (T - j0 < KB) ? (T - j0) : KB;
      relpos_dot_block(qu_t, k + j0 * DK, g_ac, tq, kb, j0, T);
    }
    // -- phase P: stream p in KB-row blocks, fill BD[:, j0:j0+kb] --
    for (int j0 = 0; j0 < P; j0 += KB) {
      int pb = (P - j0 < KB) ? (P - j0) : KB;
      relpos_dot_block(qv_t, p + j0 * DK, g_bd, tq, pb, j0, P);
    }
    // -- softmax over the assembled full score rows (GLOBAL-index rel_shift) --
    relpos_scores_softmax_rows(g_ac, g_bd, g_probs, tq, T, P, q0, inv_scale);
    // -- phase V: stream V in KB-row blocks, accumulate ctx in resident f32 --
    relpos_ctx_zero(g_ctxf, tq);
    for (int j0 = 0; j0 < T; j0 += KB) {
      int vb = (T - j0 < KB) ? (T - j0) : KB;
      relpos_ctx_block(g_probs, V + j0 * DK, g_ctxf, tq, T, vb, j0);
    }
    relpos_ctx_narrow(g_ctxf, ctx_t, tq); // f32 ctx -> bf16 out tile
  }
  event1();
}

// ---------------------------------------------------------------------------
// STEP-7 STREAMING-CORE extern "C" bricks -- the SAME block compute the
// monolithic driver calls, exposed as separately-callable kernels for the
// MemTile-streaming core (relpos_rowtiled_stream_iron.py). The streaming core
// drives the query-tile x key-block loop in the IRON Worker (ObjectFifo
// acquire/release per block) and calls these per block, passing the resident
// scratch buffers (g_ac/g_bd/g_probs/g_ctxf as core-local Buffers) plus the
// runtime block descriptors (Tq, kb, j0, ...) as int32 scalars.
//
// The int32 scalar ABI here is deliberate: it is the ONE thing the streaming
// generator needs the IRON toolchain to support -- passing a range_-loop-derived
// int32 into a Kernel call (see relpos_rowtiled_stream_iron.py "PROBE 1"). The
// compute itself is identical to the device-gateable monolithic driver above.
// ---------------------------------------------------------------------------
extern "C" void relpos_stream_dot(bfloat16 *restrict Aq, bfloat16 *restrict Bblk,
                                  float *restrict out, int32_t Tq, int32_t kb,
                                  int32_t j0, int32_t ncol) {
  event0();
  relpos_dot_block(Aq, Bblk, out, Tq, kb, j0, ncol);
  event1();
}

// Distinct symbol for the BD (p-block) call so the generator can declare a second
// Kernel with the [Tq,P] output type without redefining relpos_stream_dot's symbol
// (IRON emits one func.func per Kernel object; same symbol + different type => MLIR
// redefinition). Identical compute.
extern "C" void relpos_stream_dot_p(bfloat16 *restrict Aq, bfloat16 *restrict Bblk,
                                    float *restrict out, int32_t Tq, int32_t pb,
                                    int32_t j0, int32_t ncol) {
  event0();
  relpos_dot_block(Aq, Bblk, out, Tq, pb, j0, ncol);
  event1();
}

extern "C" void relpos_stream_ctx_zero(float *restrict ctxf, int32_t Tq) {
  relpos_ctx_zero(ctxf, Tq);
}

extern "C" void relpos_stream_ctx(bfloat16 *restrict probs,
                                  bfloat16 *restrict Vblk, float *restrict ctxf,
                                  int32_t Tq, int32_t T, int32_t kb, int32_t j0) {
  event0();
  relpos_ctx_block(probs, Vblk, ctxf, Tq, T, kb, j0);
  event1();
}

extern "C" void relpos_stream_softmax(float *restrict AC, float *restrict BD,
                                      bfloat16 *restrict probs, int32_t Tq,
                                      int32_t T, int32_t P, int32_t q0) {
  static_assert(DK == 128, "baked inv_scale is 1/sqrt(128)");
  event0();
  relpos_scores_softmax_rows(AC, BD, probs, Tq, T, P, q0, 0.08838834764831843f);
  event1();
}

extern "C" void relpos_stream_narrow(float *restrict ctxf, bfloat16 *restrict ctx,
                                     int32_t Tq) {
  relpos_ctx_narrow(ctxf, ctx, Tq);
}
