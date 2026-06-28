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
      aie::vector<float, VL> a = aie::load_v<VL>(ac_row + j);
      aie::vector<float, VL> b = aie::load_v<VL>(bd_row + j);
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
      aie::store_v(prob_row + j, e);
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
      aie::vector<bfloat16, VL> e = aie::load_v<VL>(prob_row + j);
      aie::store_v(prob_row + j, aie::mul(e, inv_sum_v).to_vector<bfloat16>());
    }
    for (; j < T; j++) {
      prob_row[j] = (bfloat16)((float)prob_row[j] * (float)inv_sum);
    }
  }
  event1();
}
