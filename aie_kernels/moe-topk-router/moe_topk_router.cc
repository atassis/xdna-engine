//===- moe_topk_router.cc -------------------------------------*- C++ -*-===//
//
// BRICK: moe-topk-router  [group: spike]
//
// Generic op-TYPE against the resident [tile, D] stream contract: MoE gate
// projection GEMM (hidden state -> per-expert logits) FUSED with a top-K
// select, so a decode-time MoE layer never round-trips the full
// [1, NUM_EXPERTS] gate-logits tensor back to host just to find the handful
// of experts that will actually run.
//
//   logits[1, EXPERTS] = hidden[1, HIDDEN] @ Wg[HIDDEN, EXPERTS]  (bf16 in, f32 acc)
//   (top_values[K], top_indices[K]) = topk(logits, K)             (max_cmp, iterated)
//
// This is the SAME gemm+argmax skeleton as bricks/lm-head-argmax
// (lm_head_argmax.cc), which co-produces a SINGLE (value, index) via
// aie::max_cmp + aie::select. The extension asked for by this spike is:
// "extend max_cmp beyond k=1". The generic, parameterized way to do that
// (rather than hand-rolling a K-wide bitonic/sorting network kernel) is
// SELECTION-SORT-BY-ARGMAX:
//
//   for iter in 0..K-1:
//     (best_value, best_index) = argmax_full_reduce(logits_buf)   // same
//                                                                  // max_cmp
//                                                                  // reduction
//                                                                  // tree as
//                                                                  // lm-head-
//                                                                  // argmax,
//                                                                  // just
//                                                                  // scanning
//                                                                  // EXPERTS
//                                                                  // (small)
//                                                                  // instead
//                                                                  // of VOCAB
//                                                                  // (huge)
//     top_values[iter]  = best_value
//     top_indices[iter] = best_index
//     logits_buf[best_index] = -inf                                // mask so
//                                                                   // the next
//                                                                   // pass
//                                                                   // finds the
//                                                                   // NEXT best
//
// K argmax passes over EXPERTS is O(K * EXPERTS); for the MoE router regime
// (EXPERTS ~= 8..128, K ~= 1..8, e.g. Mixtral-style top-2 of 8) that is a
// handful of vector ops, not a bottleneck -- the GEMM (HIDDEN x EXPERTS) and
// the eventual per-expert FFN dispatch dominate. This keeps the primitive a
// GENERIC op-TYPE: EXPERTS, HIDDEN and K are compile-time knobs, never a
// per-model hand-fused router.
//
// The masking step reuses the exact same (value, index) merge convention as
// lm_head_argmax's reduce_chunk: aie::max_cmp produces a merged value vector
// plus a 0/1 mask, and aie::select (same mask) merges the index vector, so
// value and index can never disagree. The one addition beyond lm-head-argmax
// is that after each full-array argmax we write -inf back into the SAME
// resident logits buffer at the winning index (single scalar store) before
// re-scanning -- that scalar store is the only new "op" this brick adds to
// the lm-head-argmax skeleton.
//
// Router-weight softmax over the top-K logits (the usual "normalize the
// K survivors, zero the rest" MoE combine step) is DELIBERATELY NOT done
// on-device here: this target has no libm on the AIE2P freestanding
// toolchain (no expf/cmath, confirmed by grepping every other kernel in this
// tree) and no SFU exp2 identity has been exercised by this tree yet. Softmax
// normalization of top_values[K] belongs in a *separate* softmax-over-K
// brick (K is tiny, so it can even be done host-side or in the FFN-dispatch
// core without cost) -- see the findings note for the full read.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// ---- compile-time brick parameters (generic knobs, not per-model) --------
#ifndef MOE_HIDDEN
#define MOE_HIDDEN 256   // HIDDEN dim (K of the gate GEMM); multiple of K_TILE
#endif
#ifndef MOE_EXPERTS
#define MOE_EXPERTS 64   // NUM_EXPERTS (N of the gate GEMM); multiple of N_TILE
#endif
#ifndef MOE_TOPK
#define MOE_TOPK 2       // K experts to route to per token (e.g. Mixtral top-2)
#endif
#ifndef MOE_M_PAD
#define MOE_M_PAD 8      // padded decode row count (row 0 = live query)
#endif

namespace moe_topk_router_detail {

constexpr unsigned kHidden = MOE_HIDDEN;
constexpr unsigned kExperts = MOE_EXPERTS;
constexpr unsigned kTopK = MOE_TOPK;
constexpr unsigned kMPad = MOE_M_PAD;
constexpr unsigned kKTile = 8;   // mmul K tile width (aie2p bf16 native mmul shape)
constexpr unsigned kNTile = 8;   // mmul N tile width
static_assert(kHidden % kKTile == 0, "HIDDEN must tile evenly by K_TILE");
static_assert(kExperts % kNTile == 0, "EXPERTS must tile evenly by N_TILE");
static_assert(kTopK >= 1 && kTopK <= kExperts, "0 < TOPK <= EXPERTS");

using MMUL = aie::mmul<kMPad, kKTile, kNTile, bfloat16, bfloat16, accauto>;

// One EXPERTS-tile GEMM: hidden[kMPad, kHidden] (bf16, resident) x
// gate_weight_tile[kHidden, kNTile] (bf16, resident, column-major-by-tile so
// each K-subtile of kKTile*kNTile elements is contiguous) -> logits_chunk
// [kMPad*kNTile] f32. Identical shape/idiom to lm_head_argmax's
// gemm_vocab_tile -- same brick skeleton, just N=EXPERTS instead of N=VOCAB.
inline aie::vector<float, MMUL::size_C> gemm_expert_tile(
    const bfloat16 *__restrict hidden,            // [kMPad, kHidden], bf16, resident
    const bfloat16 *__restrict gate_weight_tile) { // [kHidden, kNTile] bf16, resident
  MMUL acc;
  for (unsigned k = 0; k < kHidden; k += kKTile) {
    aie::vector<bfloat16, MMUL::size_A> A =
        aie::load_v<MMUL::size_A>(hidden + k * kMPad);
    aie::vector<bfloat16, MMUL::size_B> B =
        aie::load_v<MMUL::size_B>(gate_weight_tile + k * kNTile);
    if (k == 0)
      acc.mul(A, B);
    else
      acc.mac(A, B);
  }
  return acc.template to_vector<float>();
}

// Merge one freshly-computed logits chunk into the running (value, index)
// vectors using aie::max_cmp -- same co-produced merge as
// lm_head_argmax::reduce_chunk. A single mask drives both merges, so value
// and index never drift apart.
inline void reduce_chunk(aie::vector<float, kNTile> &running_val,
                          aie::vector<int32_t, kNTile> &running_idx,
                          const aie::vector<float, kNTile> &chunk_val,
                          unsigned expert_offset) {
  int32_t idx_local[kNTile];
  for (unsigned i = 0; i < kNTile; ++i)
    idx_local[i] = (int32_t)(expert_offset + i);
  aie::vector<int32_t, kNTile> chunk_idx = aie::load_v<kNTile>(idx_local);

  auto [merged_val, m] = aie::max_cmp(running_val, chunk_val);
  running_val = merged_val;
  // m[i] == 1 means chunk_val[i] won -> take chunk_idx[i]; m[i] == 0 keeps
  // running_idx[i] (aie::select has the same 0->v1 / 1->v2 convention).
  running_idx = aie::select(running_idx, chunk_idx, m);
}

// Final small horizontal reduction across the kNTile-wide running vectors
// down to one scalar (value, index). kNTile is tiny (8), so a scalar tail
// loop is the standard aie_api idiom (mirrors lm_head_argmax's
// horizontal_reduce).
inline void horizontal_reduce(const aie::vector<float, kNTile> &running_val,
                               const aie::vector<int32_t, kNTile> &running_idx,
                               float &best_value, int32_t &best_index) {
  best_value = running_val[0];
  best_index = running_idx[0];
  for (unsigned i = 1; i < kNTile; ++i) {
    float v = running_val[i];
    if (v > best_value) {
      best_value = v;
      best_index = running_idx[i];
    }
  }
}

// One full-array argmax pass over the resident logits_buf[kExperts] (row 0
// of the gate GEMM output). This is EXACTLY lm_head_argmax's top-level loop
// body (gemm-tile -> reduce_chunk -> horizontal_reduce), just reading a
// precomputed logits buffer instead of re-running the GEMM per iteration
// (the GEMM only needs to run once; the K argmax passes reuse its output).
inline void argmax_full_reduce(const float *__restrict logits_buf,
                                float &best_value, int32_t &best_index) {
  aie::vector<float, kNTile> running_val = aie::broadcast<float, kNTile>(-3.0e38f);
  aie::vector<int32_t, kNTile> running_idx = aie::zeros<int32_t, kNTile>();

  for (unsigned n = 0; n < kExperts; n += kNTile) {
    aie::vector<float, kNTile> chunk_val = aie::load_v<kNTile>(logits_buf + n);
    reduce_chunk(running_val, running_idx, chunk_val, n);
  }
  horizontal_reduce(running_val, running_idx, best_value, best_index);
}

}  // namespace moe_topk_router_detail

extern "C" {

// hidden:       [kMPad, kHidden] bf16, resident (row 0 = live decode query,
//               rows 1..kMPad-1 zero-padded -- decode M=1 padding convention,
//               same as lm_head_argmax / decode_norm_gemv).
// gate_weight:  [kHidden, kExperts] bf16, resident, laid out as
//               kExperts/kNTile column tiles, each tile [kHidden, kNTile]
//               with kKTile*kNTile contiguous per K-subtile (matches
//               gemm_expert_tile's load_v).
// top_values:   out, f32[kTopK] -- gate logits of the kTopK selected experts,
//               in DESCENDING order (top_values[0] is the largest).
// top_indices:  out, int32[kTopK] -- expert ids in [0, kExperts) matching
//               top_values, same order.
void moe_topk_router(const bfloat16 *__restrict hidden,
                      const bfloat16 *__restrict gate_weight,
                      float *__restrict top_values,
                      int32_t *__restrict top_indices) {
  using namespace moe_topk_router_detail;
  event0();

  // --- pass 0: gate GEMM, row 0 of the [kMPad, kExperts] result only -----
  // logits_buf lives in core-local scratch (kExperts is small: MoE routers
  // are 8..128-wide, never VOCAB-sized), so it round-trips through no DMA --
  // it is filled once by the GEMM below and then repeatedly re-scanned /
  // masked in place by the kTopK argmax passes.
  float logits_buf[kExperts];
  for (unsigned n = 0; n < kExperts; n += kNTile) {
    const bfloat16 *gate_weight_tile = gate_weight + n * kHidden;
    aie::vector<float, MMUL::size_C> chunk = gemm_expert_tile(hidden, gate_weight_tile);
    // Row 0 of the [kMPad, kNTile] accumulator is the live query's logits
    // (rows 1..kMPad-1 come from the zero-padded decode rows and are
    // unused, matching the decode_norm_gemv M-padding convention).
    aie::vector<float, kNTile> row0 = chunk.template extract<kNTile>(0);
    aie::store_v(logits_buf + n, row0);
  }

  // --- pass 1..kTopK: iterated argmax = "max_cmp extended beyond k=1" ----
  constexpr float kNegInf = -3.0e38f;
  for (unsigned iter = 0; iter < kTopK; ++iter) {
    float best_value;
    int32_t best_index;
    argmax_full_reduce(logits_buf, best_value, best_index);
    top_values[iter] = best_value;
    top_indices[iter] = best_index;
    logits_buf[best_index] = kNegInf;  // mask the winner so the next pass
                                        // finds the NEXT best (selection-sort
                                        // step; single scalar store, same
                                        // buffer, no DMA).
  }

  event1();
}

}  // extern "C"
