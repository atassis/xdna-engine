//===- lm_head_argmax.cc --------------------------------------*- C++ -*-===//
//
// BRICK: lm-head-argmax  [group: head]
//
// Generic op-TYPE against the resident [tile, D] stream contract: vocab
// projection GEMM (hidden state -> logits) FUSED with argmax reduction, so
// the decode head never round-trips the full [1, VOCAB] logits tensor back
// to host just to find one scalar token id. Only (best_logit, best_index)
// leave the tile.
//
//   logits[1, VOCAB] = hidden[1, HIDDEN] @ W[HIDDEN, VOCAB]      (bf16 in, f32 acc)
//   (best_value, best_index) = argmax_v(logits)                  (max_cmp co-produce)
//
// This is NOT a per-model (e.g. Gemma-270M / FLM) hand-fused kernel: HIDDEN,
// VOCAB, and the GEMM tiling are compile-time knobs (K_TILE/N_TILE), and the
// argmax reduction tree is independent of both — any resident [tile, D] head
// projection can instantiate this brick by picking HIDDEN/VOCAB.
//
// ---------------------------------------------------------------------------
// Dataflow (decode, M=1 padded to M_PAD rows; row 0 = the live query, rows
// 1..M_PAD-1 are zero-padded so the mmul<M,K,N> tile shape is met — same
// padding convention as route_b_kernels/decode_norm_gemv/norm_gemv_iron.py):
//
//   for each VOCAB tile of width N_TILE:
//     acc = 0                                              (accauto, f32)
//     for each HIDDEN tile of width K_TILE:
//       A = load_v<M_PAD*K_TILE>(hidden_tile)               bf16
//       B = load_v<K_TILE*N_TILE>(weight_tile)              bf16
//       acc.mac(A, B)                                       aie::mmul chain
//     logits_chunk = acc.to_vector<float>()                 [M_PAD*N_TILE] f32
//     (running_val, running_idx) =
//         reduce_chunk(running_val, running_idx, logits_chunk, vocab_offset)
//   (best_value, best_index) = horizontal_reduce(running_val, running_idx)
//
// The API pattern (aie::mmul<M,K,N,TA,TB,accauto> -> load_v -> acc.mac ->
// acc.to_vector<Out>()) is copied verbatim from
// route_b_kernels/ffn_bfp16/repro_847_mmul888_bfp16.cc. The co-produced
// (value, mask) argmax primitive is aie::max_cmp, whose mask semantics
// (m[i]==0 -> keep v1[i], m[i]==1 -> take v2[i]) are documented in
// aie_api/include/aie_api/aie.hpp and mirrored exactly by aie::select, so a
// SINGLE mask drives both the running-max vector merge and the running-index
// vector merge -> value and index never disagree.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// ---- compile-time brick parameters (generic knobs, not per-model) --------
#ifndef LMHEAD_HIDDEN
#define LMHEAD_HIDDEN 256   // HIDDEN dim (K); must be a multiple of K_TILE
#endif
#ifndef LMHEAD_VOCAB
#define LMHEAD_VOCAB 512    // VOCAB dim (N); must be a multiple of N_TILE
#endif
#ifndef LMHEAD_M_PAD
#define LMHEAD_M_PAD 8      // padded decode row count (row 0 = live query)
#endif

namespace lm_head_argmax_detail {

constexpr unsigned kHidden = LMHEAD_HIDDEN;
constexpr unsigned kVocab = LMHEAD_VOCAB;
constexpr unsigned kMPad = LMHEAD_M_PAD;
constexpr unsigned kKTile = 8;   // mmul K tile width (aie2p bf16 native mmul shape)
constexpr unsigned kNTile = 8;   // mmul N tile width
static_assert(kHidden % kKTile == 0, "HIDDEN must tile evenly by K_TILE");
static_assert(kVocab % kNTile == 0, "VOCAB must tile evenly by N_TILE");

using MMUL = aie::mmul<kMPad, kKTile, kNTile, bfloat16, bfloat16, accauto>;

// One VOCAB-tile GEMM: hidden[kMPad, kHidden] (bf16, resident) x
// weight_tile[kHidden, kNTile] (bf16, resident, column-major-by-tile so each
// K-subtile of kKTile*kNTile elements is contiguous) -> logits_chunk
// [kMPad*kNTile] f32.
inline aie::vector<float, MMUL::size_C> gemm_vocab_tile(
    const bfloat16 *__restrict hidden,       // [kMPad, kHidden], bf16, resident
    const bfloat16 *__restrict weight_tile) { // [kHidden, kNTile] bf16, resident
  MMUL acc;
  for (unsigned k = 0; k < kHidden; k += kKTile) {
    aie::vector<bfloat16, MMUL::size_A> A =
        aie::load_v<MMUL::size_A>(hidden + k * kMPad);
    aie::vector<bfloat16, MMUL::size_B> B =
        aie::load_v<MMUL::size_B>(weight_tile + k * kNTile);
    if (k == 0)
      acc.mul(A, B);
    else
      acc.mac(A, B);
  }
  return acc.template to_vector<float>();
}

// Merge one freshly-computed logits chunk (row 0 of the kMPad-padded tile,
// i.e. the live query's kNTile logits) into the running (value, index)
// vectors using aie::max_cmp. A single mask drives both merges, so value and
// index are always co-produced / never drift apart.
inline void reduce_chunk(aie::vector<float, kNTile> &running_val,
                          aie::vector<int32_t, kNTile> &running_idx,
                          const aie::vector<float, MMUL::size_C> &logits_chunk,
                          unsigned vocab_offset) {
  // Row 0 of the [kMPad, kNTile] accumulator is the live query's logits
  // (rows 1..kMPad-1 come from the zero-padded decode rows and are unused,
  // matching the decode_norm_gemv M-padding convention).
  aie::vector<float, kNTile> chunk_val =
      logits_chunk.template extract<kNTile>(0);

  int32_t idx_local[kNTile];
  for (unsigned i = 0; i < kNTile; ++i)
    idx_local[i] = (int32_t)(vocab_offset + i);
  aie::vector<int32_t, kNTile> chunk_idx = aie::load_v<kNTile>(idx_local);

  auto [merged_val, m] = aie::max_cmp(running_val, chunk_val);
  running_val = merged_val;
  // m[i] == 1 means chunk_val[i] won -> take chunk_idx[i]; m[i] == 0 keeps
  // running_idx[i] (aie::select has the same 0->v1 / 1->v2 convention).
  running_idx = aie::select(running_idx, chunk_idx, m);
}

// Final small horizontal reduction across the kNTile-wide running vectors
// down to one scalar (value, index). kNTile is tiny (8) so a scalar tail
// loop is the standard aie_api idiom (cf. reduce_max/reduce_add callers) and
// costs nothing next to the O(VOCAB/N_TILE) tiled GEMM+reduce above.
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

}  // namespace lm_head_argmax_detail

extern "C" {

// hidden:      [kMPad, kHidden] bf16, resident (row 0 = live decode query,
//              rows 1..kMPad-1 zero-padded — decode M=1 padding convention).
// weight:      [kHidden, kVocab] bf16, resident, laid out as kVocab/kNTile
//              column tiles, each tile [kHidden, kNTile] with kKTile*kNTile
//              contiguous per K-subtile (matches gemm_vocab_tile's load_v).
// best_value:  out, f32 scalar — logit at the argmax position.
// best_index:  out, int32 scalar — argmax token id in [0, kVocab).
void lm_head_argmax(const bfloat16 *__restrict hidden,
                     const bfloat16 *__restrict weight, float *__restrict best_value,
                     int32_t *__restrict best_index) {
  using namespace lm_head_argmax_detail;
  event0();

  aie::vector<float, kNTile> running_val = aie::broadcast<float, kNTile>(-3.0e38f);
  aie::vector<int32_t, kNTile> running_idx = aie::zeros<int32_t, kNTile>();

  for (unsigned n = 0; n < kVocab; n += kNTile) {
    const bfloat16 *weight_tile = weight + n * kHidden;
    aie::vector<float, MMUL::size_C> logits_chunk =
        gemm_vocab_tile(hidden, weight_tile);
    reduce_chunk(running_val, running_idx, logits_chunk, n);
  }

  float value;
  int32_t index;
  horizontal_reduce(running_val, running_idx, value, index);
  *best_value = value;
  *best_index = index;

  event1();
}

}  // extern "C"
