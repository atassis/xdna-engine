//===- mha_decode.cc ------------------------------------------*- C++ -*-===//
//
// M1 Task 0 — on-chip SINGLE-QUERY (M=1) multi-head attention for the Whisper
// decoder. Standalone parity kernel: validated against the host reference
// `attend_one` (rust/npu-engine/src/asr/whisper_decoder.rs) BEFORE integration.
//
// Whisper-small: D=768, n_heads=12, head_dim hd=64.
//
// STREAMING / FLASH design (forced by L1 capacity): a single compute tile has
// only ~64 KB data memory and 2 input DMA channels. The full per-head K+V at
// S=448 (2*448*64 bf16 = 114 KB) does NOT fit, let alone all 12 heads. So K/V
// are STREAMED in tiles of TKV keys and the softmax is ONLINE (flash-attention):
// per head we keep only the running max m, running denom l, and the f32 ctx
// accumulator acc[HD] across tiles — never the full score row.
//
// Per head h, over kv-tiles:
//   for each key s in the tile:
//     score = (q_h . K[s]) * (1/sqrt(64))           (bf16 in, f32 accum)
//     m_new = max(m, score)
//     correction = exp(m - m_new)                    (rescale prior state)
//     p = exp(score - m_new)
//     l   = l*correction + p
//     acc = acc*correction + p * V[s]                (f32 accumulator)
//     m   = m_new
//   ctx_h = acc / l   (emitted after the head's last tile)
//
// This is mathematically identical to the host's offline 3-pass softmax (verified
// by the probe) but needs only one TKV-key tile resident at a time.
//
// PRECISION (NON-NEGOTIABLE per the task contract):
//   * q.K dot         : bf16 inputs, accumulated in accfloat (f32).
//   * softmax math    : m, l, correction, p, the final divide — all f32 scalars.
//     (exp itself goes through aie::exp2 which on AIE2P only emits bf16 — same
//      hw constraint as the stock softmax.cc; immediately widened back to f32.)
//   * ctx accumulator : f32 (accfloat), V widened bf16->f32 before the mac.
//   The lossy long reductions (denom, ctx sum) stay f32 end to end.
//
// CALL CONTRACT (per (head, kv-tile), driven by the IRON core loop):
//   mha_tile(q_h, kv_tile, ctx_h, tile_idx, s_in_tile_baked)
//     q_h       : [HD] bf16            (this head's query; same for all its tiles)
//     kv_tile   : [TKV*HD | TKV*HD | hdr] bf16  (K-tile, V-tile, then a 2-bf16 (4-byte)
//                 RUNTIME-S header — see below)
//     ctx_h     : [HD] f32             (written only on the head's LAST non-empty tile)
//     tile_idx  : 0-based tile within the head; the tile with the smallest idx that has
//                 keys (always tile 0 here) resets the online state
//     s_in_tile_baked : IGNORED (kept for ABI stability). The real per-tile key count is
//                 read at RUNTIME from the kv-tile header so ONE xclbin (max S=448, 7 tiles)
//                 serves every cache length S<=448 — see "RUNTIME S" below.
//
// RUNTIME S (the fix for the growing self-KV cache; zero-pad would poison softmax — a zero
// K row scores 0, not -inf, so it steals weight). The IRON design unrolls a FIXED 7 tiles
// (n_tiles for S=448). The host writes, into the 4 bytes immediately after each tile's
// V-tile, an int32 `s_in_tile`:  >0 = normal tile with that many real keys;  <0 = LAST
// non-empty tile, finalize (ctx_h = acc/l), |val| real keys;  0 = empty tile (skip — for
// tiles beyond ceil(S/TKV)). Bit-exact int32 transport via two bf16 lanes (no rounding,
// unlike storing the count as a bf16 float). The kernel reads this and ignores the baked arg.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// e^x = 2^(x*log2e). No libm on aie2p; exp via native aie::exp2 over f32 vectors.
static constexpr float LOG2E = 1.4426950408889634f;

static constexpr int HD = 64;
static constexpr int VL = 16;
static constexpr int HD_VECS = HD / VL; // 4

// TKV = keys per streamed K/V tile (compile-time; the IRON design must match).
#ifndef MHA_TKV
#define MHA_TKV 64
#endif
static constexpr int TKV = MHA_TKV;

// Per-tile bf16 layout: K-tile (TKV*HD) | V-tile (TKV*HD) | runtime-S header (2 bf16 = 1 int32).
static constexpr int KV_TILE_BF16 = 2 * TKV * HD + 2;

// --- online-softmax state, core-local, persists across tile calls for one head ---
static float g_m;                  // running max
static float g_l;                  // running denominator
alignas(aie::vector_decl_align) static float g_acc[HD]; // running f32 ctx accumulator

// scalar exp via the only hw exp (vector exp2->bf16); 1-lane use is fine here.
static inline float exp_scalar(float x) {
  aie::vector<float, VL> v = aie::broadcast<float, VL>(x * LOG2E);
  aie::vector<bfloat16, VL> e = aie::exp2<bfloat16>(v);
  return (float)e.get(0);
}

template <int Tkv>
static void mha_tile_impl(const bfloat16 *restrict q_h,
                          const bfloat16 *restrict kv, float *restrict ctx_h,
                          int32_t tile_idx, int32_t /*s_in_tile_baked, ignored*/) {
  event0();
  const float scale = 1.0f / 8.0f; // 1/sqrt(64)

  // RUNTIME S: read the per-tile int32 count from the 4 bytes after the V-tile (bit-exact,
  // host-written). >0 normal; <0 last non-empty (finalize); 0 empty (skip).
  const int32_t s_in_tile = *reinterpret_cast<const int32_t *>(kv + 2 * Tkv * HD);
  const bool last = s_in_tile < 0;
  const int s_real = last ? -s_in_tile : s_in_tile;

  const bfloat16 *Kt = kv;             // [Tkv, HD]
  const bfloat16 *Vt = kv + Tkv * HD;  // [Tkv, HD]

  // reset online state on the head's first tile (tile 0 always has >=1 key since S>=1).
  if (tile_idx == 0) {
    g_m = -3.0e38f;
    g_l = 0.0f;
    for (int i = 0; i < HD; i++)
      g_acc[i] = 0.0f;
  }

  // empty tile (beyond ceil(S/TKV)): nothing to accumulate, never finalizes.
  if (s_real == 0) {
    event1();
    return;
  }

  // load q_h once for this tile (4 native bf16 vectors).
  aie::vector<bfloat16, VL> q0 = aie::load_v<VL>(q_h + 0 * VL);
  aie::vector<bfloat16, VL> q1 = aie::load_v<VL>(q_h + 1 * VL);
  aie::vector<bfloat16, VL> q2 = aie::load_v<VL>(q_h + 2 * VL);
  aie::vector<bfloat16, VL> q3 = aie::load_v<VL>(q_h + 3 * VL);

  for (int s = 0; s < s_real; s++) {
    const bfloat16 *krow = Kt + s * HD;
    aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
    acc = aie::mac(acc, q0, aie::load_v<VL>(krow + 0 * VL));
    acc = aie::mac(acc, q1, aie::load_v<VL>(krow + 1 * VL));
    acc = aie::mac(acc, q2, aie::load_v<VL>(krow + 2 * VL));
    acc = aie::mac(acc, q3, aie::load_v<VL>(krow + 3 * VL));
    float score = aie::reduce_add(acc.to_vector<float>()) * scale;

    // online softmax update (all f32 scalars).
    float m_old = g_m;
    float m_new = score > m_old ? score : m_old;
    float corr = exp_scalar(m_old - m_new); // 1.0 on the very first key
    float p = exp_scalar(score - m_new);
    g_l = g_l * corr + p;
    g_m = m_new;

    // acc = acc*corr + p * V[s]  (f32 accumulator, V widened bf16->f32).
    const bfloat16 *vrow = Vt + s * HD;
    aie::vector<float, VL> corr_v = aie::broadcast<float, VL>(corr);
    aie::vector<float, VL> p_v = aie::broadcast<float, VL>(p);
    for (int vi = 0; vi < HD_VECS; vi++) {
      aie::accum<accfloat, VL> va;
      va.from_vector(aie::load_v<VL>(vrow + vi * VL)); // bf16 -> f32
      aie::vector<float, VL> acc_old = aie::load_v<VL>(g_acc + vi * VL);
      // acc_old*corr + p*v
      aie::accum<accfloat, VL> t = aie::mul(acc_old, corr_v);
      t = aie::mac(t, p_v, va.to_vector<float>());
      aie::store_v(g_acc + vi * VL, t.to_vector<float>());
    }
  }

  // finalize on the head's last tile: ctx_h = acc / l.
  if (last) {
    float inv = 1.0f / g_l;
    aie::vector<float, VL> inv_v = aie::broadcast<float, VL>(inv);
    for (int vi = 0; vi < HD_VECS; vi++) {
      aie::vector<float, VL> a = aie::load_v<VL>(g_acc + vi * VL);
      aie::store_v(ctx_h + vi * VL, aie::mul(a, inv_v).to_vector<float>());
    }
  }
  event1();
}

extern "C" {
void mha_tile(bfloat16 *q_h, bfloat16 *kv, float *ctx_h, int32_t tile_idx,
              int32_t s_in_tile) {
  mha_tile_impl<TKV>(q_h, kv, ctx_h, tile_idx, s_in_tile);
}
}
