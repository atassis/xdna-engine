//===- mha_decode.cc ------------------------------------------*- C++ -*-===//
//
// M1 Task 0 -- on-chip SINGLE-QUERY (M=1) multi-head attention. Originally a Whisper-decoder-only
// kernel hardcoding head_dim=64 and no GQA; PARAMETERIZED here (2026-07-31) on head_dim (MHA_HD)
// so the SAME kernel body also serves the S2 fast-decoder's HD=128, 8->32 GQA config. This is a
// parameterization of an existing, working kernel, not a rewrite -- see "BIT-IDENTICAL" below for
// exactly what changed and why the shipped HD=64 path is unaffected.
//
// Whisper-small (shipped default, MHA_HD unset -> 64): D=768, n_heads=12, head_dim HD=64, TKV=64.
//   Validated against the host reference `attend_one` (rust/npu-engine/src/asr/whisper_decoder.rs)
//   via rust/npu-probes/src/bin/mha_decode_probe.rs, gate rel-L2 <= 0.08.
// S2 fast decoder (new, -DMHA_HD=128 -DMHA_TKV=32): head_dim HD=128, n_head=32, n_head_kv=8
//   (n_rep=4). Validated against `scripts/s2_ar_ref.py::causal_attention` + `repeat_kv` via
//   route_b_kernels/bricks/_verify/verify_mha_decode_hd128.py, gate rel-L2 <= 3e-2.
//
// STREAMING / FLASH design (forced by L1 capacity): a single compute tile has only ~64 KB data
// memory and 2 input DMA channels. The full per-head K+V does not fit, let alone all heads. So K/V
// are STREAMED in tiles of TKV keys and the softmax is ONLINE (flash-attention): per head we keep
// only the running max m, running denom l, and the f32 ctx accumulator acc[HD] across tiles --
// never the full score row.
//
// Per head h, over kv-tiles:
//   for each key s in the tile:
//     score = (q_h . K[s]) * (1/sqrt(HD))            (bf16 in, f32 accum)
//     m_new = max(m, score)
//     correction = exp(m - m_new)                    (rescale prior state)
//     p = exp(score - m_new)
//     l   = l*correction + p
//     acc = acc*correction + p * V[s]                (f32 accumulator)
//     m   = m_new
//   ctx_h = acc / l   (emitted after the head's last tile)
//
// This is mathematically identical to a host offline 3-pass softmax (verified by the Whisper
// probe and, for HD=128, by verify_mha_decode_hd128.py's numpy-tiled self-check against
// causal_attention) but needs only one TKV-key tile resident at a time.
//
// PRECISION (NON-NEGOTIABLE per the original task contract, unchanged):
//   * q.K dot         : bf16 inputs, accumulated in accfloat (f32).
//   * softmax math    : m, l, correction, p, the final divide -- all f32 scalars.
//     (exp itself goes through aie::exp2 which on AIE2P only emits bf16 -- same
//      hw constraint as the stock softmax.cc; immediately widened back to f32.)
//   * ctx accumulator : f32 (accfloat), V widened bf16->f32 before the mac.
//   The lossy long reductions (denom, ctx sum) stay f32 end to end.
//
// CALL CONTRACT (per (head, kv-tile), driven by the IRON core loop) -- UNCHANGED ABI:
//   mha_tile(q_h, kv_tile, ctx_h, tile_idx, s_in_tile_baked)
//     q_h       : [HD] bf16            (this head's query; same for all its tiles)
//     kv_tile   : [TKV*HD | TKV*HD | hdr] bf16  (K-tile, V-tile, then a 2-bf16 (4-byte)
//                 RUNTIME-S header -- see below)
//     ctx_h     : [HD] f32             (written only on the head's LAST non-empty tile)
//     tile_idx  : 0-based tile within the head; the tile with the smallest idx that has
//                 keys (always tile 0 here) resets the online state
//     s_in_tile_baked : IGNORED (kept for ABI stability). The real per-tile key count is
//                 read at RUNTIME from the kv-tile header so ONE xclbin serves every cache
//                 length S<=S_MAX -- see "RUNTIME S" below.
//
// RUNTIME S (the fix for the growing self-KV cache; zero-pad would poison softmax -- a zero
// K row scores 0, not -inf, so it steals weight). The IRON design unrolls a FIXED tile count
// (n_tiles = ceil(S_MAX/TKV)). The host writes, into the 4 bytes immediately after each tile's
// V-tile, an int32 `s_in_tile`:  >0 = normal tile with that many real keys;  <0 = LAST
// non-empty tile, finalize (ctx_h = acc/l), |val| real keys;  0 = empty tile (skip -- for
// tiles beyond ceil(S/TKV)). Bit-exact int32 transport via two bf16 lanes (no rounding,
// unlike storing the count as a bf16 float). The kernel reads this and ignores the baked arg.
//
// GQA (2026-07-31 addition, S2 fast decoder: 8 KV heads -> 32 Q heads, n_rep=4). This kernel
// is HEAD-AGNOSTIC by construction: mha_tile always processes exactly ONE query vector against
// ONE stream of K/V tiles, whatever head that data happens to belong to. GQA is therefore NOT
// kernel-internal state -- it is entirely a question of which K/V a given call is fed, i.e. a
// driver/host-buffer-layout concern, not a kernel-body concern. Per
// `scripts/s2_ar_ref.py::repeat_kv`'s docstring -- REPEAT-INTERLEAVE, i.e. `np.repeat(x, n_rep,
// axis=1)`, explicitly NOT `np.tile` -- output (Q) head `oh` reads KV head `oh // n_rep` (each KV
// head's block of n_rep CONSECUTIVE Q heads maps to it). The doctrine-required consequence: a
// correct integration NEVER materializes an expanded [n_head, ...] KV copy (this project counts
// bytes moved, not FLOPs, and that copy is 4x the necessary DRAM bytes at n_rep=4) -- it INDEXES,
// sizing the host KV buffer at n_head_kv and feeding the SAME KV-head region to the n_rep Q-head
// calls that share it. That host-buffer-layout change belongs in mha_decode_iron.py (or a new S2
// driver script) at integration time and is deliberately NOT made here (out of scope for this
// parameterization pass -- see the verify script's own header). What IS proven here, numerically,
// is that the index map is correct: verify_mha_decode_hd128.py's golden computes the reference
// attention output for one Q head BOTH via full repeat_kv-expand-then-slice AND via direct
// oh//n_rep indexing into the unexpanded K/V, and asserts they are identical -- confirming the
// kernel-level contract (feed it whichever head's data) composes correctly with an index-not-copy
// driver without needing any GQA-specific code path inside mha_tile_impl itself.
//
// PARAMETERIZATION (2026-07-31): HD generalized from a hardcoded 64 to MHA_HD (default 64,
// preserving the shipped Whisper build -- see BIT-IDENTICAL below). TKV was already a -D macro
// (MHA_TKV, default 64); both are compile-time constants, never a runtime branch -- this is an
// M=1 kernel where dispatch overhead dominates, so a shape check in the per-key inner loop would
// be a straight regression. mha_tile_impl is templated on <Hd, Tkv> (the `template<int N>` shape
// this codebase uses, e.g. route_b_kernels/bricks/sin/sin.cc) rather than `static inline`d: an
// identical body behind a plain `static inline` helper returned NaN on this target in an earlier
// kernel, and the fix that is proven safe here is exactly this shape -- the real work stays
// directly in the function bound (via one concrete instantiation) to the extern "C" symbol.
//
// L1 FOOTPRINT (why TKV must shrink when HD doubles -- a core tile is 64 KB, and at objectFifo
// depth=2 (double-buffered) a resident fifo's STEADY-STATE footprint is 2x one tile, i.e. only
// ~32 KB of the 64 KB budget is available per double-buffered stream before other consumers
// (code, stack, the small q/ctx fifos, g_acc) even enter the count:
//   shipped HD=64,  TKV=64 : per-tile KV = (2*64*64  + 2) bf16 = 8194  bf16 = 16.00 KB/tile
//                            depth=2 double buffer                        = 32.02 KB  (fits)
//   naive   HD=128, TKV=64 : per-tile KV = (2*64*128 + 2) bf16 = 16386 bf16 = 32.00 KB/tile
//                            depth=2 double buffer                        = 64.02 KB  (DOES NOT
//                            FIT -- exceeds the entire 64 KB core tile on the kv fifo alone,
//                            before q/ctx fifos, g_acc, code, or stack)
//   this build HD=128, TKV=32 : per-tile KV = (2*32*128 + 2) bf16 = 8194 bf16 = 16.00 KB/tile
//                            depth=2 double buffer                        = 32.02 KB  (fits --
//                            byte-for-byte the SAME footprint as the shipped kernel, since
//                            TKV*HD is held constant at 4096 elements/tile-plane)
// q/ctx fifos and g_acc double in size too (HD=64->128) but stay in the hundreds-of-bytes range
// (q: 128B->256B doubled at depth 2; ctx: 256B->512B; g_acc: 256B->512B) -- negligible against the
// ~32 KB kv double-buffer either way. So MHA_TKV=32 is the required pairing for MHA_HD=128, not a
// free choice; it is NOT re-derived per build, it is this exact halving.
//
// BIT-IDENTICAL FOR HD=64/TKV=64 (the shipped Whisper path) -- the structural changes below do
// not alter the shipped kernel's floating-point result:
//   (a) the 4 named q0..q3 / krow unrolled `aie::mac` calls became a `for (vi < HdVecs)` loop:
//       same 4 macs, same order, same operands -> bit-identical accumulation.
//   (b) q is now re-loaded from L1 every s-iteration instead of cached in registers across the
//       whole tile. q_h's bytes never change during a call, so a fresh load returns a
//       bit-identical vector every time; this also now matches the V-accumulation loop just below
//       it, which already reloaded g_acc from L1 every s-iteration in the original kernel. This
//       trades a few extra cheap L1 reads (not DMA) for bounded register pressure -- see the
//       comment at the call site for why that trade matters at HD=128.
//   (c) g_m/g_l/g_acc moved from file scope into `mha_tile_impl`-local `static`s, sized by the
//       template parameter Hd instead of the file-scope HD macro. Same storage duration (zero-
//       initialized once, persists across calls within a head's tile sweep), same
//       reset-on-tile_idx==0 semantics -- a scoping change only, made so that a build instantiating
//       more than one Hd (only the compile-time dual-instantiation self-check does this;
//       production always instantiates exactly one, matching MHA_HD) gets independent, correctly
//       sized storage per Hd instead of two Hds silently aliasing through one macro-sized array.
//   This keeps the "no static L1 state" caution in mind without removing the ONE static state this
//   algorithm structurally needs (the online-softmax accumulator persisting across a head's tile
//   sweep) -- that state already shipped and is unchanged in kind, only in scope.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// e^x = 2^(x*log2e). No libm on aie2p; exp via native aie::exp2 over f32 vectors.
static constexpr float LOG2E = 1.4426950408889634f;

static constexpr int VL = 16; // native bf16 vector width on aie2p; a hardware property, not a
                               // function of head_dim.

// HD = head_dim. Compile-time only, never a runtime branch (M=1 decode is dispatch-bound).
// Default 64 preserves the shipped Whisper build byte-for-byte; the S2 build passes -DMHA_HD=128.
#ifndef MHA_HD
#define MHA_HD 64
#endif
static constexpr int HD = MHA_HD;

// TKV = keys per streamed K/V tile (compile-time; the IRON design must match). See the L1
// footprint note above for why this must be halved (64->32) when MHA_HD goes 64->128.
#ifndef MHA_TKV
#define MHA_TKV 64
#endif
static constexpr int TKV = MHA_TKV;

// Per-tile bf16 layout: K-tile (TKV*HD) | V-tile (TKV*HD) | runtime-S header (2 bf16 = 1 int32).
static constexpr int KV_TILE_BF16 = 2 * TKV * HD + 2;

// Compile-time sqrt (Newton-Raphson; evaluated entirely by the compiler and folded to a float
// literal -- no libm call, no runtime cost, no dependency on a hardware scalar-sqrt op). Only
// needed because HD is now a parameter: the original HD=64 kernel just wrote the literal
// `1.0f/8.0f`. Gate is rel-L2 <= 3e-2, so exact IEEE-rounded correctness of this constant is not
// required -- a handful of Newton iterations at full float32 precision (~1e-7 relative) is far
// tighter than that already.
static constexpr float ct_sqrt_iter(float x, float curr, float prev) {
  return curr == prev ? curr : ct_sqrt_iter(x, 0.5f * (curr + x / curr), curr);
}
static constexpr float ct_sqrt(float x) {
  return ct_sqrt_iter(x, x > 0.0f ? x : 1.0f, 0.0f);
}

// scalar exp via the only hw exp (vector exp2->bf16); 1-lane use is fine here. Proven-safe
// static-inline-helper usage (unchanged from the shipped kernel): called only from within this
// same translation unit, unlike the cross-file static-inline pattern that returned NaN elsewhere.
static inline float exp_scalar(float x) {
  aie::vector<float, VL> v = aie::broadcast<float, VL>(x * LOG2E);
  aie::vector<bfloat16, VL> e = aie::exp2<bfloat16>(v);
  return (float)e.get(0);
}

template <int Hd, int Tkv>
static void mha_tile_impl(const bfloat16 *restrict q_h,
                          const bfloat16 *restrict kv, float *restrict ctx_h,
                          int32_t tile_idx, int32_t /*s_in_tile_baked, ignored*/) {
  event0();
  constexpr int HdVecs = Hd / VL;
  constexpr float SCALE = 1.0f / ct_sqrt(static_cast<float>(Hd));

  // Online-softmax state, core-local, persists across tile calls for one head. See
  // "BIT-IDENTICAL (c)" above for why these are function-local (Hd-sized) statics rather than the
  // original file-scope (HD-sized) ones.
  static float g_m;
  static float g_l;
  alignas(aie::vector_decl_align) static float g_acc[Hd];

  // RUNTIME S: read the per-tile int32 count from the 4 bytes after the V-tile (bit-exact,
  // host-written). >0 normal; <0 last non-empty (finalize); 0 empty (skip).
  const int32_t s_in_tile = *reinterpret_cast<const int32_t *>(kv + 2 * Tkv * Hd);
  const bool last = s_in_tile < 0;
  const int s_real = last ? -s_in_tile : s_in_tile;

  const bfloat16 *Kt = kv;             // [Tkv, Hd]
  const bfloat16 *Vt = kv + Tkv * Hd;  // [Tkv, Hd]

  // reset online state on the head's first tile (tile 0 always has >=1 key since S>=1).
  if (tile_idx == 0) {
    g_m = -3.0e38f;
    g_l = 0.0f;
    for (int i = 0; i < Hd; i++)
      g_acc[i] = 0.0f;
  }

  // empty tile (beyond ceil(S/TKV)): nothing to accumulate, never finalizes.
  if (s_real == 0) {
    event1();
    return;
  }

  for (int s = 0; s < s_real; s++) {
    const bfloat16 *krow = Kt + s * Hd;
    aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
    // q_h re-loaded from L1 every s (not cached in registers across the whole tile): q_h's bytes
    // never change during this call, so this is bit-identical to caching it once, and matches the
    // V-accumulation loop below, which already reloads g_acc from L1 every s. This bounds live
    // vectors to ~2 (one q, one k) + one accumulator regardless of HdVecs, instead of HdVecs live
    // q-vectors held for the whole tile -- HD=128 doubles HdVecs vs the shipped HD=64 kernel, and
    // register pressure (not L1 capacity) is the risk there, same lesson as sin.cc's header on
    // this target ("results ALIASED across variables" is the spill signature to watch for on
    // device; this loop shape is the mitigation, not a proof it cannot happen).
    for (int vi = 0; vi < HdVecs; vi++)
      acc = aie::mac(acc, aie::load_v<VL>(q_h + vi * VL), aie::load_v<VL>(krow + vi * VL));
    float score = aie::reduce_add(acc.to_vector<float>()) * SCALE;

    // online softmax update (all f32 scalars).
    float m_old = g_m;
    float m_new = score > m_old ? score : m_old;
    float corr = exp_scalar(m_old - m_new); // 1.0 on the very first key
    float p = exp_scalar(score - m_new);
    g_l = g_l * corr + p;
    g_m = m_new;

    // acc = acc*corr + p * V[s]  (f32 accumulator, V widened bf16->f32).
    const bfloat16 *vrow = Vt + s * Hd;
    aie::vector<float, VL> corr_v = aie::broadcast<float, VL>(corr);
    aie::vector<float, VL> p_v = aie::broadcast<float, VL>(p);
    for (int vi = 0; vi < HdVecs; vi++) {
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
    for (int vi = 0; vi < HdVecs; vi++) {
      aie::vector<float, VL> a = aie::load_v<VL>(g_acc + vi * VL);
      aie::store_v(ctx_h + vi * VL, aie::mul(a, inv_v).to_vector<float>());
    }
  }
  event1();
}

extern "C" {
void mha_tile(bfloat16 *q_h, bfloat16 *kv, float *ctx_h, int32_t tile_idx,
              int32_t s_in_tile) {
  mha_tile_impl<HD, TKV>(q_h, kv, ctx_h, tile_idx, s_in_tile);
}
}
