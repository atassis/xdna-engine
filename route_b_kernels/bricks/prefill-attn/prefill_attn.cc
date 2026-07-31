//===- prefill_attn.cc -------------------------------------------*- C++ -*-===//
//
// GENERIC AIE2P brick: causal, GQA, NO relative-position-bias attention over a SHORT resident
// sequence (op-TYPE `attention{mask: causal, pos: none, heads, kv_heads}`). Written for the S2
// AR fast decoder's PREFILL step (scripts/s2_ar_ref.py's `fast_transformer_forward`, ~line 693),
// where `hp.fast_context_length <= 11` (s2_ar_ref.py:337) bounds the query count M small.
//
// NOTHING ELSE IN THE CATALOG MATCHES THIS SHAPE (all three axes differ from the two closest
// relatives, each checked by reading its source before writing this file):
//   * route_b_kernels/mha_decode/mha_decode.cc is M=1 STREAMED/FLASH decode: K/V arrive as
//     TKV-key tiles and softmax is ONLINE (running max/denom/accumulator) because a full head's
//     K/V does not fit L1 at decode-time cache lengths (up to S_MAX=448). Wrong shape here: at
//     M<=11 the ENTIRE per-head K/V (2*11*128*2 = 5632 B) fits L1 trivially, so paying flash's
//     online-softmax bookkeeping (running m/l, correction-factor rescale) buys nothing and only
//     adds a `static` cross-call state dependency this brick does not need (see "NO STATIC L1
//     STATE" below). Its bf16 dot-product loop and its GQA index convention are reused; its ONE-
//     CALL-PER-QUERY-STEP shape is reused too (see "ONE ROW PER CALL" below), but unlike
//     mha_decode this kernel's per-call ABI carries NO scalar argument at all (see "MASK IS DATA,
//     NOT A SCALAR" below) -- a stricter, not looser, version of mha_decode's shape.
//   * route_b_kernels/relpos_mha/relpos_mha.cc is the Parakeet encoder's attention: M=T
//     (compute-bound, large), BIDIRECTIONAL (no causal mask), relative-position bias (AC+BD_shifted
//     terms), and NO GQA (H=8 query heads, no separate kv-head count). Wrong on exactly the three
//     axes this brick needs: causal, GQA, short-M.
//
// ONE ROW PER CALL, REVISION 2 -- READ THIS BEFORE THE MASK-IS-DATA SECTION BELOW.
// Revision 1 processed all M rows in one call via a device-side loop; that measured NaN (see git
// history / the session log for the full bisect). Revision 1 tried to fix it by moving the loop
// to a runtime `row_idx` scalar plus a Python-UNROLLED per-row call site in the harness (352 call
// sites for the 32-head x 11-row main case) -- this compiled, but FAILED TO LOAD ON DEVICE:
//   _XAie_LoadProgMemSection(): Overflow of program memory / XAIE_INVALID_ELF
// Not a numerics bug -- it never ran. An AIE core's PROGRAM memory is a SEPARATE, small budget
// from the 64 KB DATA memory this file's L1 FOOTPRINT section sizes against; 352 emitted call
// sites (each one a full copy of the compiled body, since Python-level unrolling is source
// duplication, not a device loop) does not fit it. THE ACTUAL SAFE PATTERN, proven on every green
// brick in this catalog (snake, gather-rows, softmax, gelu-erf) as well as mha_decode.cc, is the
// WORKER's `for _ in range_(n_tiles)` loop inside `bricklib._build_streamed` -- that is a single
// call site in the emitted program, with volume supplied by a genuine (safe) device dispatch
// loop, as opposed to a `for` loop INSIDE this file's kernel body (unsafe, see Revision 1's own
// history: green at one iteration, red at two -- an entirely different code path than the
// worker's dispatch loop, compiled differently, and not implicated by that defect).
//
// MASK IS DATA, NOT A SCALAR -- the actual fix. `row_idx` existed for exactly one purpose: the
// `j > row_idx` causal-masking branch. That branch is now GONE, along with the scalar argument
// that fed it -- both are unnecessary. This kernel receives a per-row ADDITIVE MASK as a second
// half of its streamed operand instead, exactly the convention this codebase's own two causal-
// attention oracles already use: `scripts/codec_quantizer_ref.py::_causal_window_mask` (~line
// 350: "Additive mask... 0.0 where key k is allowed for query q, -1e9 otherwise") and
// `scripts/s2_ar_ref.py::causal_attention` (~line 590: `np.triu(-inf, k=1)` ADDED to the raw
// scores before softmax). This file was already doing the mathematically equivalent thing with a
// branch (`row_scores[j] = -1e9f` instead of `row_scores[j] = score + mask[j]`); switching to the
// additive form is not a new algorithm, it is the SAME masking already-cited oracles use, applied
// as data instead of control flow -- which is what removes the scalar and lets this brick use the
// generic bricklib.verify_streamed builder (2 buffer inputs, 1 buffer output, no scalar; see
// `_check_symbol_arity`) instead of a hand-rolled design.
//
// WHY NAIVE FULL-ROW SOFTMAX, NOT FLASH (justifying the "SxS score matrix is fine" call from the
// task brief). fast_context_length<=11 means the per-head K/V block is 2*11*128*2 = 5632 bytes --
// about 8.6% of a 64 KB core tile -- so nothing forces a streamed/tiled K/V or an online-softmax
// accumulator here; that complexity in mha_decode.cc exists ONLY because decode-time S can reach
// 448. This kernel never holds more than ONE query row's score vector at a time, so the "SxS" in
// the task brief is the algorithm class (full softmax, not flash), not a literal 2-D buffer this
// file allocates.
//
// ORACLE (scripts/s2_ar_ref.py, read in full before writing this file):
//   * `causal_attention(q, k, v, scale)` (~line 582): q (n_tokens,n_head,head_dim); k,v
//     (n_tokens,n_head,head_dim) ALREADY GQA-expanded to n_head. n_past=0 causal mask (query i
//     attends keys 0..i). scores = einsum("qhd,khd->hqk", q, k)*scale; additive triu(-inf,k=1)
//     mask; softmax over the last (key) axis; out = einsum("hqk,khd->qhd", probs, v).
//   * `repeat_kv(x, n_rep)` (~line 571) -- ITS DOCSTRING, READ CAREFULLY: "REPEAT-INTERLEAVE...
//     matching repeat_interleave_heads() (s2_model.cpp:57-68)... i.e. np.repeat(x, n_rep, axis=1),
//     NOT np.tile." Output (Q) head `oh` therefore reads KV head `oh // n_rep` -- each KV head's
//     block of n_rep CONSECUTIVE output heads maps to it. Getting this backwards (oh % n_head_kv,
//     the np.tile convention) passes every self-consistent gate and only fails against the real
//     oracle -- golden.py's `_selftest_gqa_mapping_has_teeth` proves the two conventions actually
//     diverge on this brick's data, so a backwards driver could not pass silently.
//   Shapes at the fast-decoder call site (s2_ar_ref.py:708-711): head_dim=128 (fast_head_dim,
//   default), n_head=32 (fast_head_count), n_head_kv=8 (fast_head_count_kv), n_rep=4,
//   scale=1/sqrt(head_dim).
//
// GQA -- INDEX, NEVER MATERIALIZE (same doctrine, same proof shape as mha_decode.cc's header).
// Like mha_decode.cc, this kernel is HEAD-AGNOSTIC BY CONSTRUCTION: one call processes exactly
// ONE query row of ONE query head against exactly ONE kv head's resident K/V, whatever head that
// data belongs to. GQA is therefore NOT kernel-internal state -- it is entirely which K/V bytes a
// given call is fed, i.e. a driver/host-buffer-layout concern (golden.py / verify_prefill_attn.py
// here, mha_decode_iron.py's own note there), never a kernel-body branch. This project counts
// bytes moved, not FLOPs, and materializing an [n_head, M, head_dim] expanded KV copy would be 4x
// (n_rep) the necessary DRAM/L1 bytes for no compute benefit -- exactly the avoidable data
// movement the doctrine targets. mha_decode.cc's header already carries the numeric proof that
// index-then-select equals expand-then-slice (3.866e-08 across all 32 heads, same n_rep=4
// config); golden.py here re-derives the identical equivalence independently for the PREFILL
// (all-M-rows) case via `_selftest_index_equals_expand`, since mha_decode's proof only covers the
// M=1 decode slice. Because GQA means different heads need DIFFERENT resident K/V, and a
// bricklib resident operand is fixed for one whole design/build, the verify script builds ONE
// `bricklib.verify_streamed` DESIGN PER HEAD (32 builds) rather than one design covering all
// heads -- 32 builds is a host-side compile-time cost, not a device program-memory cost (each
// build's emitted program is the SAME single-call-site shape either way).
//
// REUSE FROM mha_decode.cc (read in full, this session, before writing this file): the bf16 Q.K
// dot-product loop (accfloat accumulator, `for (vi < HdVecs) acc = aie::mac(...)`, then
// `aie::reduce_add(acc.to_vector<float>())`) is copied verbatim in spirit -- same op sequence,
// same VL=16, same "reload q from L1 every inner iteration rather than cache it across the loop"
// register-pressure discipline mha_decode.cc's header documents choosing at HD=128 (HdVecs=8 live
// q-vectors held across the whole loop was the risk it was written to avoid; reloading trades a
// cheap L1 read for that). The V-accumulate step reuses mha_decode.cc's EXACT proven call
// sequence (`aie::mul(acc_old, corr_v)` seeding an `accum<accfloat,VL>`, then `aie::mac(t, p_v,
// va.to_vector<float>())`) with `corr_v` pinned to the identity 1.0f -- this brick has no online
// rescaling (see WHY NAIVE above), so the "correction factor" mha_decode.cc's flash loop needs is
// always 1 here.
//
// SOFTMAX -- REUSED, NOT REWRITTEN, AND NOT THE CAUSE OF ANY DEFECT SEEN SO FAR (an A/B with
// softmax_core's call inlined, run during the NaN investigation, was STILL broken -- ruling out
// the call boundary before the real cause, the Revision-1 unroll, was found). `#include
// "../softmax/softmax.cc"` (sibling-brick pattern already established by snake.cc's `#include
// "../sin/sin.cc"`) and this file calls `route_b_bricks::softmax_core<VL>` DIRECTLY (the internal
// template, like snake.cc calls `sin_v` directly rather than going through sin.cc's extern "C"
// wrapper) -- device-green THIS SESSION at 9.490e-08. One reusable primitive, one op-TYPE on top;
// no second softmax implementation. softmax_core<N>'s own internal `chunks = cols/N` loop is safe
// here specifically because cols=SPAD=VL=16 makes it always exactly ONE iteration -- do not widen
// SPAD past VL, or split a row across more than one softmax_core call, without re-deriving this.
//
// PADDING TO SPAD=16. softmax_core<N> requires `cols` to be a multiple of N (it loads/stores in
// N-wide vector chunks); N=VL=16 is this codebase's one native chunk width (softmax.cc,
// mha_decode.cc both use it). fast_context_length<=11 < 16, so every call here pads the row to
// SPAD=16 columns (one chunk). Slots j>=Mq have no real K data (the resident KV block is sized
// exactly Mq rows, not SPAD), so the dot-product loop below scores them 0 -- but the additive
// mask (host-supplied, see MASK IS DATA above) ALWAYS carries -1e9 in those slots regardless of
// which row is being processed, so the 0 never leaks any weight through softmax. Real causal
// masking (row i cannot see key j>i, for j<Mq) is likewise carried entirely by the mask array;
// the kernel does not need to know, or branch on, which row it is processing at all.
//
// NO STATIC L1 STATE. Every scratch buffer below (`row_scores`, `row_probs`, `acc_row`) is a
// plain automatic array, freshly overwritten (row_scores, row_probs) or explicitly zeroed
// (acc_row) at the top of every CALL -- never a `static`. This brick has no cross-call state to
// carry (no online softmax, and now not even a row-position scalar): every call is fully self-
// contained given its two input buffers.
//
// alignas(64). `row_scores`, `row_probs` (16 f32 = exactly 64 B, one aligned vector store/load)
// and `acc_row` (Hd f32, Hd a multiple of VL=16 so always a whole number of 64 B lines) are all
// `alignas(64)` -- a 16xf32 store is 64 B and an unaligned one truncates silently (values shifted
// by 2 floats, not an error), per the task brief's own warning; this kernel has exactly the
// score-matrix-shaped scratch that bites.
//
// A CLEAN COMPILE IS NOT A DEVICE PASS, AND A DEVICE PASS THAT NEVER LOADS IS NOT A NUMERICS BUG.
// This file is authored device-free (no NPU access in this task) and self-checked only via
// `route_b_kernels/bricks/_verify/compile_check.sh` (Peano compile, no device) plus `golden.py`
// run standalone on host. Three distinct failure classes have hit this brick's earlier revisions
// despite a clean compile and a clean host emulation: a device-side kernel loop miscompile (NaN,
// Revision 1a), and a program-memory overflow from Python-unrolled call sites (never ran at all,
// Revision 1b) -- neither is visible from this file compiling, or from golden.py agreeing with
// the oracle. The device gate (`verify_prefill_attn.py`, to be run under the device lock by the
// session that owns NPU access) is the only thing that can actually confirm this kernel works.
//
// L1 FOOTPRINT (steady state; M=11, Hd=128, VL=16, SPAD=16 defaults; the streamed operand is now
// [q_row | mask_row] PACKED into one tile, KV is resident per head at depth=1):
//   qm_row streamed tile = (Hd+SPAD)*2   = (128+16)*2 = 288 B (bf16) x depth 2  =   576 B
//   kv resident          = 2*M*Hd*2  = 2*11*128*2 = 5632 B (bf16, K|V packed) x depth 1 =  5632 B
//   ctx_row streamed tile = Hd*4          = 128*4 = 512 B (f32)  x depth 2  =  1024 B
//   objectFifo total                                                          =  7232 B  (7.1 KB)
//   kernel-local scratch: row_scores 16*4=64 B + row_probs 16*4=64 B + acc_row 128*4=512 B
//                                                                          =    640 B
//   TOTAL                                                                    7872 B  (7.7 KB,
//   12.3% of the 64 KB core tile) -- fits with enormous headroom. This is DATA memory, the budget
//   this section has always sized against; the Revision 1b failure was PROGRAM memory, a separate
//   budget this section does not (and cannot) size -- the fix there is structural (one call site,
//   see ONE ROW PER CALL REVISION 2 above), not a smaller buffer.
//
// DMA CHANNELS: 2 inputs (qm_row packed, kv resident) + 1 output (ctx_row) -- exactly the 2-input-
// DMA-channel budget a core tile has, and exactly what `_check_symbol_arity` expects for a
// resident-operand streamed design (3 buffer params = 2 in + 1 out, NO scalar).
//
// PARAMETERIZATION: head_dim and M are compile-time -D macros (PREFILL_HD, PREFILL_M), the same
// convention mha_decode.cc uses for MHA_HD/MHA_TKV -- never a runtime branch for the SHAPE
// (dispatch overhead dominates at these small M). There is no per-call runtime parameter of any
// kind left in this kernel's ABI (see MASK IS DATA above). n_head and n_head_kv are DELIBERATELY
// NOT kernel-side macros: per "GQA -- INDEX, NEVER MATERIALIZE" above, the kernel never needs to
// know how many heads exist, only which one head's data it was handed -- exactly mha_decode.cc's
// own division of labor (grep its file: no NHEAD/NHEAD_KV macro exists there either).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../softmax/softmax.cc"

static constexpr int VL = 16; // native chunk width used throughout this codebase (softmax.cc,
                               // mha_decode.cc) for both bf16 and f32 vectors on aie2p.

#ifndef PREFILL_HD
#define PREFILL_HD 128 // fast_head_dim default (s2_ar_ref.py ARHParams.fast_head_dim)
#endif
static constexpr int HD = PREFILL_HD;

#ifndef PREFILL_M
#define PREFILL_M 11 // fast_context_length default (s2_ar_ref.py:337); must be <= SPAD (16).
#endif
static constexpr int M = PREFILL_M;

static constexpr int SPAD = 16; // one softmax_core<VL> chunk; see header PADDING note.

// Compile-time sqrt (Newton-Raphson, folded entirely at compile time) -- same helper mha_decode.cc
// carries (not shared via include, to keep this brick's only cross-file dependency the sanctioned
// softmax reuse; five lines, not worth a third file).
static constexpr float ct_sqrt_iter(float x, float curr, float prev) {
  return curr == prev ? curr : ct_sqrt_iter(x, 0.5f * (curr + x / curr), curr);
}
static constexpr float ct_sqrt(float x) {
  return ct_sqrt_iter(x, x > 0.0f ? x : 1.0f, 0.0f);
}

// ONE QUERY ROW PER CALL, NO SCALAR ARGUMENT -- see header "ONE ROW PER CALL, REVISION 2" and
// "MASK IS DATA, NOT A SCALAR". qm_row packs [q_row (Hd) | mask_row (SPAD)] -- the harness builds
// this once per (head, row) pair; kv is the FULL resident [Mq,Hd] K | [Mq,Hd] V block for this
// call's KV head, held across all Mq calls that share it via the WORKER's dispatch loop
// (bricklib's `_build_streamed`, a single call site -- never a loop inside this function).
template <int Hd, int Mq>
static void prefill_attn_row_impl(const bfloat16 *restrict qm_row, const bfloat16 *restrict kv,
                                  float *restrict ctx_row) {
  event0();
  static_assert(Hd % VL == 0, "Hd must be a multiple of VL=16");
  static_assert(Mq >= 1 && Mq <= SPAD, "M must fit one softmax_core<VL> chunk (<=16); "
                "a bigger M needs SPAD=ceil(M/VL)*VL and multiple softmax calls -- not built, "
                "see the report's 'deliberately not done' note");
  constexpr int HdVecs = Hd / VL;
  constexpr float SCALE = 1.0f / ct_sqrt(static_cast<float>(Hd));

  const bfloat16 *q_row = qm_row;         // [Hd]
  const bfloat16 *mask_row = qm_row + Hd; // [SPAD], additive: 0 = visible, -1e9 = masked

  const bfloat16 *K = kv;           // [Mq, Hd]
  const bfloat16 *V = kv + Mq * Hd; // [Mq, Hd]

  alignas(64) float row_scores[SPAD];
  alignas(64) float row_probs[SPAD];
  alignas(64) float acc_row[Hd];

  // Pass 1a: dot products for the Mq real keys; j>=Mq (no K data) scores 0 -- see header PADDING
  // (the mask, added next, unconditionally carries -1e9 there regardless of row).
  for (int j = 0; j < SPAD; j++) {
    if (j >= Mq) {
      row_scores[j] = 0.0f;
      continue;
    }
    const bfloat16 *kj = K + j * Hd;
    aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
    for (int vi = 0; vi < HdVecs; vi++)
      acc = aie::mac(acc, aie::load_v<VL>(q_row + vi * VL), aie::load_v<VL>(kj + vi * VL));
    row_scores[j] = aie::reduce_add(acc.to_vector<float>()) * SCALE;
  }

  // Pass 1b: additive mask -- DATA, not a branch (see header MASK IS DATA). SPAD==VL==16 -> one
  // vector op, no loop. mask_row widened bf16->f32 the same way V is widened in pass 3 below.
  {
    aie::accum<accfloat, VL> mask_acc;
    mask_acc.from_vector(aie::load_v<VL>(mask_row));
    aie::vector<float, VL> scores_v = aie::load_v<VL>(row_scores);
    aie::store_v(row_scores, aie::add(scores_v, mask_acc.to_vector<float>()));
  }

  // Pass 2: full-row softmax, reused verbatim from the softmax brick (route_b_bricks namespace,
  // internal template -- see header SOFTMAX section). Distinct buffers, no aliasing.
  route_b_bricks::softmax_core<VL>(row_scores, row_probs, SPAD);

  // Pass 3: ctx_row = sum_{j=0}^{Mq-1} probs[j] * V[j]. Fresh accumulator (see header NO STATIC
  // L1 STATE) -- zeroed here, never carried from a previous call.
  for (int vi = 0; vi < HdVecs; vi++)
    aie::store_v(acc_row + vi * VL, aie::zeros<float, VL>());

  for (int j = 0; j < Mq; j++) {
    const bfloat16 *vj = V + j * Hd;
    aie::vector<float, VL> p_v = aie::broadcast<float, VL>(row_probs[j]);
    // Identity correction factor (1.0): this is mha_decode.cc's own proven V-accumulate call
    // sequence with corr pinned to 1 -- see header REUSE FROM mha_decode.cc for why this exact
    // shape (not a plain from_vector seed) was chosen.
    aie::vector<float, VL> one_v = aie::broadcast<float, VL>(1.0f);
    for (int vi = 0; vi < HdVecs; vi++) {
      aie::accum<accfloat, VL> va;
      va.from_vector(aie::load_v<VL>(vj + vi * VL)); // bf16 -> f32 widen
      aie::vector<float, VL> acc_old = aie::load_v<VL>(acc_row + vi * VL);
      aie::accum<accfloat, VL> t = aie::mul(acc_old, one_v); // t = acc_old
      t = aie::mac(t, p_v, va.to_vector<float>());           // t += p * V[j]
      aie::store_v(acc_row + vi * VL, t.to_vector<float>());
    }
  }

  for (int vi = 0; vi < HdVecs; vi++)
    aie::store_v(ctx_row + vi * VL, aie::load_v<VL>(acc_row + vi * VL));
  event1();
}

extern "C" {

// One call = one query ROW of one (Q head, its GQA-mapped KV head) pair -- see header "ONE ROW
// PER CALL, REVISION 2" and "MASK IS DATA". qm_row: [HD+SPAD] bf16, [q_row (HD) | mask_row
// (SPAD)] packed (mask additive: 0=visible, -1e9=masked -- caller builds it per row, see
// golden.py's build_causal_mask). kv: [M,HD] K | [M,HD] V bf16 (this KV head's FULL resident
// keys/values -- caller selects h_kv = oh // n_rep, see header GQA section; NEVER an expanded-to-
// n_head buffer). ctx_row: [HD] f32 (this row's context). Exactly 3 buffer params (2 in + 1 out),
// NO scalar argument -- see _check_symbol_arity's arity=3 resident-operand contract.
void prefill_attn_row(bfloat16 *qm_row, bfloat16 *kv, float *ctx_row) {
  prefill_attn_row_impl<HD, M>(qm_row, kv, ctx_row);
}

} // extern "C"
