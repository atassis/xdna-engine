//===- rope_interleaved.cc ---------------------------------------*- C++ -*-===//
//
// Generic RoPE (rotary position embedding) brick, ADJACENT-PAIR convention, applied as a Q/K
// prologue over a resident [tile,D] stream.
//
// THIS CORRECTS route_b_kernels/bricks/rope-lut, which implements split-half (GPT-NeoX / Llama)
// rotation: pairs (i, i+D/2). The S2 model's `ggml_rope_ext(..., mode=0)` ==
// GGML_ROPE_TYPE_NORMAL (ggml.h:250) is the OTHER convention: ADJACENT pairs (2i, 2i+1) each get
// one (cos,sin). Call sites: s2.cpp/src/s2_codec.cpp:338, s2.cpp/src/s2_model.cpp:984. Both
// conventions are self-consistent rotations -- every gate that only checks itself against itself
// passes either way, and the golden here is cross-checked against the real oracle
// (scripts/s2_ar_ref.py::rope_interleaved AND scripts/codec_quantizer_ref.py::_rope_normal, two
// independently-written formulations) precisely because of that hazard -- see golden.py.
//
//   out[2i]   = x[2i]*cos(theta_i) - x[2i+1]*sin(theta_i)
//   out[2i+1] = x[2i]*sin(theta_i) + x[2i+1]*cos(theta_i)     for i in [0, ROPE_ROT/2)
//
// This is a generic op-TYPE (position x rotary), not a per-model kernel:
//   - ROPE_D    : total feature dim per row (head_dim, or a flattened multi-head row)
//   - ROPE_ROT  : rotary width <= ROPE_D (PARTIAL rotary; dims [ROPE_ROT, ROPE_D) pass through
//                 untouched, in place)
//   - ROPE_M    : resident tile rows (tokens, or token*head pairs) this call processes
//
// SIN/COS ROUTE: precomputed on the HOST, streamed in as a resident [ROPE_M, ROPE_ROT] float32
// operand (route (a) from the task brief), NOT computed on device. Two reasons this beats
// rope-lut's on-device gather-LUT approach for THIS brick:
//   1. rope-lut's aie::lut<4>/parallel_lookup ab/cd bank-duplication layout is an OPEN,
//      unresolved question in this repo (see sin.cc's header and
//      log/2026-07/2026-07-25-rope-lut-root-cause-bank-granularity.md) -- it already blocks
//      rope-lut itself. This brick routes around that hazard entirely rather than inheriting it.
//   2. theta = position * inv_freq is position-dependent, so a real dataflow has to deliver a
//      per-row value from the host (or a prologue that computes it) regardless; folding the
//      sin/cos THEMSELVES host-side (float64, exact) costs nothing extra over folding inv_freq
//      alone and removes the LUT's 256-bin quantization error (rope-lut's dominant error term)
//      entirely -- the only error left is the bf16 round-trip on the Q/K values, ~4e-3 ulp.
// The cossin buffer is one INPUT of the two a core tile's DMA channels allow (qk is the other),
// so cos and sin for a row are packed contiguously as [cos(0..half-1) | sin(0..half-1)] --
// same packing discipline as rope-lut's pos+inv_freq pack (see verify_rope_interleaved.py).
//
// DEINTERLEAVE / REINTERLEAVE: aie::filter_even(v,1) / aie::filter_odd(v,1) split a raw
// 2*kVec-wide interleaved load into the pair-first (x[2i]) and pair-second (x[2i+1]) operands
// (this is the exact primitive rope-lut already uses to undo its LUT gather's own
// de-interleaving, so it is proven on this toolchain). The inverse -- rebuilding the interleaved
// row from the two rotated halves -- is aie::concat(aie::interleave_zip(a,b,1)), the textbook
// pairing documented at aie_api/aie.hpp's own interleave_zip example.
//
// bf16<->f32 PROMOTION: via the accum<accfloat> round-trip (accum.from_vector /
// accum.to_vector<T>), the same pattern rope_lut.cc and norm_gemv_prologue.cc use -- NOT a direct
// cast_to<float>, which is untested on this path.
//
// NO LOCAL SCRATCH BUFFER: unlike rope-lut (which builds a whole row of sin/cos before applying
// them in a second pass, needing alignas(64) sin_buf/cos_buf), this kernel is a single fused
// load/compute/store pass per kVec-chunk with no intermediate array -- so the alignas(64) local
// buffer hazard (a 16xf32 store is 64B; unaligned silently truncates/shifts) does not arise here.
//
// REGISTER PRESSURE: kept to ONE live vector per role (xv, x0/x1, cos/sin, out0/out1, mixed),
// reassigned rather than accumulated, mirroring sin.cc's and rope_lut.cc's measured-safe shape.
// The whole body is macro-expanded directly into the bound extern "C" symbol (no static/static
// inline helper) -- an opaque helper function returned NaN on this toolchain even when byte-
// identical to the inlined form (see sin.cc's header); rope_lut.cc's own extern "C" function is
// the proof this shape compiles and runs correctly, so this brick follows it exactly.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef ROPE_D
#define ROPE_D 128            // head_dim (feature width of one Q/K row)
#endif
#ifndef ROPE_ROT
#define ROPE_ROT ROPE_D       // rotary width; ROPE_ROT <= ROPE_D, must be even
#endif
#ifndef ROPE_M
#define ROPE_M 64             // resident tile rows this call processes
#endif

static_assert(ROPE_ROT <= ROPE_D, "ROPE_ROT must be <= ROPE_D (partial rotary)");
static_assert(ROPE_ROT % 2 == 0, "ROPE_ROT must be even (paired adjacent rotation)");

namespace {

constexpr unsigned kHalf = ROPE_ROT / 2;   // number of (cos,sin) pairs per row
constexpr unsigned kVec = 16;              // f32 elementwise apply width (native aie2p f32 lane count)
// Each iteration raw-loads 2*kVec interleaved bf16 elements (kVec pairs) -- that load must tile
// kHalf exactly, otherwise the final group would over-read past ROPE_ROT. ROPE_ROT a multiple of
// 32 satisfies this (64/128/256/... all do); fail at compile time instead of over-reading.
static_assert(kHalf % kVec == 0, "ROPE_ROT/2 must be a whole number of kVec groups (ROPE_ROT % 32 == 0)");

}  // namespace

extern "C" {

// qk     : [ROPE_M, ROPE_D] bf16, resident, rotated IN PLACE. ADJACENT-PAIR convention: pairs
//          (row[2i], row[2i+1]) for i in [0, ROPE_ROT/2) rotate; row[ROPE_ROT, ROPE_D) is
//          untouched (partial-rotary pass-through).
// cossin : [ROPE_M, ROPE_ROT] float32, resident. Per row m: [cos(theta_m,0..half-1) |
//          sin(theta_m,0..half-1)], HOST-PRECOMPUTED from that row's position and inv_freq (see
//          golden.py's build_cossin_resident) -- the device performs no trig call at all.
void rope_interleaved_prologue(bfloat16 *restrict qk, const float *restrict cossin) {
  event0();

  for (unsigned m = 0; m < ROPE_M; ++m) {
    bfloat16 *row = qk + (size_t)m * ROPE_D;
    const float *cos_row = cossin + (size_t)m * ROPE_ROT;
    const float *sin_row = cos_row + kHalf;

    for (unsigned i = 0; i < kHalf; i += kVec) {
      // raw load: 2*kVec interleaved elements covering kVec pairs, i.e. row[2i .. 2i+2*kVec)
      ::aie::vector<bfloat16, 2 * kVec> xv = ::aie::load_v<2 * kVec>(row + 2 * i);
      ::aie::vector<bfloat16, kVec> x0_bf = ::aie::filter_even(xv, 1);  // x[2i]   (pair-first)
      ::aie::vector<bfloat16, kVec> x1_bf = ::aie::filter_odd(xv, 1);   // x[2i+1] (pair-second)

      ::aie::accum<accfloat, kVec> x0a; x0a.from_vector(x0_bf, 0);
      ::aie::accum<accfloat, kVec> x1a; x1a.from_vector(x1_bf, 0);
      ::aie::vector<float, kVec> x0 = x0a.to_vector<float>();
      ::aie::vector<float, kVec> x1 = x1a.to_vector<float>();

      ::aie::vector<float, kVec> cos = ::aie::load_v<kVec>(cos_row + i);
      ::aie::vector<float, kVec> sin = ::aie::load_v<kVec>(sin_row + i);

      ::aie::vector<float, kVec> out0 = ::aie::sub(::aie::mul(x0, cos).template to_vector<float>(),
                                                    ::aie::mul(x1, sin).template to_vector<float>());
      ::aie::vector<float, kVec> out1 = ::aie::add(::aie::mul(x0, sin).template to_vector<float>(),
                                                    ::aie::mul(x1, cos).template to_vector<float>());

      ::aie::accum<accfloat, kVec> out0a; out0a.from_vector(out0, 0);
      ::aie::accum<accfloat, kVec> out1a; out1a.from_vector(out1, 0);
      ::aie::vector<bfloat16, kVec> out0_bf = out0a.template to_vector<bfloat16>();
      ::aie::vector<bfloat16, kVec> out1_bf = out1a.template to_vector<bfloat16>();

      // re-interleave: (out0[0],out1[0],out0[1],out1[1],...) == the original pair order.
      ::aie::vector<bfloat16, 2 * kVec> mixed = ::aie::concat(::aie::interleave_zip(out0_bf, out1_bf, 1));
      ::aie::store_v(row + 2 * i, mixed);
    }
    // row[ROPE_ROT, ROPE_D) untouched (partial-rotary pass-through) -- already correct in place.
  }

  event1();
}

}  // extern "C"
