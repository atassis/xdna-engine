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
// elementwise helpers (runtime length n -> one symbol serves both shapes).
// ---------------------------------------------------------------------------

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
