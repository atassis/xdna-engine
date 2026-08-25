//===- mm_silu_epilogue.cc ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Fused epilogue for the single-core matmul: given the f32 accumulator C-tile
// produced by matmul_bf16_f32 (mm.cc), apply SiLU (Swish) and write a bf16
// C-tile. This lets ONE xclbin compute
//   out = silu(A @ B + bias)
// with no host post-processing, and also performs the f32(acc) -> bf16(out)
// down-conversion on chip.
//
// BIAS: the per-N bias is folded into the matmul via K-augmentation on the host
// (an extra k-block of A = ones in col 0 / B = bias in row 0 yields
// ones @ bias = bias added to every output row), so this kernel does NOT take a
// bias argument and the core needs only 2 input DMA channels (A and B), which
// is the NPU2 compute-tile limit. The epilogue is therefore pure elementwise
// SiLU + downconvert, which is layout-independent: the matmul stores the C-tile
// in mmul-blocked order, but SiLU is per-element so the blocked order is
// irrelevant here; the output ObjectFifo's c_dims de-shuffle to row-major on
// the way out exactly as in the plain matmul.
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// Pure elementwise SiLU + f32->bf16 downconvert over an m*n C-tile.
// Bias is already folded into the matmul (K-augmentation), so this is just
//   out = silu(in),  in f32 -> out bf16
// SiLU is per-element, so the mmul-blocked storage order is irrelevant; we walk
// the tile flat in 16-wide chunks.
template <int size>
static inline void mm_silu_epilogue(const float *__restrict pC_in,
                                    bfloat16 *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");

  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);

  const float *__restrict in_ptr = pC_in;
  bfloat16 *__restrict out_ptr = pC_out;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    // Load f32 chunk, narrow to bf16 via an accumulator.
    aie::vector<float, 16> accf = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    aie::accum<accfloat, 16> a;
    a.from_vector(accf);
    aie::vector<bfloat16, 16> xv = a.to_vector<bfloat16>();

    // SiLU via the tanh identity (mirrors silu.cc):
    //   sigmoid(x) = 0.5*(1 + tanh(x/2)),  silu(x) = x*sigmoid(x)
    auto half_x = aie::mul(xv, half);
    auto tanh_half_x = aie::tanh<bfloat16>(half_x.to_vector<float>());
    auto tanh_p1 = aie::add(tanh_half_x, one);
    aie::vector<bfloat16, 16> sig = aie::mul(tanh_p1, half);
    aie::vector<bfloat16, 16> outv = aie::mul(xv, sig);
    aie::store_v(out_ptr, outv);
    out_ptr += 16;
  }

  event1();
}

// Pure elementwise f32 -> bf16 downconvert over an m*n C-tile, NO activation.
// Used by the no-activation ("bias mode") variant, e.g. FFN linear2 which wants
//   out = A@B + bias        (bias still folded via K-augmentation on the host)
// down-converted to bf16 on chip. Layout-independent for the same reason as the
// SiLU variant: it is per-element, so the mmul-blocked storage order is moot.
template <int size>
static inline void mm_narrow_epilogue(const float *__restrict pC_in,
                                      bfloat16 *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");

  const float *__restrict in_ptr = pC_in;
  bfloat16 *__restrict out_ptr = pC_out;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::vector<float, 16> accf = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    aie::accum<accfloat, 16> a;
    a.from_vector(accf);
    aie::vector<bfloat16, 16> outv = a.to_vector<bfloat16>();
    aie::store_v(out_ptr, outv);
    out_ptr += 16;
  }

  event1();
}

// --- f32-OUT variants (Step A resident modal epilogue) ------------------------
// The bf16-out epilogue forces the host to re-expand bf16->f32 for its downstream
// math (mha/glu/accumulate), which MEASURED as a net loss (s10 narrow backfire,
// +100ms). Keeping the output f32 means the host consumer needs NOTHING back.
// SiLU is still computed in bf16 (the proven, accurate-enough path; WER-gated),
// then up-converted to f32 for the store. Bias is folded into the matmul via
// K-augmentation (host), so these are pure elementwise.

// silu(x) computed in bf16, stored f32.
template <int size>
static inline void mm_silu_epilogue_f32o(const float *__restrict pC_in,
                                         float *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const float *__restrict in_ptr = pC_in;
  float *__restrict out_ptr = pC_out;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::vector<float, 16> accf = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    aie::accum<accfloat, 16> a;
    a.from_vector(accf);
    aie::vector<bfloat16, 16> xv = a.to_vector<bfloat16>();
    auto half_x = aie::mul(xv, half);
    auto tanh_half_x = aie::tanh<bfloat16>(half_x.to_vector<float>());
    auto tanh_p1 = aie::add(tanh_half_x, one);
    aie::vector<bfloat16, 16> sig = aie::mul(tanh_p1, half);
    aie::vector<bfloat16, 16> outv = aie::mul(xv, sig);
    // up-convert bf16 -> f32 via an accumulator (mirrors the f32->bf16 narrow path in reverse).
    aie::accum<accfloat, 16> oacc;
    oacc.from_vector(outv);
    aie::store_v(out_ptr, oacc.to_vector<float>());
    out_ptr += 16;
  }
  event1();
}

// silu(x), higher-precision hybrid: sigmoid via the bf16 tanh (bounded in [0,1],
// so bf16 is accurate enough), but keep x and the FINAL x*sigmoid multiply in f32.
// This removes the two bf16 roundings the plain f32o path incurs -- narrowing x
// before the multiply, and rounding the silu output before the up-convert -- which
// together cost ~+0.3 RU WER on the Parakeet FFN. The sigmoid stays bf16-tanh (a
// full-f32 tanh blows the cycle budget, see the GELU note below); only a single
// extra f32 mul is added, which is proven safe on this unit (the int8 dequant
// epilogue below ships an f32 aie::mul). f32 acc in -> f32 out.
template <int size>
static inline void mm_silu_epilogue_f32o_hiprec(const float *__restrict pC_in,
                                                float *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  const aie::vector<float, 16> halff = aie::broadcast<float, 16>(0.5f);
  const aie::vector<float, 16> onef = aie::broadcast<float, 16>(1.0f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> halfb = aie::broadcast<bfloat16, 16>(0.5f);
  const float *__restrict in_ptr = pC_in;
  float *__restrict out_ptr = pC_out;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::vector<float, 16> accf = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    // Keep the tanh ARGUMENT in f32 (x/2 in f32 from the un-rounded accumulator);
    // only the tanh OUTPUT is bf16 (bounded in [-1,1], so bf16 is fine). This is the
    // precision-critical fix vs the all-bf16 f32o path, which rounded x before tanh.
    aie::vector<float, 16> half_x = aie::mul(accf, halff);
#if defined(SILU_F32_TANH)
    // BROKEN, measured 2026-08-26 -- do not enable without fixing aie::tanh first. aie::tanh<float>
    // compiles to an EMPTY function on aie2p (.text size 0), so tf is undef and the optimiser folds
    // the sigmoid away, leaving out = accf. It does not fail to compile and does not fault; it
    // silently turns SiLU into a passthrough. See the GELU note below for the measurement.
    aie::vector<float, 16> tf = aie::tanh<float>(half_x);
    aie::vector<float, 16> sigf = aie::mul(aie::add(tf, onef), halff);
#elif defined(SILU_F32_TAIL)
    // Keep tanh's bf16 output (one narrow, unavoidable without an f32 tanh) but do the +1 and the
    // *0.5 in f32. The bf16 add was the SECOND lossy rounding; *0.5 is exact either way.
    aie::vector<bfloat16, 16> tanh_half_x = aie::tanh<bfloat16>(half_x);
    aie::accum<accfloat, 16> tacc;
    tacc.from_vector(tanh_half_x);
    aie::vector<float, 16> tf = tacc.to_vector<float>();
    aie::vector<float, 16> sigf = aie::mul(aie::add(tf, onef), halff);
#else
    aie::vector<bfloat16, 16> tanh_half_x = aie::tanh<bfloat16>(half_x);
    aie::vector<bfloat16, 16> tanh_p1 = aie::add(tanh_half_x, one);
    aie::vector<bfloat16, 16> sig = aie::mul(tanh_p1, halfb); // bf16 sigmoid in [0,1]
    // up-convert sigmoid to f32; final multiply uses the UN-rounded f32 x.
    aie::accum<accfloat, 16> sacc;
    sacc.from_vector(sig);
    aie::vector<float, 16> sigf = sacc.to_vector<float>();
#endif
    aie::vector<float, 16> outv = aie::mul(accf, sigf);
    aie::store_v(out_ptr, outv);
    out_ptr += 16;
  }
  event1();
}

// GELU (tanh approx, matches torch gelu(approximate="tanh") + the decode gelu_tile_bf16):
//   gelu(x) = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
// f32 acc in -> bf16 gelu -> f32 out (mirrors the silu f32o path). Used by the modal GELU mode (rtp[0]==2)
// to fold the Whisper encoder FFN activation into the fc1 GEMM epilogue (drops the ~260 ms/utt host GELU).
template <int size>
static inline void mm_gelu_epilogue_f32o(const float *__restrict pC_in,
                                         float *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  // The body is bf16 for CYCLES, not because f32 fails. A previous note here recorded that a
  // full-f32 GELU "HANGS on-device (run_matmul8: kernel run did not complete)" and blamed the cycle
  // budget; both f32 arms were rebuilt as GELU_F32_X (f32 cube + tail, hw tanh) and GELU_F32_TANH
  // (also tanh in f32) and MEASURED 2026-08-26 at 512x800x3072, and neither hangs:
  //
  //   arm            rel-L2 vs host f32 GELU    dispatch (median of 30)
  //   bf16 (shipped) 6.5527e-03                 0.568 ms
  //   GELU_F32_X     5.8921e-03  (1.11x)        0.979 ms  (1.72x)
  //
  // So the f32 narrows are a real 1.11x for a real +72% -- a cost/benefit call, not a wall. The
  // 793x-1611x arm needs the narrows AND a software tanh, and its cost is still unmeasured.
  //
  // GELU_F32_TANH is NOT that arm and must not be read as one: aie::tanh<float> compiles to an
  // EMPTY function on aie2p (.text size 0 -- the only Tanh specialisation returns bfloat16, and
  // unlike its sibling exp2, aie::tanh does not constrain TR), so the undef result lets the
  // optimiser fold this whole loop into a copy. It measures 0.445 ms -- FASTER than the bf16 body,
  // which is the tell -- and returns pC_in bit-exactly. Same trap sits under SILU_F32_TANH above.
  //
  // Not testable as prescribed: raising the reservation to 8192 does not BUILD at this design
  // (basic-sequential L1 allocation fails; the objectFIFO buffers already reach 0x10000), so a
  // stack-overflow explanation cannot be tested that way -- and needs no test, since both arms
  // complete at the default 0xD00.
  //
  // Isolated against its own f32 input (rtp[0]=0 is a pure copy, so it yields the exact pC_in), this
  // loop scores 6.5e-03 against a host f32 GELU; aie::tanh is ~7.9x off bf16 representability and
  // owns most of that. Do NOT rank a change here by Whisper's `encoded` rel: that gate sits behind
  // ln_post, which divides out a scale error, so it prefers the truncating form this swap replaces
  // even though every per-block rel says otherwise.
  //
  // crRnd is one sticky per-core register shared with the matmul, hence the restore on exit.
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  const float *__restrict in_ptr = pC_in;
  float *__restrict out_ptr = pC_out;
#if defined(GELU_F32_X) || defined(GELU_F32_TANH)
  // x is never narrowed: cube and tail both in f32. GELU_F32_TANH additionally takes tanh's f32
  // overload, so no bf16 survives anywhere; GELU_F32_X keeps the bf16 tanh output (the hybrid).
  const aie::vector<float, 16> halff = aie::broadcast<float, 16>(0.5f);
  const aie::vector<float, 16> onef = aie::broadcast<float, 16>(1.0f);
  const aie::vector<float, 16> c0f = aie::broadcast<float, 16>(0.7978845608f); // sqrt(2/pi)
  const aie::vector<float, 16> c1f = aie::broadcast<float, 16>(0.044715f);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::vector<float, 16> accf = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    aie::vector<float, 16> x2 = aie::mul(accf, accf);                 // x^2
    aie::vector<float, 16> x3 = aie::mul(x2, accf);                   // x^3
    aie::vector<float, 16> inner_f =
        aie::mul(aie::add(accf, aie::vector<float, 16>(aie::mul(c1f, x3))), c0f);
#if defined(GELU_F32_TANH)
    aie::vector<float, 16> tf = aie::tanh<float>(inner_f);
#else
    aie::accum<accfloat, 16> tacc;
    tacc.from_vector(aie::tanh<bfloat16>(inner_f));
    aie::vector<float, 16> tf = tacc.to_vector<float>();
#endif
    aie::vector<float, 16> xt = aie::mul(accf, aie::vector<float, 16>(aie::add(tf, onef)));
    aie::store_v(out_ptr, aie::vector<float, 16>(aie::mul(xt, halff)));
    out_ptr += 16;
  }
#else
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> c0 = aie::broadcast<bfloat16, 16>(0.7978845608f); // sqrt(2/pi)
  const aie::vector<bfloat16, 16> c1 = aie::broadcast<bfloat16, 16>(0.044715f);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::vector<float, 16> accf = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    aie::accum<accfloat, 16> a;
    a.from_vector(accf);
    aie::vector<bfloat16, 16> xv = a.to_vector<bfloat16>();
    aie::vector<bfloat16, 16> x2 = aie::mul(xv, xv);                  // x^2  (mul -> accum -> vector)
    aie::vector<bfloat16, 16> x3 = aie::mul(x2, xv);                  // x^3
    aie::vector<bfloat16, 16> c1x3 = aie::mul(c1, x3);               // c1*x^3
    aie::vector<bfloat16, 16> inner_b = aie::add(xv, c1x3);          // x + c1*x^3  (add -> vector)
    auto inner = aie::mul(c0, inner_b);                             // c0*(x + c1*x^3)  (accum)
    aie::vector<bfloat16, 16> t = aie::tanh<bfloat16>(inner.to_vector<float>()); // tanh(inner)
    aie::vector<bfloat16, 16> t_p1 = aie::add(t, one);              // 1 + tanh
    aie::vector<bfloat16, 16> xt = aie::mul(xv, t_p1);              // x*(1+tanh)
    aie::vector<bfloat16, 16> gx = aie::mul(half, xt);             // 0.5*x*(1+tanh)
    aie::accum<accfloat, 16> oacc;
    oacc.from_vector(gx);
    aie::store_v(out_ptr, oacc.to_vector<float>());
    out_ptr += 16;
  }
#endif
  aie::set_rounding(saved_rounding);
  event1();
}

// --- GLU epilogue (conv-module pw1) ---------------------------------------------
// out[t, c] = a[t, c] * sigmoid(g[t, c]), where pw1 produces [a | g] over 2*D columns.
//
// WHY THIS IS NOT LAYOUT-AWARE, even though GLU pairs two elements while every other
// epilogue here is per-element. mm.cc stores C as a grid of r x t sub-tiles, row-major,
// row-major within (`is_c_row_maj`, the default -- C_COL_MAJ is defined nowhere in this
// tree). Pair that with a W1 COLUMN PERMUTATION applied once at weight load, putting the
// 64 value columns in tile positions 0..63 and their gate partners in 64..127, and the
// partner of every element lands at a CONSTANT offset: half a row-block. Verified
// exhaustively for the shipped config -- 4096 of 4096 (row, position) pairs at +512, with
// the value half contiguous. So this walks flat runs like the others; the permutation, not
// the kernel, is what absorbs the blocked order.
//
// The result is written INTO the value half, which the C drain tap then takes 64-of-128 of
// (instruction-stream-only, so the array program and the xclbin are unchanged). In-place is
// safe: within a row-block we only ever write below the halfway point and only ever read the
// gate above it.
//
// Sigmoid is the hiprec ladder the silu epilogue already ships and that is already WER-gated
// (tanh ARGUMENT in f32, tanh output bf16, final multiply in f32). Folding GLU here also
// deletes the separate bf16 round-trip the standalone glu.cc had to do, because this holds
// the f32 accumulator and never narrows to hand GLU its input.
template <int rows, int cols, int r, int t>
static inline void mm_glu_epilogue_f32o(const float *__restrict pC_in,
                                        float *__restrict pC_out) {
  event0();
  static_assert(r == t, "the +half-row-block pairing assumes square mmul sub-tiles");
  static_assert(cols % (2 * t) == 0, "value/gate split must fall on a sub-tile boundary");
  static_assert(rows % r == 0, "tile rows must be a whole number of sub-tile rows");
  // One row-block = r rows x cols columns, laid out contiguously by the blocked store.
  constexpr int kRowBlock = r * cols;
  constexpr int kHalf = kRowBlock / 2;  // value half; gate partner sits exactly kHalf on
  static_assert(kHalf % 16 == 0, "epilogue walks 16-wide chunks");

  const aie::vector<float, 16> halff = aie::broadcast<float, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> halfb = aie::broadcast<bfloat16, 16>(0.5f);

  for (int blk = 0; blk < rows / r; blk++) {
    const float *__restrict a_ptr = pC_in + blk * kRowBlock;
    const float *__restrict g_ptr = a_ptr + kHalf;
    float *__restrict out_ptr = pC_out + blk * kRowBlock;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (int off = 0; off < kHalf; off += 16) {
      aie::vector<float, 16> av = aie::load_v<16>(a_ptr + off);
      aie::vector<float, 16> gv = aie::load_v<16>(g_ptr + off);
      // sigmoid(g): keep g/2 in f32 (un-rounded); only the tanh OUTPUT narrows to bf16.
      aie::vector<float, 16> half_g = aie::mul(gv, halff);
      aie::vector<bfloat16, 16> tanh_half_g = aie::tanh<bfloat16>(half_g);
      aie::vector<bfloat16, 16> tanh_p1 = aie::add(tanh_half_g, one);
      aie::vector<bfloat16, 16> sig = aie::mul(tanh_p1, halfb);
      aie::accum<accfloat, 16> sacc;
      sacc.from_vector(sig);
      aie::vector<float, 16> sigf = sacc.to_vector<float>();
      // final multiply against the UN-rounded f32 a. Bind to a vector first: aie::mul on two
      // f32 vectors yields an accum, and only the assignment converts it back.
      aie::vector<float, 16> outv = aie::mul(av, sigf);
      aie::store_v(out_ptr + off, outv);
    }
  }
  event1();
}

// identity: copy f32 acc -> f32 out (the matmul already folded bias via K-aug).
template <int size>
static inline void mm_identity_epilogue_f32o(const float *__restrict pC_in,
                                            float *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  const float *__restrict in_ptr = pC_in;
  float *__restrict out_ptr = pC_out;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::store_v(out_ptr, aie::load_v<16>(in_ptr));
    in_ptr += 16;
    out_ptr += 16;
  }
  event1();
}

// --- bf16-OUT modal variants (fc1 -> fc2 seam, deinterleave folded away) ------
// Same math as the f32-out modal epilogue above; only the STORE narrows to bf16.
//
// Why a bf16-out C at all, when the f32-out note above says bf16 out MEASURED as
// a net loss: that verdict was about a HOST consumer, which then had to re-expand
// bf16->f32 for its own math. Here the consumer is the fc2 GEMM, which wants bf16
// in. Paired with the chunk-major C drain (--c-chunk-width) the GEMM writes
// exactly the buffer fc2 reads, so the separate deinterleave+cast dispatch -- and
// the two hardware-context transitions around it -- disappear, along with the f32
// intermediate's DDR write and read-back.
//
// NUMERICS: bit-identical to the f32-out epilogue followed by cast_f32_bf16_row,
// which is what this replaces. The activation math stays exactly as it was (the
// hiprec silu keeps x and the final multiply in f32); only the final store is
// narrowed. The rounding mode is LOAD-BEARING, not decoration: an accum narrow
// TRUNCATES by default, while cast_f32_bf16_row selects round-to-nearest-even to
// match the host pack. Truncating here would silently shift every fc1 output by
// up to one ulp against the f32 truth the parity gate compares to.
template <int size>
static inline void mm_silu_epilogue_bf16o_hiprec(const float *__restrict pC_in,
                                                 bfloat16 *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  // Hand the mode back. crRnd is one sticky per-core register and this epilogue shares its core
  // with the matmul, so a mode left set here governs the bf16 -> bfp16 conversions of the NEXT
  // tile's mmul -- an accuracy coupling between two kernels that share nothing else.
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  const aie::vector<float, 16> halff = aie::broadcast<float, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> halfb = aie::broadcast<bfloat16, 16>(0.5f);
  const float *__restrict in_ptr = pC_in;
  bfloat16 *__restrict out_ptr = pC_out;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::vector<float, 16> accf = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    aie::vector<float, 16> half_x = aie::mul(accf, halff);
    aie::vector<bfloat16, 16> tanh_half_x = aie::tanh<bfloat16>(half_x);
    aie::vector<bfloat16, 16> tanh_p1 = aie::add(tanh_half_x, one);
    aie::vector<bfloat16, 16> sig = aie::mul(tanh_p1, halfb);
    aie::accum<accfloat, 16> sacc;
    sacc.from_vector(sig);
    aie::vector<float, 16> sigf = sacc.to_vector<float>();
    aie::vector<float, 16> outv = aie::mul(accf, sigf);
    // the one difference from the f32-out sibling: narrow the f32 result to bf16.
    aie::accum<accfloat, 16> oacc;
    oacc.from_vector(outv);
    aie::store_v(out_ptr, oacc.to_vector<bfloat16>());
    out_ptr += 16;
  }
  aie::set_rounding(saved_rounding);
  event1();
}

// gelu, bf16 out. Mirrors mm_gelu_epilogue_f32o exactly (same bf16 tanh approx),
// narrowing at the store instead of up-converting back to f32.
template <int size>
static inline void mm_gelu_epilogue_bf16o(const float *__restrict pC_in,
                                          bfloat16 *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> c0 = aie::broadcast<bfloat16, 16>(0.7978845608f);
  const aie::vector<bfloat16, 16> c1 = aie::broadcast<bfloat16, 16>(0.044715f);
  const float *__restrict in_ptr = pC_in;
  bfloat16 *__restrict out_ptr = pC_out;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::vector<float, 16> accf = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    aie::accum<accfloat, 16> a;
    a.from_vector(accf);
    aie::vector<bfloat16, 16> xv = a.to_vector<bfloat16>();
    aie::vector<bfloat16, 16> x2 = aie::mul(xv, xv);
    aie::vector<bfloat16, 16> x3 = aie::mul(x2, xv);
    aie::vector<bfloat16, 16> c1x3 = aie::mul(c1, x3);
    aie::vector<bfloat16, 16> inner_b = aie::add(xv, c1x3);
    auto inner = aie::mul(c0, inner_b);
    aie::vector<bfloat16, 16> t = aie::tanh<bfloat16>(inner.to_vector<float>());
    aie::vector<bfloat16, 16> t_p1 = aie::add(t, one);
    aie::vector<bfloat16, 16> xt = aie::mul(xv, t_p1);
    aie::vector<bfloat16, 16> gx = aie::mul(half, xt);
    aie::store_v(out_ptr, gx);
    out_ptr += 16;
  }
  aie::set_rounding(saved_rounding);
  event1();
}

// identity, bf16 out: the plain f32 acc -> bf16 narrow (bias already folded via
// K-augmentation). Same as cast_f32_bf16_row, done in the epilogue.
template <int size>
static inline void mm_identity_epilogue_bf16o(const float *__restrict pC_in,
                                              bfloat16 *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  const float *__restrict in_ptr = pC_in;
  bfloat16 *__restrict out_ptr = pC_out;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::vector<float, 16> accf = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    aie::accum<accfloat, 16> a;
    a.from_vector(accf);
    aie::store_v(out_ptr, a.to_vector<bfloat16>());
    out_ptr += 16;
  }
  aie::set_rounding(saved_rounding);
  event1();
}

// glu, bf16 out. Same value/gate pairing and hiprec sigmoid ladder as
// mm_glu_epilogue_f32o; the two differences are the narrowing store and that the
// result lands in a SEPARATE bf16 tile rather than in place, so nothing here
// depends on the +kHalf partner still being readable after a write.
//
// The result keeps the value half's ELEMENT positions, leaving the gate half
// undefined, so the drain tap that takes 64 of every 128 columns carries over
// unchanged from the f32-out mode.
template <int rows, int cols, int r, int t>
static inline void mm_glu_epilogue_bf16o(const float *__restrict pC_in,
                                         bfloat16 *__restrict pC_out) {
  event0();
  static_assert(r == t, "the +half-row-block pairing assumes square mmul sub-tiles");
  static_assert(cols % (2 * t) == 0, "value/gate split must fall on a sub-tile boundary");
  static_assert(rows % r == 0, "tile rows must be a whole number of sub-tile rows");
  constexpr int kRowBlock = r * cols;
  constexpr int kHalf = kRowBlock / 2;
  static_assert(kHalf % 16 == 0, "epilogue walks 16-wide chunks");

  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  const aie::vector<float, 16> halff = aie::broadcast<float, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> halfb = aie::broadcast<bfloat16, 16>(0.5f);

  for (int blk = 0; blk < rows / r; blk++) {
    const float *__restrict a_ptr = pC_in + blk * kRowBlock;
    const float *__restrict g_ptr = a_ptr + kHalf;
    bfloat16 *__restrict out_ptr = pC_out + blk * kRowBlock;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(2)
    for (int off = 0; off < kHalf; off += 16) {
      aie::vector<float, 16> av = aie::load_v<16>(a_ptr + off);
      aie::vector<float, 16> gv = aie::load_v<16>(g_ptr + off);
      aie::vector<float, 16> half_g = aie::mul(gv, halff);
      aie::vector<bfloat16, 16> tanh_half_g = aie::tanh<bfloat16>(half_g);
      aie::vector<bfloat16, 16> tanh_p1 = aie::add(tanh_half_g, one);
      aie::vector<bfloat16, 16> sig = aie::mul(tanh_p1, halfb);
      aie::accum<accfloat, 16> sacc;
      sacc.from_vector(sig);
      aie::vector<float, 16> sigf = sacc.to_vector<float>();
      aie::vector<float, 16> outv = aie::mul(av, sigf);
      aie::accum<accfloat, 16> oacc;
      oacc.from_vector(outv);
      aie::store_v(out_ptr + off, oacc.to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved_rounding);
  event1();
}

// --- int8 DEQUANT epilogue (L3: on-chip i32 -> f32 dequant) -------------------
// The int8 matmul (matmul_i8_i32) reduces i8*i8 into an i32 accumulator tile,
// IN-PLACE in the C tile (4 bytes/elem, exactly like the f32-out modal). This
// epilogue reads that i32 tile, multiplies by a single per-dispatch scalar
//   S = scale_a (dynamic per-tensor activation scale) * w_scale (per-tensor weight scale)
// and writes f32 out — IN-PLACE (i32 and f32 are both 4B; we read each lane as
// i32 before overwriting it as f32, so aliasing pC_in==pC_out is safe). This is
// the whole L3 win: it moves the fat per-element dequant MULTIPLY off the host
// (where it materialised a fresh f32 Vec, ~50ms, the reason int8 lost to bf16)
// onto the array, so the host epilogue becomes the same near-no-op as the bf16
// modal. Per-column weight scale + bias + SiLU stay on the host for this first
// cut (bias/silu are cheap; per-column w_scale would need per-column on-chip
// delivery — a later upgrade via an expanded RTP). Layout-independent: dequant
// is per-element, so the mmul-blocked storage order is irrelevant, the C
// ObjectFifo de-shuffles to row-major on the way out exactly as elsewhere.
template <int size>
static inline void mm_dequant_epilogue_i32_f32(const int32_t *__restrict pC_in,
                                               float *__restrict pC_out,
                                               float scale) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  const int32_t *__restrict in_ptr = pC_in;
  float *__restrict out_ptr = pC_out;
  const aie::vector<float, 16> sv = aie::broadcast<float, 16>(scale);
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(2)
  for (int off = 0; off < size; off += 16) {
    aie::vector<int32_t, 16> iv = aie::load_v<16>(in_ptr);
    in_ptr += 16;
    // i32 -> f32 (full-range, no shift), then scale by S.
    aie::vector<float, 16> fv = aie::to_float(iv, 0);
    aie::vector<float, 16> outv = aie::mul(fv, sv);
    aie::store_v(out_ptr, outv);
    out_ptr += 16;
  }
  event1();
}

extern "C" {

// Tile dims provided at compile time (same DIM_M/DIM_N as the matmul).
#ifndef EPI_M
#define EPI_M 32
#endif
#ifndef EPI_N
#define EPI_N 32
#endif
// The A tile's inner dim, needed only by the resadd MODE below (the epilogues are all
// C-tile-shaped). Passed by the generator; a default that drifts from the matmul's k
// would size the mode's loop against the wrong buffer.
#ifndef EPI_K
#define EPI_K 32
#endif
// mmul sub-tile dims, needed ONLY by the GLU mode -- the other epilogues are per-element and
// so are layout-blind. These are the shape mm.cc was compiled with (microkernel_mac_dim_map:
// 8,8,8 for the bfp16 path we run, 4,8,8 native), and a mismatch would silently pair the wrong
// elements, so the generator passes them rather than letting this default drift.
#ifndef EPI_R
#define EPI_R 8
#endif
#ifndef EPI_T
#define EPI_T 8
#endif

void mm_silu_epilogue_f32_bf16(const float *__restrict c_in,
                               bfloat16 *__restrict c_out) {
  mm_silu_epilogue<EPI_M * EPI_N>(c_in, c_out);
}

// No-activation variant: just f32 acc -> bf16 (bias already folded into matmul).
void mm_narrow_epilogue_f32_bf16(const float *__restrict c_in,
                                 bfloat16 *__restrict c_out) {
  mm_narrow_epilogue<EPI_M * EPI_N>(c_in, c_out);
}

// MODAL f32-out epilogue for the Step-A resident xclbin: rtp[0] selects the mode
// per instruction-stream (1 = SiLU for the FFN mm1, 0 = identity for every other
// op whose bias is K-augmented into the matmul). One xclbin, mode chosen by which
// stream the host dispatches -> zero context switches (the V2 mechanism, extended
// from N-selection to N+mode-selection).
void mm_modal_epilogue_f32_f32(const float *__restrict c_in,
                               float *__restrict c_out,
                               const int32_t *__restrict rtp) {
#ifdef MODAL_ROUND_EVEN
  // Round-to-nearest-even instead of the hardware default (truncate). The rounding mode is a CORE
  // CONTROL REGISTER, not a per-op argument, so this does not only affect the epilogue below -- it
  // persists and governs the bfp16 conversions inside the NEXT tile's matmul. That is the point:
  // under `emulate_bfloat16_mmul_with_bfp16` every A/B tile is converted bf16 -> bfp16, and doing
  // that with truncation biases every product low.
  aie::set_rounding(aie::rounding_mode::conv_even);
#endif
  // rtp[0]: 0=identity, 1=silu, 2=gelu, 3=glu (modal modes; baked per instruction stream).
  if (rtp[0] == 3) {
    // Halves the live output width -- the drain tap takes 64 of every 128 columns.
    mm_glu_epilogue_f32o<EPI_M, EPI_N, EPI_R, EPI_T>(c_in, c_out);
  } else if (rtp[0] == 1) {
    // Higher-precision hybrid silu (f32 x + f32 final multiply, bf16 sigmoid) --
    // WER-neutral vs the host f32 silu, unlike the all-bf16 f32o path.
    mm_silu_epilogue_f32o_hiprec<EPI_M * EPI_N>(c_in, c_out);
  } else if (rtp[0] == 2) {
    mm_gelu_epilogue_f32o<EPI_M * EPI_N>(c_in, c_out);
  } else {
    mm_identity_epilogue_f32o<EPI_M * EPI_N>(c_in, c_out);
  }
}

// MODE 4 -- resadd as a MODE of the GEMM rather than a sibling xclbin. It borrows the
// GEMM's own A and B input channels and its C drain, so it adds no fifo, no DMA channel
// and no switchbox route: the movement layer is 100% spoken for by the GEMM
// (routed-modal-gemm-dest-port-use), and a second topology has nowhere to go. Only what
// flows through the existing topology may vary.
//
// Length is EPI_M*EPI_K, the A tile, because A ([m,k]) is the smaller of the two input
// buffers ([k,n] for B) and a mode may not read past the buffer it borrows. The host taps
// decide what actually lands in those 2048 elements.
static constexpr unsigned EPI_RESADD_ELEMS = EPI_M * EPI_K;

// ACCADD arm. The encoder's `accadd` brick -- the fc2 K-split accumulation, and the highest-ranked
// context-merge candidate at 320 ms/clip -- is out[t,c] = a[t,c] + b[t,c] where the RUNNING
// ACCUMULATOR a is f32 and only the partial b is bf16. f32 there is not a taste: order-preserving
// f32 accumulation is what makes the device K-split bit-identical to the host one (acc_add.cc).
// So the mode has to carry an operand whose dtype is not the fifo's. It rides the GEMM's bf16 A
// fifo as raw BYTES -- the same 4096 B, read as 1024 f32 instead of 2048 bf16 -- which halves the
// element count. b keeps the fifo's own dtype, so exactly one operand is reinterpreted.
static constexpr unsigned EPI_ACCADD_ELEMS = EPI_RESADD_ELEMS / 2;

void mm_mode_resadd_bf16_f32(const bfloat16 *__restrict a,
                             const bfloat16 *__restrict b,
                             float *__restrict out) {
  event0();
  static_assert(EPI_RESADD_ELEMS % 32 == 0, "resadd tile must be a multiple of 32");
  static_assert(EPI_ACCADD_ELEMS % 32 == 0, "accadd tile must be a multiple of 32");
  const bfloat16 *__restrict pa = a;
  const bfloat16 *__restrict pb = b;
  float *__restrict pout = out;
#ifdef MODE4_ACCADD
  const float *__restrict pa32 = reinterpret_cast<const float *>(pa);
#endif
#ifndef MODE4_BISECT_NOPIPE
  // The NOPIPE arm suppresses this to try to break the store/VALU bundle the mode-4 hang tracks.
  // It does not: Peano pipelines anyway and emits a BYTE-IDENTICAL core ELF, so the arm measures
  // nothing. Kept because that is the finding -- it also voids an earlier note recording "the loop
  // and pragmas" as ruled out for this hang.
  AIE_PREPARE_FOR_PIPELINING
#endif
  AIE_LOOP_MIN_ITERATION_COUNT(2)
#ifdef MODE4_ACCADD
  for (unsigned off = 0; off < EPI_ACCADD_ELEMS; off += 32) {
#else
  for (unsigned off = 0; off < EPI_RESADD_ELEMS; off += 32) {
#endif
#ifdef MODE4_ACCADD
    // Same identity detour as bisect 8, for the same reason: an accumulator-sourced f32 add whose
    // result reaches memory without passing through the vector file hangs this core
    // (2026-08-21-the-opcode-is-innocent-the-store-path-is-the-discriminator). max(s, s-1) == s for
    // every finite s. Verify it survives on the disassembly, not on the source.
    aie::accum<accfloat, 32> ba;
    ba.from_vector(aie::load_v<32>(pb), 0);
    const aie::vector<float, 32> s =
        aie::add(aie::load_v<32>(pa32), ba.to_vector<float>());
    aie::store_v(pout, aie::max(s, aie::sub(s, aie::broadcast<float, 32>(1.0f))));
#elif defined(MODE4_BISECT_STORE_ONLY)
    aie::store_v(pout, aie::broadcast<float, 32>(3.0f));
#elif defined(MODE4_BISECT_LOAD_AB)
    // Widen both operands to f32 and add there. bf16 -> f32 is exact (bf16 is a truncated
    // f32), so the widening costs nothing numerically -- the same discipline residual_add.cc
    // uses on its bf16 arm, and it is why this matches a host f32 add bit-for-bit.
    // both operands loaded and widened, but only A is stored -- isolates the second LOAD from
    // the ADD. A dead widened B would be folded away, so it is kept live by a select the
    // compiler cannot resolve (off is a runtime value here only in the trivial sense, but the
    // store of the min keeps both accums observable).
    aie::accum<accfloat, 32> aa;
    aa.from_vector(aie::load_v<32>(pa), 0);
    aie::accum<accfloat, 32> bb;
    bb.from_vector(aie::load_v<32>(pb), 0);
#ifdef MODE4_BISECT_ADD
    aie::store_v(pout, aie::add(aa.to_vector<float>(), bb.to_vector<float>()));
#elif defined(MODE4_BISECT_SUB)
    // Same shape as the ADD arm with one token changed: vsub.f into an accumulator, stored
    // straight out of it. ADD hangs and MIN completes, but MIN also routes its result through
    // the vector file (sub-compare-select), so opcode and store path are confounded. This arm
    // holds the store path at ADD's and varies only the opcode.
    aie::store_v(pout, aie::sub(aa.to_vector<float>(), bb.to_vector<float>()));
#elif defined(MODE4_BISECT_ADD_VIAVEC)
    // The other cell: ADD's opcode with MIN's store path. The relu is a compare-select the
    // compiler cannot fold on unknown data, so the sum lands in the vector file before the
    // store. With the SUB arm this closes the 2x2 over (opcode, store register file).
    aie::store_v(pout, aie::max(aie::add(aa.to_vector<float>(), bb.to_vector<float>()),
                                aie::broadcast<float, 32>(0.0f)));
#elif defined(MODE4_BISECT_ADD_IDENTDETOUR)
    // Shippable form of the VIAVEC arm. The relu detours through the vector file but clamps at
    // zero, so it is not a resadd; max(s, s-1) == s for every finite s and detours the same way.
    // Verify on the disassembly, not on the source: the fold that would delete it also deletes
    // the reason this arm exists.
    const aie::vector<float, 32> s =
        aie::add(aa.to_vector<float>(), bb.to_vector<float>());
    aie::store_v(pout, aie::max(s, aie::sub(s, aie::broadcast<float, 32>(1.0f))));
#else
    aie::store_v(pout, aie::min(aa.to_vector<float>(), bb.to_vector<float>()));
#endif
#elif defined(MODE4_BISECT_LOAD_A)
    aie::accum<accfloat, 32> aa;
    aa.from_vector(aie::load_v<32>(pa), 0);
    aie::store_v(pout, aa.to_vector<float>());
#else
    aie::accum<accfloat, 32> aa;
    aa.from_vector(aie::load_v<32>(pa), 0);
    aie::accum<accfloat, 32> ba;
    ba.from_vector(aie::load_v<32>(pb), 0);
    aie::store_v(pout, aie::add(aa.to_vector<float>(), ba.to_vector<float>()));
#endif
#ifdef MODE4_ACCADD
    pa32 += 32;
#else
    pa += 32;
#endif
    pb += 32;
    pout += 32;
  }
  event1();
}

// MODAL bf16-out epilogue: same rtp[0] mode selection as the f32-out sibling, but
// the C tile is stored bf16. Used by the fc1 build that drains chunk-major straight
// into the fc2 K-split's input buffer, so no separate deinterleave+cast runs.
void mm_modal_epilogue_f32_bf16(const float *__restrict c_in,
                                bfloat16 *__restrict c_out,
                                const int32_t *__restrict rtp) {
#ifdef MODAL_DRAIN_ROUND_FLOOR
  // A/B arm for ru02-frame77-convergent-signature. Unlike the f32-out sibling above, this
  // drain sets NO rounding mode, so its bf16 narrow runs under whatever crRnd the previous
  // kernel on this core happened to leave -- the ambient-state dependence the frame-77
  // convergence is hypothesised to come from. Forcing floor EXPLICITLY (rather than just
  // omitting the set, which is what leaves it ambient) makes the drain deterministic, so a
  // damage delta against the shipped arm is attributable to the rounding mode and not to
  // whichever kernel ran first. Restored on exit -- this arm must not itself become the
  // ambient-state bug it is testing for.
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::floor);
#endif
  // rtp[0]: 0=identity, 1=silu, 2=gelu, 3=glu (same encoding as mm_modal_epilogue_f32_f32).
  if (rtp[0] == 3) {
    // Halves the live output width -- the drain tap takes 64 of every 128 columns.
    mm_glu_epilogue_bf16o<EPI_M, EPI_N, EPI_R, EPI_T>(c_in, c_out);
  } else if (rtp[0] == 1) {
    mm_silu_epilogue_bf16o_hiprec<EPI_M * EPI_N>(c_in, c_out);
  } else if (rtp[0] == 2) {
    mm_gelu_epilogue_bf16o<EPI_M * EPI_N>(c_in, c_out);
  } else {
    mm_identity_epilogue_bf16o<EPI_M * EPI_N>(c_in, c_out);
  }
#ifdef MODAL_DRAIN_ROUND_FLOOR
  aie::set_rounding(saved_rounding);
#endif
}

// MODAL int8 DEQUANT epilogue (L3): i32 acc -> f32 out, scaled by the per-dispatch
// scalar S delivered as the f32 bit-pattern in rtp[0] (the host bitcasts
// scale_a*w_scale into an i32 RTP slot before each dispatch). One mode (dequant);
// bias/SiLU run on the host (cheap). In-place: c_in and c_out alias the same 4B
// C tile. Reads S from RTP (not a build constant) so the resident xclbin serves
// every op — each dispatch patches its own S into the instruction stream's RTP.
void mm_modal_dequant_i32_f32(const int32_t *__restrict c_in,
                              float *__restrict c_out,
                              const int32_t *__restrict rtp) {
  union { int32_t i; float f; } s;
  s.i = rtp[0];
  mm_dequant_epilogue_i32_f32<EPI_M * EPI_N>(c_in, c_out, s.f);
}

} // extern "C"
