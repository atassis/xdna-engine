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

// GELU (tanh approx, matches torch gelu(approximate="tanh") + the decode gelu_tile_bf16):
//   gelu(x) = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
// f32 acc in -> bf16 gelu -> f32 out (mirrors the silu f32o path). Used by the modal GELU mode (rtp[0]==2)
// to fold the Whisper encoder FFN activation into the fc1 GEMM epilogue (drops the ~260 ms/utt host GELU).
template <int size>
static inline void mm_gelu_epilogue_f32o(const float *__restrict pC_in,
                                         float *__restrict pC_out) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  // NOTE (2026-06-28): a full-f32 polynomial GELU here recovers the RU precision in theory but HANGS
  // on-device ("run_matmul8: kernel run did not complete") -- f32 elementwise/transcendental ops are NOT
  // free on this bf16-native unit (the cube + tanh + finish in f32 blow the cycle budget; both the pure-f32
  // and the f32-cube+bf16-tanh hybrid timed out). So this stays the bf16 tanh-approx (fast, ships). The
  // bf16 GELU costs RU +0.4 (whisper-resident-ffn-wer-gate); recovering it needs a bf16-NATIVE precision
  // trick (e.g. keep the cube in the f32 accumulator without full-f32 vectors), parked as kernel R&D.
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  const aie::vector<bfloat16, 16> c0 = aie::broadcast<bfloat16, 16>(0.7978845608f); // sqrt(2/pi)
  const aie::vector<bfloat16, 16> c1 = aie::broadcast<bfloat16, 16>(0.044715f);
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
  // rtp[0]: 0=identity, 1=silu, 2=gelu (modal modes; baked per instruction stream).
  if (rtp[0] == 1) {
    mm_silu_epilogue_f32o<EPI_M * EPI_N>(c_in, c_out);
  } else if (rtp[0] == 2) {
    mm_gelu_epilogue_f32o<EPI_M * EPI_N>(c_in, c_out);
  } else {
    mm_identity_epilogue_f32o<EPI_M * EPI_N>(c_in, c_out);
  }
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
