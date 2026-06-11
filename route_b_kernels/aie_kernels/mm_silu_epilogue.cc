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

} // extern "C"
