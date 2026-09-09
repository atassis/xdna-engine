// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Weight-quantized sibling of mv.cc: A (the GEMV's MxK matrix operand) streams as group-quantized
// int4 or int8 with a per-row, per-group scale (f32 or bf16, see SCALE_BF16 below), dequantized
// ON-CORE right before the same bf16xbf16 MAC mv.cc already uses. B (the vector) and C (the
// output) are unchanged bf16 -- only the WEIGHT-STREAM byte format is an axis here
// (iron/operators/gemv `weight_dtype`), never the arithmetic: narrow arithmetic buys nothing at
// M=1 decode (nothing to speed up, pack/unpack only adds ops), so this kernel still does a bf16
// MAC. The lever is DDR/L2 bytes for the weight, not FLOPs.
//
// Row layout (one row = one output feature), K elements, GROUP_SIZE-wide quant groups,
// n_groups = DIM_K / GROUP_SIZE:
//   [n_groups x scale][ payload ]
// payload is K/2 nibble-packed bytes (int4, low nibble = even column, high nibble = odd column --
// same convention as dequant_int4_group_row.cc) or K int8 bytes (int8, one byte per element).
// Packing the scale into the SAME buffer as the weight (rather than a 3rd input FIFO) is forced by
// the AIE2P tile DMA budget: a core has only 2 input channels, both already spent on A and B (see
// gemm_int8xint4_dequant.cc's identical constraint). A ROW-granularity header (not a
// tile-granularity one) keeps the row stride uniform, so the existing per-column contiguous TAP
// arithmetic in gemv/design.py needs no tiling-aware special case.
//
// Dequant follows the established, device-gated idiom (dequant_int4_group.cc /
// gemm_int8xint4_dequant.cc): scalar nibble/byte unpack into a float buffer, vector multiply by
// the (broadcast) group scale, narrow to bf16 via an explicit accum with conv_even rounding rather
// than a raw cast (the banked WER lesson: default truncation biases toward zero).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 64
#endif
// Per-group scale storage width: 0 (default) = f32, matching quant.py's default
// scale_dtype="f32" and today's row layout byte-for-byte. 1 = bf16 (quant.py's
// scale_dtype="bf16"): the header shrinks from 4 to 2 bytes/group -- e.g. 544 -> 528B at
// K=1024,GROUP_SIZE=128, 2.9% of the row -- and the scalar `(bfloat16)scale[...]` cast below
// becomes a no-op read instead of a narrowing one, since the value is already bf16 in memory. The
// two sides (this macro and quant.py's scale_dtype) must be set together; mismatched, the payload
// offset is wrong and every dequant reads garbage.
#ifndef SCALE_BF16
#define SCALE_BF16 0
#endif

namespace {

#if SCALE_BF16
using scale_t = bfloat16;
#else
using scale_t = float;
#endif

inline int8_t sext4(uint8_t nibble) {
  return (int8_t)(((int8_t)(nibble << 4)) >> 4);
}

// r: vector chunk width (VEC_SIZE); k: full row length (DIM_K); g: quant group width
// (GROUP_SIZE). A vector chunk must never straddle a group boundary, so g must be a multiple of r.
template <uint32_t r, uint32_t k, uint32_t g>
void matvec_int4_dequant(uint32_t m, const int8_t *__restrict a, const bfloat16 *__restrict b,
                         bfloat16 *__restrict c) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(k % g == 0, "DIM_K must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t row_stride = n_groups * sizeof(scale_t) + k / 2;
  // %4 is the shared bf16-granule arena's alignment (iron/common/sequence.py), not scale_t's own
  // (2-byte for bf16 would only need 2-byte alignment by itself) -- see quant.py's
  // row_stride_bytes for the full derivation, including the odd-n_groups bf16 case this
  // static_assert alone does not distinguish.
  static_assert(row_stride % 4 == 0, "row stride must be 4-byte aligned (per-row scale read)");
  constexpr uint32_t chunks_per_group = g / r;

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const scale_t *scale = reinterpret_cast<const scale_t *>(rowp);
    const int8_t *packed = reinterpret_cast<const int8_t *>(rowp + n_groups * sizeof(scale_t));
    // ONE flat loop and ONE reduce per row. Hoisting the scale per GROUP is arithmetically nicer
    // but costs a reduce_add per group (8 per row at k=1024,g=128) against the 16 vector muls it
    // saves, and a 64-lane reduce is a log-depth shuffle chain -- far more than a vector mul.
    // It also nests the loops, and hardware loops are innermost-only (contract K013).
    ::aie::accum<accfloat, r> acc = ::aie::zeros<accfloat, r>();
    uint32_t chunk = 0;
    for (const bfloat16 *__restrict b_cur = b; b_cur < b + k; b_cur += r, chunk++) {
      ::aie::vector<int8, r / 2> raw = ::aie::load_v<r / 2>(packed + chunk * (r / 2));
      ::aie::vector<int8, r> q8 = ::aie::unpack(::aie::vector_cast<int4>(raw));
      ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
      ::aie::vector<bfloat16, r> sv =
          ::aie::broadcast<bfloat16, r>((bfloat16)scale[(chunk * r) / g]);
      acc = ::aie::mac(acc, ::aie::mul(qbf, sv).template to_vector<bfloat16>(),
                       ::aie::load_v<r>(b_cur));
    }
    c[row] = static_cast<bfloat16>(::aie::reduce_add(acc.template to_vector<float>()));
  }
  ::aie::set_rounding(saved_rounding);
}

template <uint32_t r, uint32_t k, uint32_t g>
void matvec_int8_dequant(uint32_t m, const int8_t *__restrict a, const bfloat16 *__restrict b,
                         bfloat16 *__restrict c) {
  static_assert(g % r == 0, "GROUP_SIZE must be a multiple of VEC_SIZE");
  static_assert(k % g == 0, "DIM_K must be a whole number of groups");
  constexpr uint32_t n_groups = k / g;
  constexpr uint32_t row_stride = n_groups * sizeof(scale_t) + k;
  static_assert(row_stride % 4 == 0, "row stride must be 4-byte aligned (per-row scale read)");
  constexpr uint32_t chunks_per_group = g / r;

  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const uint8_t *a_bytes = reinterpret_cast<const uint8_t *>(a);
  for (uint32_t row = 0; row < m; row++) {
    const uint8_t *rowp = a_bytes + row * row_stride;
    const scale_t *scale = reinterpret_cast<const scale_t *>(rowp);
    const int8_t *packed = reinterpret_cast<const int8_t *>(rowp + n_groups * sizeof(scale_t));
    const bfloat16 *__restrict b_cur = b;
    float row_sum = 0.0f;
    for (uint32_t gi = 0; gi < n_groups; gi++) {
      // Same per-group scale hoist as the int4 form; int8 (-127..127) is exact in bf16 too.
      ::aie::accum<accfloat, r> acc = ::aie::zeros<accfloat, r>();
      for (uint32_t ci = 0; ci < chunks_per_group; ci++) {
        ::aie::vector<int8, r> q8 = ::aie::load_v<r>(packed + (gi * chunks_per_group + ci) * r);
        ::aie::vector<bfloat16, r> qbf = ::aie::to_float<bfloat16>(q8, 0);
        acc = ::aie::mac(acc, qbf, ::aie::load_v<r>(b_cur));
        b_cur += r;
      }
      row_sum += (float)scale[gi] * ::aie::reduce_add(acc.template to_vector<float>());
    }
    c[row] = static_cast<bfloat16>(row_sum);
  }
  ::aie::set_rounding(saved_rounding);
}

}  // namespace

extern "C" {

// Naming matches mv.cc's `matvec_{func_type}_{dtype_in}_{dtype_out}` convention
// (iron/operators/gemv/design.py builds the Kernel() name from the same template).
void matvec_vectorized_int4_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                 const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int4_dequant<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}

void matvec_vectorized_int8_bf16(uint32_t m, uint32_t row_offset, const int8_t *__restrict a_in,
                                 const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
  c_out += row_offset;
  matvec_int8_dequant<VEC_SIZE, DIM_K, GROUP_SIZE>(m, a_in, b_in, c_out);
}

}  // extern "C"
