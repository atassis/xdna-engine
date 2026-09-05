//===- dequant_int4_group.cc -----------------------------------*- C++ -*-===//
//
// BRICK: dequant-int4-group  [layer: FORMAT]
//
// Generic int4 GROUP-quant weight -> bf16 dequant epilogue (AWQ-style). This is
// the FORMAT-layer primitive the brick catalog lists as X (unused): "int4 | 4b +
// group scale/zp | (paired only) | AWQ, untested" and "int8 x int4 | 4x16x16,
// 1024 MAC/issue densest | ref dequants to bf16 instead". Two ways this brick is
// meant to be consumed:
//
//   (a) STANDALONE dequant: unpack a whole int4-packed [tile,D] weight/activation
//       stream to bf16 BEFORE a plain bf16 mmul (the "ref dequants to bf16"
//       path noted in the catalog) -- what this file implements/verifies.
//   (b) POST-mmul group-scale fold: when paired with a native int8 x int4
//       systolic mmul (4x16x16 tile), the K-reduction accumulator (acc32) comes
//       out ALREADY int-summed; you do NOT dequant every element -- you fold
//       the group scale onto the acc32 partial sums after the mmul, at GROUP
//       granularity along K (see the catalog's "Precision rule": accumulate
//       f32/acc32, fold quant scales POST-mmul at GROUP granularity). That
//       epilogue shares the SAME per-group-scale broadcast idiom below (the
//       `GROUP % N == 0` contract keeps one vector chunk inside one group so
//       the scale is a single broadcast, never a per-lane gather) but reads an
//       acc32 accumulator instead of raw int4 -- left as the natural next
//       brick once a paired int8 x int4 mmul kernel exists to attach it to.
//
// CONTRACT (generic op-TYPE, not per-model): resident [tile, D] stream, ONE row
// of `cols` int4-packed values per call (mirrors the affine_cast.cc /
// glu.cc per-row core_body convention). int4 is packed 2 values/byte, LOW
// nibble = even column index, HIGH nibble = odd column index (little-endian
// nibble order), sign-extended 4-bit two's complement in [-8, 7].
//
// GROUP granularity: `cols` is partitioned into cols/GROUP contiguous groups,
// each sharing ONE f32 scale and (optionally) ONE int8 zero-point:
//     dequant(q) = (q - zp) * scale        (symmetric: zp == 0, HAS_ZP off)
// Contract `GROUP % N == 0` (N = the bf16 output vector width) guarantees every
// N-wide chunk lives inside exactly one group, so scale/zp are a single
// broadcast per chunk -- never a per-lane gather (the parallel_lookup brick is
// reserved for an actual embedding/codebook use, not this affine per-group
// case). `cols % GROUP == 0` and GROUP even (nibble-byte alignment).
//
// Unpack is scalar (nibble extract cannot vector-load as a native dtype
// portably here); the scale/zero-point apply and bf16 narrow are vectorized,
// matching affine_cast.cc's f32 accum -> to_vector<bfloat16> narrowing idiom
// with `conv_even` rounding (round-to-nearest-even, not the WER-regressing
// truncation default -- banked lesson from affine_cast.cc / glu.cc).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef HAS_ZP
#define HAS_ZP 0   // 0 = symmetric (zp folded away at compile time); 1 = asymmetric
#endif

// Sign-extend a 4-bit nibble (0..15) to a signed value in [-8, 7].
static inline int8_t sext4(uint8_t nibble) {
  return (int8_t)(((int8_t)(nibble << 4)) >> 4);
}

template <int N, int GROUP>
void dequant_int4_group_row(const uint8_t *restrict packed, // ceil(cols/2) bytes
                             const float *restrict scale,    // [cols/GROUP]
                             const int8_t *restrict zp,      // [cols/GROUP], ignored if !HAS_ZP
                             bfloat16 *restrict output,       // [cols]
                             int32_t cols) {
  static_assert(GROUP % N == 0, "GROUP must be a multiple of N (one chunk stays in one group)");
  event0();
  // Round-to-nearest-even, matching the host AVX512 pack_f32_to_bf16 path
  // (banked WER lesson: default truncation biases toward zero).
  const auto saved_rounding = ::aie::swap_rounding(::aie::rounding_mode::conv_even);

  float unpacked[N];
  for (int i = 0; i < cols; i += N) {
    const int group = i / GROUP;
    const float s = scale[group];
#if HAS_ZP
    const float z = (float)zp[group];
#else
    const float z = 0.0f;
#endif
    // scalar nibble unpack (byte-aligned: i and N both even by contract)
    for (int n = 0; n < N; n += 2) {
      const uint8_t byte = packed[(i + n) / 2];
      unpacked[n]     = (float)sext4(byte & 0x0F)        - z;
      unpacked[n + 1] = (float)sext4((byte >> 4) & 0x0F) - z;
    }

    ::aie::vector<float, N> q = ::aie::load_v<N>(unpacked);
    ::aie::vector<float, N> sv = ::aie::broadcast<float, N>(s);
    ::aie::vector<float, N> y = ::aie::mul(q, sv); // (q - zp) * scale
    ::aie::accum<accfloat, N> a;
    a.from_vector(y);
    ::aie::store_v(output + i, a.template to_vector<bfloat16>());
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
// Default instantiation: N=16 (one bf16 vector store width), GROUP=64
// (typical AWQ group size, GROUP % N == 0 holds for 64/128/256...).
void dequant_int4_group_row(uint8_t *packed, float *scale, int8_t *zp,
                             bfloat16 *output, int32_t cols) {
  dequant_int4_group_row<16, 64>(packed, scale, zp, output, cols);
}
}
