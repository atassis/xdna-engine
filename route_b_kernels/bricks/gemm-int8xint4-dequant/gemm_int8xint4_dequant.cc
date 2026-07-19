// gemm-int8xint4-dequant brick [group: matmul; layer: FORMAT + COMPUTE]
//
// The FUSED int8xint4 GEMM + POST-mmul group-dequant epilogue -- the "natural
// next brick" that dequant-int4-group.cc names in its header (path (b)): now
// that a paired int8 x int4 systolic mmul exists (gemm-int8xint4.cc), fold the
// per-group weight scale onto the acc32 partial sums AFTER the mmul, at GROUP
// granularity along K, instead of dequanting every int4 element to bf16 before
// a plain bf16 mmul. This is the rails-grade int4 FORMAT lever
// (int4-dequant-brick): ~4x fewer weight bytes/token on the LPDDR wall at M=1
// decode, one generic parameterized op-TYPE against the resident [tile,D]
// stream contract -- NOT a per-model kernel.
//
// MATH (symmetric group-quant, AWQ int4; the common case, HAS_ZP=0):
//   C[M,N] (bf16) = sum over K-groups g of  scale[g, n] * (A[M, Kg] @ Bq[Kg, N])
// where B is int4, group-quantized along K with group size G (a multiple of the
// native 16-wide K tile), and scale[g, n] is the per-group, per-output-column
// f32 dequant scale ([K/G, N]). Because different K-groups carry different
// scales, you canNOT sum all of K in one accumulator and fold one scale: you
// accumulate WITHIN a group in the native int32 accumulator (exact int math),
// convert that group partial to f32, multiply by the group scale, and sum the
// group contributions in f32 -- the standard AWQ dequant-GEMM structure.
// Narrow the f32 result to bf16 in the epilogue with conv_even rounding (banked
// WER lesson: the default truncation biases toward zero).
//
// ASYMMETRIC (zero-point) is deliberately OUT of scope for this brick: a
// post-mmul zp fold needs the extra per-group sum_k A[m,k] correction term
// (scale*(A@Bq - zp*rowsum(A))), a second reduction. YAGNI -- AWQ int4 is
// typically symmetric; the zp variant is a documented follow-on, mirroring the
// HAS_ZP gating in dequant-int4-group.cc.
//
// LAYOUT / verification convention (matches the gemm-int8xint4 sibling exactly):
//   - The GOLDEN (golden.py) computes plain row-major groupwise-dequant matmul;
//     tiling is a device-harness concern, not the golden's math (same split the
//     gemm-int8xint4 golden documents: pack/tile only matter for DMA, not math).
//   - This KERNEL reads A and B in the sibling's TILE-BLOCKED order
//     (A: [mTiles][kSteps][4x16], B: [kSteps][nTiles][16x16] packed int4) and
//     writes C tile-blocked ([mTiles][nTiles][4x16] bf16). scale is read as a
//     natural [numGroups, N] row-major buffer. The (future) device-verify
//     harness pre-tiles A/B/scale and un-tiles C, then gates rel-L2 vs golden.
//
// Build (CPU-only, no device): plain -O2, -I mlir_aie/include, self-contained
// (aie_api/aie.hpp + stdint.h) -- same class as gemm-int8xint4.cc.
//
// STATUS: compiles-clean (Peano clang acc2a72c, --target=aie2p). Numerical
// correctness is UNVERIFIED (no device this run) -> int4-dequant-brick device
// rel-L2 gate + a J/token energy delta are the deferred, single-tenant gate.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// One (mi, ni) output tile of the fused group-dequant GEMM.
// Walks K as numGroups groups of (G/16) native 4x16x16 mmul_8_4 steps; each
// group's int32 partial is scaled by its per-column f32 group scale and summed
// in f32; the f32 tile is narrowed to bf16 once at the end.
template <unsigned M, unsigned K, unsigned N, unsigned G>
static inline void gemm_int8xint4_dequant_tile(const int8_t *__restrict pA,
                                               const int4 *__restrict pB,
                                               const float *__restrict pScale,
                                               bfloat16 *__restrict pC) {
  static_assert(M % 4 == 0, "M must be a multiple of 4 (mmul_8_4 tile)");
  static_assert(K % 16 == 0, "K must be a multiple of 16 (mmul_8_4 tile)");
  static_assert(N % 16 == 0, "N must be a multiple of 16 (mmul_8_4 tile)");
  static_assert(G % 16 == 0, "group size G must be a multiple of the 16-wide K tile");
  static_assert(K % G == 0, "K must be a whole number of groups");

  using MMUL = aie::mmul<4, 16, 16, int8_t, int4, accauto>;

  constexpr unsigned kStepsPerGroup = G / 16;
  constexpr unsigned numGroups = K / G;
  constexpr unsigned kSteps = K / 16;
  constexpr unsigned mTiles = M / 4;
  constexpr unsigned nTiles = N / 16;

  // Round-to-nearest-even for the bf16 narrow (banked WER lesson).
  ::aie::set_rounding(::aie::rounding_mode::conv_even);

  for (unsigned mi = 0; mi < mTiles; ++mi) {
    for (unsigned ni = 0; ni < nTiles; ++ni) {
      // f32 running sum for this [4,16] output tile, over all K-groups.
      ::aie::vector<float, MMUL::size_C> cf =
          ::aie::zeros<float, MMUL::size_C>();

      for (unsigned g = 0; g < numGroups; ++g) {
        // Exact int8xint4 -> int32 partial for this group's K-slice.
        MMUL acc;
        for (unsigned ks = 0; ks < kStepsPerGroup; ++ks) {
          const unsigned ki = g * kStepsPerGroup + ks;
          const int8_t *pA_tile = pA + (mi * kSteps + ki) * MMUL::size_A;
          const int4 *pB_tile = pB + (ki * nTiles + ni) * MMUL::size_B;

          ::aie::vector<int8_t, MMUL::size_A> A =
              ::aie::load_v<MMUL::size_A>(pA_tile);
          ::aie::vector<int4, MMUL::size_B> B =
              ::aie::load_v<MMUL::size_B>(pB_tile);

          if (ks == 0)
            acc.mul(A, B);
          else
            acc.mac(A, B);
        }

        // group partial [4,16] int32 -> f32
        ::aie::vector<float, MMUL::size_C> pf =
            ::aie::to_float(acc.template to_vector<int32_t>(), 0);

        // per-output-column group scale for columns [ni*16, ni*16+16),
        // replicated across the 4 M rows so it lines up with the row-major
        // [4,16] tile (element r*16 + c uses scale of column c).
        const float *sg = pScale + g * N + ni * 16;
        float sbuf[MMUL::size_C];
        for (unsigned r = 0; r < 4; ++r)
          for (unsigned c = 0; c < 16; ++c)
            sbuf[r * 16 + c] = sg[c];
        ::aie::vector<float, MMUL::size_C> sv =
            ::aie::load_v<MMUL::size_C>(sbuf);

        // aie::mul of two float vectors yields an accfloat accum; narrow the
        // elementwise product back to a vector before the running f32 add.
        cf = ::aie::add(cf, ::aie::mul(pf, sv).template to_vector<float>());
      }

      // narrow the f32 tile to bf16 (conv_even) and store tile-blocked.
      ::aie::accum<accfloat, MMUL::size_C> a;
      a.from_vector(cf);
      bfloat16 *pC_tile = pC + (mi * nTiles + ni) * MMUL::size_C;
      ::aie::store_v(pC_tile, a.template to_vector<bfloat16>());
    }
  }
}

extern "C" {

// Fixed instantiations the golden/verify harness exercises. Grow this
// vocabulary per model instance (earn-from-instance YAGNI) -- do NOT
// template-explode every M/K/N/G up front. G=64 is the typical AWQ group size
// (G % 16 == 0, K % G == 0 hold for the shapes below).
void gemm_int8xint4_dequant_8x64x64_g64(const int8_t *__restrict a,
                                        const int4 *__restrict b,
                                        const float *__restrict scale,
                                        bfloat16 *__restrict c) {
  gemm_int8xint4_dequant_tile<8, 64, 64, 64>(a, b, scale, c);
}

void gemm_int8xint4_dequant_64x128x128_g64(const int8_t *__restrict a,
                                           const int4 *__restrict b,
                                           const float *__restrict scale,
                                           bfloat16 *__restrict c) {
  gemm_int8xint4_dequant_tile<64, 128, 128, 64>(a, b, scale, c);
}

// M=1-style decode shape (rounded up to the native M=4 tile): tall-K FFN
// weight, the actual int4 energy-lever regime.
void gemm_int8xint4_dequant_4x1024x4096_g128(const int8_t *__restrict a,
                                             const int4 *__restrict b,
                                             const float *__restrict scale,
                                             bfloat16 *__restrict c) {
  gemm_int8xint4_dequant_tile<4, 1024, 4096, 128>(a, b, scale, c);
}

} // extern "C"
