//===- gemv_int8.cc -------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// GENERIC int8 x int8 GEMV brick (op-TYPE, not a per-model kernel).
//
// Decode-regime matvec: a resident [tile,D] stream where the live row count is
// M=1. The AIE2P mmul intrinsic needs its M-dimension ("r") divisible tile, so
// (same convention as norm_gemv_prologue.cc) the caller pads the resident A
// tile to GEMV_M = r rows: row 0 is the real query row, rows 1..GEMV_M-1 are
// zero padding carried only to satisfy the mmul shape -- callers read back row
// 0 of C and ignore the rest. This keeps the kernel itself fully generic over
// (M,K,N): it never special-cases M=1, it just tiles it like any other M.
//
// API pattern studied from norm_gemv_prologue.cc / mlir-aie aie_kernels/aie2p/mm.cc:
//   aie::mmul<r,s,t,int8,int8,accauto> acc;  load_v -> acc.mul (k=0) / acc.mac (k>0)
//   -> acc.to_vector<int32_t>()
// i8 x i8 -> i32 is the native AIE2P int8 mmul shape (r=s=t=8, matches
// matmul_vectorized_8x8x8_i8_i32 in aie_kernels/aie2p/mm.cc); no bfp16/bf16
// emulation path is needed for pure int8.
//
// Two entry points, both over the SAME core loop:
//   gemv_i8_i8_i32   -- raw i32 accumulator out (bring-your-own dequant)
//   gemv_i8_i8_f32   -- fused single-scalar dequant to f32 out, S = scale_a *
//                       w_scale delivered as one RTP f32 (mirrors the L3
//                       mm_dequant_epilogue_i32_f32 pattern in
//                       mm_silu_epilogue.cc, folded into THIS op instead of a
//                       separate epilogue dispatch -- one less inter-op
//                       transition on the decode critical path).
//
// Tile shapes are compile-time macros (GEMV_M/GEMV_K/GEMV_N), same convention
// as DIM_M/DIM_K/DIM_N in aie_kernels/aie2p/mm.cc and EPI_M/EPI_K/EPI_N in this
// dir's other epilogues -- a MODEL supplies these as data (a block schedule),
// never hard-coded per model in this file.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GEMV_M
#define GEMV_M 8 // padded row count; must be a multiple of r (row 0 = live query)
#endif
#ifndef GEMV_K
#define GEMV_K 768 // reduction dim (hidden size)
#endif
#ifndef GEMV_N
#define GEMV_N 768 // output width
#endif

using namespace aie;

// Generic resident-tile int8 GEMV core: C[M,N] (i32) = A[M,K] (i8) @ B[K,N] (i8).
// A, B, C are all row-major, tile-blocked in mmul-native (r,s,t) order exactly
// like matmul_vectorized_2x2_mmul in aie_kernels/aie2p/mm.cc (single, not 2x2,
// expansion -- M is meant to stay small/padded-decode-shaped, so the extra
// register pressure of a 2x2 unroll buys nothing here).
template <unsigned M, unsigned K, unsigned N, unsigned r, unsigned s, unsigned t>
static inline void gemv_int8_core(const int8 *__restrict pA,
                                  const int8 *__restrict pB,
                                  int32_t *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, int8, int8, accauto>;
  static_assert(M % r == 0, "GEMV_M must be a multiple of the mmul r-dim");
  static_assert(K % s == 0, "GEMV_K must be a multiple of the mmul s-dim");
  static_assert(N % t == 0, "GEMV_N must be a multiple of the mmul t-dim");

  event0();
  constexpr unsigned Mb = M / r;
  constexpr unsigned Kb = K / s;
  constexpr unsigned Nb = N / t;

  for (unsigned m = 0; m < Mb; ++m) {
    const int8 *__restrict pA_row = pA + m * Kb * MMUL::size_A;
    for (unsigned n = 0; n < Nb; ++n) {
      const int8 *__restrict pA_i = pA_row;
      const int8 *__restrict pB_i = pB + n * MMUL::size_B;
      MMUL acc;
      for (unsigned k = 0; k < Kb; ++k) {
        aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(pA_i);
        pA_i += MMUL::size_A;
        aie::vector<int8, MMUL::size_B> b = aie::load_v<MMUL::size_B>(pB_i);
        pB_i += MMUL::size_B * Nb;
        if (k == 0) {
          acc.mul(a, b);
        } else {
          acc.mac(a, b);
        }
      }
      aie::vector<int32_t, MMUL::size_C> cv = acc.template to_vector<int32_t>();
      aie::store_v(pC + (m * Nb + n) * MMUL::size_C, cv);
    }
  }
  event1();
}

// Same core, but fused single-scalar dequant to f32 (i32 acc * S -> f32), so
// the resident xclbin needs no separate elementwise epilogue dispatch on the
// decode critical path. S is a per-dispatch RTP scalar (scale_a * w_scale),
// delivered as the raw f32 bit pattern (same convention as
// mm_modal_dequant_i32_f32 in mm_silu_epilogue.cc).
template <unsigned M, unsigned K, unsigned N, unsigned r, unsigned s, unsigned t>
static inline void gemv_int8_dequant_core(const int8 *__restrict pA,
                                          const int8 *__restrict pB,
                                          float *__restrict pC,
                                          float scale) {
  using MMUL = aie::mmul<r, s, t, int8, int8, accauto>;
  static_assert(M % r == 0, "GEMV_M must be a multiple of the mmul r-dim");
  static_assert(K % s == 0, "GEMV_K must be a multiple of the mmul s-dim");
  static_assert(N % t == 0, "GEMV_N must be a multiple of the mmul t-dim");

  event0();
  constexpr unsigned Mb = M / r;
  constexpr unsigned Kb = K / s;
  constexpr unsigned Nb = N / t;
  const aie::vector<float, MMUL::size_C> sv =
      aie::broadcast<float, MMUL::size_C>(scale);

  for (unsigned m = 0; m < Mb; ++m) {
    const int8 *__restrict pA_row = pA + m * Kb * MMUL::size_A;
    for (unsigned n = 0; n < Nb; ++n) {
      const int8 *__restrict pA_i = pA_row;
      const int8 *__restrict pB_i = pB + n * MMUL::size_B;
      MMUL acc;
      for (unsigned k = 0; k < Kb; ++k) {
        aie::vector<int8, MMUL::size_A> a = aie::load_v<MMUL::size_A>(pA_i);
        pA_i += MMUL::size_A;
        aie::vector<int8, MMUL::size_B> b = aie::load_v<MMUL::size_B>(pB_i);
        pB_i += MMUL::size_B * Nb;
        if (k == 0) {
          acc.mul(a, b);
        } else {
          acc.mac(a, b);
        }
      }
      aie::vector<int32_t, MMUL::size_C> civ = acc.template to_vector<int32_t>();
      aie::vector<float, MMUL::size_C> cfv = aie::to_float(civ, 0);
      // MMUL::size_C (64 for the 8x8x8 int8 shape) exceeds the accum lane
      // limit for a direct vector<float> result (accum.hpp requires
      // size() <= 32 for that implicit conversion), so aie::mul here yields
      // an accum -- take it explicitly and narrow via to_vector, same idiom
      // as the f32 epilogues in mm_silu_epilogue.cc.
      aie::accum<accfloat, MMUL::size_C> oacc = aie::mul(cfv, sv);
      aie::vector<float, MMUL::size_C> outv = oacc.template to_vector<float>();
      aie::store_v(pC + (m * Nb + n) * MMUL::size_C, outv);
    }
  }
  event1();
}

extern "C" {

// Native AIE2P int8 mmul shape (r=s=t=8), same as matmul_vectorized_8x8x8_i8_i32
// in aie_kernels/aie2p/mm.cc.
void gemv_i8_i8_i32(const int8 *__restrict a_in, const int8 *__restrict b_in,
                    int32_t *__restrict c_out) {
  gemv_int8_core<GEMV_M, GEMV_K, GEMV_N, 8, 8, 8>(a_in, b_in, c_out);
}

// Fused dequant variant: scale delivered as one f32 RTP slot (bitcast from the
// host's int32 RTP the same way mm_modal_dequant_i32_f32 does it).
void gemv_i8_i8_f32(const int8 *__restrict a_in, const int8 *__restrict b_in,
                    float *__restrict c_out, const int32_t *__restrict rtp) {
  union {
    int32_t i;
    float f;
  } s;
  s.i = rtp[0];
  gemv_int8_dequant_core<GEMV_M, GEMV_K, GEMV_N, 8, 8, 8>(a_in, b_in, c_out, s.f);
}

} // extern "C"
