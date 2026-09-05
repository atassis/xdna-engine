//===- subsample_patch_embed.cc --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parakeet conv2D /8 subsample front-end as the PATCH-EMBED idiom:
//   conv2d  ==  im2col (gather receptive-field patches on the host/DMA)
//               -> GEMM (this kernel)  -> [+ fused ReLU activation epilogue].
//
// COMPUTE brick: aie::mmul (the systolic matrix unit), NOT a mac+reduce scalar
// loop -- see internal notes (the ~5x mmul miss). The
// subsample is the M>=8 (M = Hout*Wout patch positions) compute-bound regime
// where the mmul/format bricks pay.
//
// SHAPE: A[M,K] (im2col patches, bf16) x B[K,N] (reshaped conv weight, bf16) ->
// C[M,N] (f32 accumulator) -> bf16 out. Tiles are pre-blocked for aie::mmul in
// the host generator (r x s, s x t, r x t), exactly like IRON aie2p/mm.cc, on
// which the inner 2x2-expanded mmul loop here is modeled.
//
// BIAS: folded into the GEMM via K-augmentation on the host (an extra k-block of
// A = ones / B = bias -> ones@bias = per-N bias added to every row), so the core
// takes only A and B (2 input DMA channels = the NPU2 compute-tile limit) and the
// ReLU below correctly sees relu(A@B + bias). conv.0 / conv.3 / conv.6 use the
// ReLU variant; the depthwise (conv.2/conv.5 -- on-device hot path = sliding_mul,
// task A1) and the final out-projection use the no-activation variant.
//
// ReLU is per-element (max(x,0)), so it is layout-independent w.r.t. the
// mmul-blocked C storage (same property the SiLU/GELU epilogues rely on): the
// output ObjectFifo de-shuffles blocked->row-major on the way out.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include "aie_kernels/aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// bf16 operands, f32 accumulate, optional fused ReLU, bf16 out.
// Modeled on IRON aie2p/mm.cc matmul_vectorized_2x2_mmul (2x2 m/n expansion for
// accumulator-register utilization). The full K reduction (colA s-blocks) stays
// in the mmul accumulator -- no horizontal reduce_add, which is the whole point
// of using the systolic brick.
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s,
          unsigned t, bool relu>
static inline void
patch_embed_mmul(const bfloat16 *__restrict pA, const bfloat16 *__restrict pB,
                 bfloat16 *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accfloat>;
  event0();

  const aie::vector<bfloat16, MMUL::size_C> zero =
      aie::zeros<bfloat16, MMUL::size_C>();

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(2, )
  for (unsigned z = 0; z < rowA; z += 2) {
      bfloat16 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
      bfloat16 *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2) {
        const bfloat16 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const bfloat16 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
        const bfloat16 *__restrict pB1 = pB + (j)*MMUL::size_B;
        const bfloat16 *__restrict pB2 = pB + (j + 1) * MMUL::size_B;

        aie::vector<bfloat16, MMUL::size_A> A0, A1;
        aie::vector<bfloat16, MMUL::size_B> B0, B1;

        MMUL C00, C01, C10, C11;

        for (unsigned i = 0; i < colA; ++i) {
          A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
          A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
          B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * colB;
          B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * colB;
          if (i == 0) {
            C00.mul(A0, B0); C01.mul(A0, B1);
            C10.mul(A1, B0); C11.mul(A1, B1);
          } else {
            C00.mac(A0, B0); C01.mac(A0, B1);
            C10.mac(A1, B0); C11.mac(A1, B1);
          }
        }

        // Fused activation epilogue: drain acc -> bf16, relu = max(x, 0).
        aie::vector<bfloat16, MMUL::size_C> v00 = C00.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> v01 = C01.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> v10 = C10.template to_vector<bfloat16>();
        aie::vector<bfloat16, MMUL::size_C> v11 = C11.template to_vector<bfloat16>();
        if constexpr (relu) {
          v00 = aie::max(v00, zero); v01 = aie::max(v01, zero);
          v10 = aie::max(v10, zero); v11 = aie::max(v11, zero);
        }
        aie::store_v(pC1, v00); pC1 += MMUL::size_C;
        aie::store_v(pC1, v01); pC1 += MMUL::size_C;
        aie::store_v(pC2, v10); pC2 += MMUL::size_C;
        aie::store_v(pC2, v11); pC2 += MMUL::size_C;
      }
    }

  event1();
}

extern "C" {

// Tile dims at compile time; must be divisible by r/s/t (default mmul 8x8x8 for
// bf16 with bfp16 emulation, else 4x8x8). conv.0: M=Hout*Wout, K=(Cin*kh*kw)+1
// (K-aug bias), N=Cout. The host pre-tiles to these.
#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#define PE_R 8
#define PE_S 8
#define PE_T 8
#else
#define PE_R 4
#define PE_S 8
#define PE_T 8
#endif

// relu(A@B + bias) -- conv.0 / conv.3 / conv.6 patch-embed GEMMs.
void patch_embed_relu_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  patch_embed_mmul<(DIM_M / PE_R), (DIM_K / PE_S), (DIM_N / PE_T), PE_R, PE_S,
                   PE_T, /*relu=*/true>(a_in, b_in, c_out);
}

// A@B + bias (no activation) -- depthwise GEMM blocks + the out-projection.
void patch_embed_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  patch_embed_mmul<(DIM_M / PE_R), (DIM_K / PE_S), (DIM_N / PE_T), PE_R, PE_S,
                   PE_T, /*relu=*/false>(a_in, b_in, c_out);
}

} // extern "C"
