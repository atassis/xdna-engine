// gemm-int8 brick [group: matmul]
//
// Generic M>=8 int8 GEMM op-TYPE for the resident [tile,D] stream contract:
// C[M,N] (int32 acc) = A[M,K] (int8) @ B[K,N] (int8), tiled over the AIE2P
// aie::mmul<8,8,8,int8,int8,accauto> systolic tile. This is a PARAMETERIZED
// primitive (M/K/N are template args resolved per dispatch by the schedule,
// not per-model hand-fusion) -- brick catalog entry, matmul group, compute-
// bound M>=8 regime (see docs/aie2p-brick-catalog.md).
//
// API pattern studied from route_b_kernels/ffn_bfp16/repro_847_mmul888_bfp16.cc
// and mlir-aie/aie_kernels/aie2p/mm.cc (matmul_vectorized_8x8x8_i8_i32):
//   aie::mmul<M,K,N,int8,int8,accauto> MMUL;
//   load_v -> acc.mul(A,B) [first K-step] / acc.mac(A,B) [subsequent K-steps]
//   -> acc.to_vector<int32>() -> store_v
//
// Deliberately NOT the mm.cc 2x2-double-pumped variant (that helper depends
// on file-local is_b_row_maj/is_c_row_maj macros defined elsewhere in mm.cc);
// this brick is self-contained (aie_api/aie.hpp + stdint.h only) so it drops
// into any kernel-object build without pulling in mm.cc's macro soup. Once
// on-device tuning wants the double-pump 2-column variant, that is a second,
// still-generic brick (or a template parameter), not a per-model fork.
//
// Tiling: single AIE core, one row-tile x col-tile of C at a time, walking K
// in 8-wide steps accumulating in the MMUL's native int32 accumulator before
// a single to_vector<int32>() writeback per output tile. M,K,N must each be
// a multiple of 8 (the native aie::mmul<8,8,8,...> tile shape); this brick
// does not itself decide the outer schedule (that is schedule/IRON-side,
// generic-rails "model = schedule over bricks").
//
// Build (CPU-only, no device): see gotchas in the task/README -- plain -O2,
// -I mlir-aie/third_party/aie_api/include, no extra -D's needed (this file
// is self-contained, same class as affine_cast.cc).

#include <aie_api/aie.hpp>
#include <stdint.h>

// One 8x8x8 int8xint8->int32 systolic-tile matmul, single MMUL instance,
// A/B/C all held resident (no host round-trip) -- the atomic op this brick
// tiles across for M/K/N > 8.
template <unsigned M, unsigned K, unsigned N>
static inline void gemm_int8_tile(const int8_t *__restrict pA,
                                   const int8_t *__restrict pB,
                                   int32_t *__restrict pC) {
  static_assert(M % 8 == 0, "gemm_int8: M must be a multiple of 8 (mmul tile)");
  static_assert(K % 8 == 0, "gemm_int8: K must be a multiple of 8 (mmul tile)");
  static_assert(N % 8 == 0, "gemm_int8: N must be a multiple of 8 (mmul tile)");

  using MMUL = aie::mmul<8, 8, 8, int8_t, int8_t, accauto>;

  constexpr unsigned kSteps = K / 8;

  for (unsigned mi = 0; mi < M / 8; ++mi) {
    for (unsigned ni = 0; ni < N / 8; ++ni) {
      MMUL acc;

      for (unsigned ki = 0; ki < kSteps; ++ki) {
        const int8_t *pA_tile = pA + (mi * kSteps + ki) * MMUL::size_A;
        const int8_t *pB_tile = pB + (ki * (N / 8) + ni) * MMUL::size_B;

        aie::vector<int8_t, MMUL::size_A> A =
            aie::load_v<MMUL::size_A>(pA_tile);
        aie::vector<int8_t, MMUL::size_B> B =
            aie::load_v<MMUL::size_B>(pB_tile);

        if (ki == 0) {
          acc.mul(A, B);
        } else {
          acc.mac(A, B);
        }
      }

      int32_t *pC_tile = pC + (mi * (N / 8) + ni) * MMUL::size_C;
      aie::store_v(pC_tile, acc.template to_vector<int32_t>());
    }
  }
}

extern "C" {

// Fixed instantiations for the sizes the golden/verify harness exercises
// (grow this vocabulary per-model instance, per the earn-from-instance rule
// -- do NOT template-explode every M/K/N up front).
void gemm_int8_8x8x8(const int8_t *__restrict a, const int8_t *__restrict b,
                      int32_t *__restrict c) {
  gemm_int8_tile<8, 8, 8>(a, b, c);
}

void gemm_int8_64x64x64(const int8_t *__restrict a,
                         const int8_t *__restrict b,
                         int32_t *__restrict c) {
  gemm_int8_tile<64, 64, 64>(a, b, c);
}

void gemm_int8_128x128x128(const int8_t *__restrict a,
                            const int8_t *__restrict b,
                            int32_t *__restrict c) {
  gemm_int8_tile<128, 128, 128>(a, b, c);
}

} // extern "C"
