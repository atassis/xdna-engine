// gemm-int8xint4 brick [group: matmul]
//
// Generic M>=8 mixed-precision GEMM op-TYPE for the resident [tile,D] stream
// contract: C[M,N] (int32 acc) = A[M,K] (int8) @ B[K,N] (int4, packed 2/byte),
// tiled over the AIE2P aie::mmul<M,K,N,int8,int4,accauto> "mmul_8_4" unit.
// PARAMETERIZED primitive (M/K/N are template args resolved per dispatch by
// the schedule, not per-model hand-fusion) -- brick catalog entry, matmul
// group, compute-bound M>=8 regime (see docs/aie2p-brick-catalog.md).
//
// KEY brick per the task: this is the first int8xint4 mixed-precision GEMM in
// the catalog. If it ICEs or mislowers, that is a candidate llvm-aie PR, not a
// workaround-in-place (toolchain-bug-test-latest-before-workaround KB stance).
//
// API pattern studied from:
//   - route_b_kernels/ffn_bfp16/repro_847_mmul888_bfp16.cc (mmul<M,K,N,TA,TB,
//     accauto> -> load_v -> acc.mul/mac -> acc.to_vector<Out>() shape).
//   - bricks/gemm-int8/gemm_int8.cc (sibling brick, same group/style; int8x
//     int8 uses a 2-arg mac(A,B) because that specialization (mmul_8_8) infers
//     sign statically from the C++ type).
//   - mlir-aie/third_party/aie_api/include/aie_api/detail/aie2p/mmul_8_4.hpp:
//     the ONLY hardware tile shape this specialization provides on AIE2P is
//     mmul_8_4<M=4, K=16, N=16, TypeA, TypeB, AccumBits=32> (unlike mmul_8_8,
//     there is no free M/K/N choice here -- 4x16x16 is the native systolic
//     tile for the 8b x 4b array mode). The 8b x 4b hardware mode is shared
//     between signed/unsigned lanes, so at the detail::mmul_8_4 level mac/mul
//     take an EXPLICIT runtime sign bool per operand
//     (mac(vector_A_type a, bool a_sign, vector_B_type b, bool b_sign)); the
//     PUBLIC aie::mmul wrapper (aie.hpp) exposes the ordinary 2-arg
//     mac(a,b)/mul(a,b) instead and infers each sign from the operand's C++
//     value_type via detail::get_mul_sign() before forwarding to the 4-arg
//     detail call -- so this brick, like gemm-int8's mmul_8_8 path, just uses
//     the plain 2-arg form (both int8_t and int4 are signed, so the inferred
//     sign is always true/true).
//
// Tiling: this brick walks the GEMM's outer M/K/N (each an independent
// template arg, any multiple of the native 4x16x16 tile) as a grid of
// 4x16x16 mmul_8_4 dispatches on ONE AIE core, accumulating over K in the
// native int32 accumulator before a single to_vector<int32>() writeback per
// output tile -- mirrors gemm-int8's tiling shape exactly, just with the
// fixed 4x16x16 native tile instead of a free 8x8x8 choice. Not the mm.cc
// 2x2-double-pumped variant (that helper depends on file-local macros defined
// elsewhere in mm.cc); this brick is self-contained (aie_api/aie.hpp +
// stdint.h only) so it drops into any kernel-object build without pulling in
// mm.cc's macro soup. A double-pumped/register-blocked variant is a follow-on
// brick (or template parameter) once on-device tuning wants it, not a
// per-model fork.
//
// int4 PACKING: aie::vector<int4, N> packs two 4-bit signed lanes per
// byte (aie_api's native sub-byte vector packing, low nibble = even index,
// high nibble = odd index, standard aie_api convention -- see golden.py for
// the exact host-side pack/unpack this brick's numpy reference uses; the
// later on-device pass must feed the SAME packing convention). A holds int8
// (1 byte/elem, no packing).
//
// Build (CPU-only, no device): plain -O2, -I mlir-aie/third_party/aie_api/
// include, no extra -D's needed (this file is self-contained, same class as
// affine_cast.cc / gemm-int8).

#include <aie_api/aie.hpp>
#include <stdint.h>

// One 4x16x16 int8xint4->int32 systolic-tile matmul, single MMUL instance,
// A/B/C all held resident (no host round-trip) -- the atomic op this brick
// tiles across for M/K/N > the native 4x16x16 shape.
//
// pB is a packed int4 buffer: each K x N int4 tile of size MMUL::size_B
// (=K*N=256 int4 lanes) occupies size_B/2 = 128 bytes, addressed via
// aie::vector<int4, size_B> / aie::load_v<size_B> exactly like any other
// aie_api sub-byte vector (the packing/addressing is handled by the vector
// type itself, not by this kernel).
template <unsigned M, unsigned K, unsigned N>
static inline void gemm_int8xint4_tile(const int8_t *__restrict pA,
                                       const int4 *__restrict pB,
                                       int32_t *__restrict pC) {
  static_assert(M % 4 == 0,
                "gemm_int8xint4: M must be a multiple of 4 (mmul_8_4 tile)");
  static_assert(K % 16 == 0,
                "gemm_int8xint4: K must be a multiple of 16 (mmul_8_4 tile)");
  static_assert(N % 16 == 0,
                "gemm_int8xint4: N must be a multiple of 16 (mmul_8_4 tile)");

  using MMUL = aie::mmul<4, 16, 16, int8_t, int4, accauto>;

  constexpr unsigned kSteps = K / 16;
  constexpr unsigned mTiles = M / 4;
  constexpr unsigned nTiles = N / 16;

  for (unsigned mi = 0; mi < mTiles; ++mi) {
    for (unsigned ni = 0; ni < nTiles; ++ni) {
      MMUL acc;

      for (unsigned ki = 0; ki < kSteps; ++ki) {
        const int8_t *pA_tile = pA + (mi * kSteps + ki) * MMUL::size_A;
        const int4 *pB_tile = pB + (ki * nTiles + ni) * MMUL::size_B;

        aie::vector<int8_t, MMUL::size_A> A =
            aie::load_v<MMUL::size_A>(pA_tile);
        aie::vector<int4, MMUL::size_B> B =
            aie::load_v<MMUL::size_B>(pB_tile);

        // The mmul_8_4 detail specialization's mac/mul take explicit runtime
        // sign bools (mac(a, a_sign, b, b_sign)) -- but the PUBLIC aie::mmul
        // wrapper (aie.hpp) exposes a plain 2-arg mac(a,b)/mul(a,b) that
        // infers each operand's sign from its C++ vector value_type via
        // detail::get_mul_sign(), then forwards to the 4-arg detail call
        // itself. Since A/B are plain (non-op) vectors of int8_t/int4
        // (both signed), that inferred sign is always true/true here -- so
        // this brick just calls the ordinary 2-arg form, same call shape as
        // gemm-int8's mmul_8_8 path.
        if (ki == 0) {
          acc.mul(A, B);
        } else {
          acc.mac(A, B);
        }
      }

      int32_t *pC_tile = pC + (mi * nTiles + ni) * MMUL::size_C;
      aie::store_v(pC_tile, acc.template to_vector<int32_t>());
    }
  }
}

extern "C" {

// Fixed instantiations for the sizes the golden/verify harness exercises
// (grow this vocabulary per-model instance, per the earn-from-instance rule
// -- do NOT template-explode every M/K/N up front). All shapes are
// multiples of the native 4x16x16 mmul_8_4 tile.
void gemm_int8xint4_8x16x16(const int8_t *__restrict a,
                            const int4 *__restrict b,
                            int32_t *__restrict c) {
  gemm_int8xint4_tile<8, 16, 16>(a, b, c);
}

void gemm_int8xint4_64x64x64(const int8_t *__restrict a,
                             const int4 *__restrict b,
                             int32_t *__restrict c) {
  gemm_int8xint4_tile<64, 64, 64>(a, b, c);
}

void gemm_int8xint4_128x128x128(const int8_t *__restrict a,
                                const int4 *__restrict b,
                                int32_t *__restrict c) {
  gemm_int8xint4_tile<128, 128, 128>(a, b, c);
}

} // extern "C"
