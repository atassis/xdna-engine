// gemm-bfp16-ebs8 brick [group: matmul]
//
// Generic M>=8 GEMM op-TYPE for the resident [tile,D] stream contract:
// C[M,N] (bf16, fp32-accumulated) = A[M,K] (bf16) @ B[K,N] (bf16), tiled over
// the AIE2P TRUE-SYSTOLIC aie::mmul<8,8,8,bfp16ebs8,bfp16ebs8,accauto> tile.
// This is the native block-floating-point datapath format (NOT the
// bfloat16-emulated-via-bfp16 path) -- a PARAMETERIZED primitive (M/K/N are
// template args resolved per dispatch by the schedule, not per-model
// hand-fusion) -- brick catalog entry, matmul group, compute-bound M>=8
// regime (see docs/aie2p-brick-catalog.md).
//
// API pattern studied from:
//   - route_b_kernels/ffn_bfp16/repro_847_mmul888_bfp16.cc (mmul<M,K,N,TA,TB,
//     accauto> -> load_v -> acc.mul/mac -> acc.to_vector<Out>() skeleton)
//   - route_b_kernels/bricks/gemm-int8/gemm_int8.cc (this brick's tiling
//     shape/style, generalized from int8 to the bfp16 block-float format)
//   - mlir-aie/third_party/aie_api/include/aie_api/detail/aie2p/
//     mmul_bfp16_bfp16.hpp: the REAL `mmul<8,8,8,bfp16ebs8,bfp16ebs8,32>`
//     specialization is a native systolic op -- it does NOT route through
//     the bfloat16-emulation path that triggers the OPEN llvm-aie #847 ICE
//     (that bug is specific to `aie::mmul<8,8,8,bfloat16,bfloat16,accauto>`
//     under -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16). Using the bfp16ebs8
//     type directly, as this brick does, is a DIFFERENT (working) codepath.
//     NOTE: the public `aie::mmul<...>::mul/mac(a,b)` wrapper's concept
//     constraints (VectorOrOp / BlockVectorOp) only accept a plain
//     block_vector,block_vector pair when B carries an `op_transpose` --
//     i.e. that wrapper models "A * transpose(B)", not a bare "A*B" on two
//     already-block-typed operands. AMD's OWN reference kernel for this
//     exact tile (mlir-aie/aie_kernels/aie2p/mm_bfp.cc,
//     matmul_vectorized_2x2_bfp16) therefore bypasses the class wrapper
//     entirely and calls the underlying free intrinsics directly:
//     `mul_8x8_8x8T(A,B)` / `mac_8x8_8x8T(A,B,acc)` on
//     `aie::block_vector<bfp16ebs8,64>` operands and an
//     `aie::accum<accfloat,64>` accumulator. This brick follows that SAME
//     proven, in-tree pattern (not the mmul-class wrapper) for exactly that
//     reason.
//   - the mixed `mmul_bfp16_bfp16<8,8,8,bfloat16,bfp16ebs8,32>` specialization
//     documents the on-chip bf16 -> bfp16ebs8 conversion idiom this brick
//     reuses for BOTH operands (the resident stream contract is bf16 in/out;
//     the true-systolic tile itself is bfp16ebs8 x bfp16ebs8): build an
//     `aie::accum<accfloat,64>` from the resident bf16 vector, then
//     `::to_v64bfp16ebs8(acc)` to obtain the native `block_vector<bfp16ebs8,
//     64>` the true-systolic mmul consumes.
//
// Rounding: bfp16ebs8 quantization (block shared-exponent + 8-bit signed
// mantissa) must round-to-nearest-even, matching the host `golden.py`
// reference and the project's banked bf16-cast convention ("WER-path bf16
// casts MUST set_rounding(conv_even)"). Set once per kernel invocation before
// any bf16->bfp16ebs8 conversion.
//
// Tiling: single AIE core, one row-tile x col-tile of C at a time, walking K
// in 8-wide steps, fp32-accumulating in the MMUL's native accauto accumulator
// before a single to_vector<bfloat16>() writeback per output tile. M,K,N
// must each be a multiple of 8 (the native aie::mmul<8,8,8,...> tile shape);
// this brick does not itself decide the outer schedule (that is
// schedule/IRON-side, generic-rails "model = schedule over bricks").
//
// Build (CPU-only, no device): plain -O2, -I mlir-aie/third_party/aie_api/
// include, no extra -D's needed (self-contained: aie_api/aie.hpp + stdint.h
// only, same class as affine_cast.cc / gemm_int8.cc).

#include <aie_api/aie.hpp>
#include <stdint.h>

// Convert one resident bf16 64-lane vector to the native bfp16ebs8 block
// format (shared 8-bit exponent per 8-element block, 8-bit signed mantissa),
// via the fp32 accumulator round-trip the aie_api mixed-type mmul path uses.
static inline aie::block_vector<bfp16ebs8, 64>
to_bfp16ebs8(const aie::vector<bfloat16, 64> &v) {
  aie::accum<accfloat, 64> acc(v);
  return ::to_v64bfp16ebs8(acc);
}

// One 8x8x8 bfp16ebs8 x bfp16ebs8 -> fp32(accauto) true-systolic-tile matmul,
// single MMUL instance, A/B/C all held resident (no host round-trip) -- the
// atomic op this brick tiles across for M/K/N > 8.
template <unsigned M, unsigned K, unsigned N>
static inline void gemm_bfp16_ebs8_tile(const bfloat16 *__restrict pA,
                                        const bfloat16 *__restrict pB,
                                        bfloat16 *__restrict pC) {
  static_assert(M % 8 == 0, "gemm-bfp16-ebs8: M must be a multiple of 8 (mmul tile)");
  static_assert(K % 8 == 0, "gemm-bfp16-ebs8: K must be a multiple of 8 (mmul tile)");
  static_assert(N % 8 == 0, "gemm-bfp16-ebs8: N must be a multiple of 8 (mmul tile)");

  // Block-floating-point quantization must round-to-nearest-even (banked
  // convention for every bf16/bfp16 cast on the WER-gated path).
  aie::set_rounding(aie::rounding_mode::conv_even);

  // Native 8x8x8 tile sizes for the bfp16ebs8 true-systolic op (block_vector
  // lane count, not element byte count): size_A = size_B = size_C = 64.
  constexpr unsigned kTileElems = 64;
  constexpr unsigned kSteps = K / 8;

  for (unsigned mi = 0; mi < M / 8; ++mi) {
    for (unsigned ni = 0; ni < N / 8; ++ni) {
      aie::accum<accfloat, kTileElems> acc = aie::zeros<accfloat, kTileElems>();

      for (unsigned ki = 0; ki < kSteps; ++ki) {
        const bfloat16 *pA_tile = pA + (mi * kSteps + ki) * kTileElems;
        const bfloat16 *pB_tile = pB + (ki * (N / 8) + ni) * kTileElems;

        aie::vector<bfloat16, kTileElems> A_bf16 =
            aie::load_v<kTileElems>(pA_tile);
        aie::vector<bfloat16, kTileElems> B_bf16 =
            aie::load_v<kTileElems>(pB_tile);

        aie::block_vector<bfp16ebs8, kTileElems> A = to_bfp16ebs8(A_bf16);
        aie::block_vector<bfp16ebs8, kTileElems> B = to_bfp16ebs8(B_bf16);

        // Zero-initialized accumulator makes mac == mul for ki == 0, so a
        // single mac_8x8_8x8T covers every K-step (matches AMD's own
        // mm_bfp.cc reference kernel's accumulate-from-accum idiom).
        acc = ::mac_8x8_8x8T(A, B, acc);
      }

      bfloat16 *pC_tile = pC + (mi * (N / 8) + ni) * kTileElems;
      aie::store_v(pC_tile, acc.template to_vector<bfloat16>());
    }
  }
}

extern "C" {

// Fixed instantiations for the sizes the golden/verify harness exercises
// (grow this vocabulary per-model instance, per the earn-from-instance rule
// -- do NOT template-explode every M/K/N up front).
void gemm_bfp16_ebs8_8x8x8(const bfloat16 *__restrict a,
                            const bfloat16 *__restrict b,
                            bfloat16 *__restrict c) {
  gemm_bfp16_ebs8_tile<8, 8, 8>(a, b, c);
}

void gemm_bfp16_ebs8_64x64x64(const bfloat16 *__restrict a,
                               const bfloat16 *__restrict b,
                               bfloat16 *__restrict c) {
  gemm_bfp16_ebs8_tile<64, 64, 64>(a, b, c);
}

void gemm_bfp16_ebs8_128x128x128(const bfloat16 *__restrict a,
                                  const bfloat16 *__restrict b,
                                  bfloat16 *__restrict c) {
  gemm_bfp16_ebs8_tile<128, 128, 128>(a, b, c);
}

} // extern "C"
