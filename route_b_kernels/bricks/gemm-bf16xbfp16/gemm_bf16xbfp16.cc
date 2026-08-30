//===- gemm_bf16xbfp16.cc -----------------------------------------*- C++ -*-===//
//
// BRICK: gemm-bf16xbfp16  [group: matmul]
//
// Mixed-precision GEMM: bf16 ACTIVATION x bfp16ebs8 (weight-compressed) WEIGHT,
// aie::mmul<8,8,8,bfloat16,bfp16ebs8,accauto> -> accumulate in accfloat -> bf16 out.
//
// This is weight-COMPRESSION only: the A (activation) operand stays plain
// bfloat16 all the way to the mmul call (the mmul's own C_block internally
// funnels A through an accfloat->bfp16ebs8 conversion on-core to feed the
// systolic array -- that is an INTERNAL hardware requirement of the mixed
// mmul_bfp16_bfp16<bfloat16,bfp16ebs8> specialization, not a host-visible
// activation quantization step). The B (weight) operand is pre-quantized
// OFF-CORE (host-side, once, at load time) to bfp16ebs8 and DMA'd in packed --
// this is the "weight-compression, no act quant" brick: half the weight
// bytes moved per tile vs bf16xbf16, activations stay full bf16 precision
// end-to-end from the caller's point of view.
//
// Generic op-TYPE against the resident [tile,D] stream contract: this is a
// parameterized GEMM primitive (M,K,N tile dims are compile-time macros),
// NOT a per-model/FLM hand-fused kernel. Any block-schedule (FFN fc1/fc2,
// attention Q/K/V/out proj, decode gate proj, ...) reuses this same brick by
// picking DIM_M/DIM_K/DIM_N and DMA'ing its own bf16 activations + its own
// offline-quantized bfp16ebs8 weight blob.
//
// Studied model: route_b_kernels/ffn_bfp16/repro_847_mmul888_bfp16.cc
//   (API shape: aie::mmul<M,K,N,TA,TB,accauto> -> load_v -> acc.mul/mac ->
//    acc.to_vector<Out>()) and the upstream reference mixed-type kernel
//   mlir-aie/aie_kernels/aie2p/mm_bfp_mixed.cc (matmul_vectorized_2x2_bfp16_bf16),
//   which is the *transpose-2x2-unrolled* raw-intrinsic (mac_8x8_8x8T) sibling
//   of the same aie::mmul<bfloat16,bfp16ebs8,accauto> hardware path (see
//   aie_api/detail/aie2p/mmul_bfp16_bfp16.hpp, the
//   `mmul_bfp16_bfp16<8,8,8,bfloat16,bfp16ebs8,32>` specialization). This
//   brick uses the public aie::mmul object API directly (per the task spec)
//   rather than the raw intrinsic, and is written as a single-pump (no 2x2
//   unroll) core so the generic shape stays easy to read; a 2x2-unrolled
//   perf variant can be layered on later behind the same signature.
//
// Weight (B) memory layout: pre-transposed, K-blocked bfp16ebs8, i.e. for
// output tile (m,n) the K-th 8x8 weight sub-block lives at
//   pB + (n * (K/8) + k) * (8*8)
// matching the `pBxbfp16.seek(j * colA)` transposed-B convention in
// mm_bfp_mixed.cc. This is the natural layout for a weight tensor that is
// quantized+packed ONCE offline (weight-compression brick) and then
// DMA-streamed read-only, tile by tile, every forward pass.
//
// Build (CPU-only, no device): see golden.py header for the golden and the
// verify_spec in the accompanying PR notes. Compile recipe (confirmed
// working against the atassis/mlir-aie fork's Peano/llvm-aie toolchain):
//
//   $IRON_ENV/lib/python3.12/site-packages/llvm-aie/bin/clang++ \
//     --target=aie2p-none-unknown-elf -std=c++20 \
//     -I <mlir-aie>/third_party/aie_api/include \
//     -O2 -c gemm_bf16xbfp16.cc -o gemm_bf16xbfp16.o
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>

// ---- generic templated core -------------------------------------------------
//
// R,S,T   : hardware mmul micro-tile dims (AIE2P mixed bf16xbfp16ebs8 -> 8x8x8).
// RowA    : number of R-row tiles of A   (== M_tile / R)
// ColA    : number of S-col/K tiles      (== K_tile / S), shared contraction dim
// ColB    : number of T-col tiles of B/C (== N_tile / T)
template <unsigned R, unsigned S, unsigned T_, unsigned RowA, unsigned ColA,
          unsigned ColB>
void gemm_bf16xbfp16_core(const bfloat16 *__restrict pA,
                           const bfp16ebs8 *__restrict pB,
                           bfloat16 *__restrict pC) {
  using MMUL = aie::mmul<R, S, T_, bfloat16, bfp16ebs8, accauto>;

  constexpr unsigned sizeA = R * S;
  constexpr unsigned sizeB = S * T_;
  constexpr unsigned sizeC = R * T_;

  static_assert(MMUL::size_A == sizeA, "A micro-tile size mismatch");
  static_assert(MMUL::size_B == sizeB, "B micro-tile size mismatch");
  static_assert(MMUL::size_C == sizeC, "C micro-tile size mismatch");

  // Declare the rounding we want for the accfloat -> bfloat16 narrow below.
  //
  // DEFENSIVE, NOT A BUG FIX -- and the measurement says so. The narrow was
  // flagged by the control-register sweep as "lossy op on a register nobody
  // sets", on the documented premise that the default is floor. On device that
  // premise does not hold for THIS site: forcing floor explicitly and forcing
  // conv_even explicitly produce BIT-IDENTICAL output (rel-L2 6.777424096e-03
  // either way, 64x64, four arms in _verify/verify_rounding_ab.py). Truncation
  // and round-to-nearest differ on roughly half of random mantissas, so across
  // 4096 outputs that rules out data insensitivity: this narrow does not consult
  // crrnd at all, even though `mov crrnd, #0xc` is verifiably emitted.
  //
  // Kept anyway, for one instruction per call: it states the intended rounding
  // rather than depending on a lowering detail that measured inert on one
  // toolchain and could change on the next. Do NOT cite it as having fixed a
  // real numerical bias -- it did not, and no output moved.
  //
  // Also note the register is global and sticky, so whatever a kernel leaves set
  // affects whatever runs next on that core. That is a real hazard here: the
  // brick's own verifier shim sets conv_even for its B quantization, which is
  // why an unset kernel could never be distinguished from a set one until the
  // shim stopped setting it.
  //
  // Says nothing about Xilinx/mlir-aie#3442. That is a DIFFERENT site -- the
  // A/B bf16 -> bfp16 conversion inside aie::mmul under the emulation path.
  // Here A stays bf16 and B arrives already bfp16ebs8 through the mixed
  // intrinsic, so no A/B conversion happens. This result neither supports nor
  // refutes that one.
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);

  for (unsigned m = 0; m < RowA; ++m) {
    const bfloat16 *__restrict pA_row0 = pA + m * ColA * sizeA;

    for (unsigned n = 0; n < ColB; ++n) {
      // B is pre-transposed/blocked: tile (n,k) lives at (n*ColA + k)*sizeB.
      aie::block_vector_input_buffer_stream<bfp16ebs8, MMUL::size_B> bStream(
          pB);
      bStream.seek(n * ColA);

      const bfloat16 *__restrict pA_row = pA_row0;
      MMUL acc;

      for (unsigned k = 0; k < ColA; ++k) {
        typename MMUL::vector_A_type A = aie::load_v<MMUL::size_A>(pA_row);
        pA_row += sizeA;

        typename MMUL::vector_B_type B = bStream.pop();

        // The public aie::mmul wrapper only accepts a block_vector B operand
        // when it is explicitly wrapped in aie::op_transpose(): the
        // mixed bfloat16 x bfp16ebs8 8x8x8 hardware micro-kernel is the
        // "*_8x8_8x8T" intrinsic family (see mm_bfp_mixed.cc), i.e. B is fed
        // to the systolic array pre-transposed -- exactly the block-K-major
        // weight layout documented above.
        if (k == 0)
          acc.mul(A, aie::op_transpose(B));
        else
          acc.mac(A, aie::op_transpose(B));
      }

      aie::store_v(pC + (m * ColB + n) * sizeC,
                    acc.template to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved_rounding);
}

// ---- extern "C" entry point -------------------------------------------------
//
// Tile shape is compile-time configurable via -DDIM_M/-DDIM_K/-DDIM_N
// (element counts of the GEMM tile handed to this core each call; must be
// divisible by the 8x8x8 hardware micro-tile). Defaults match the other
// bricks' 64x64x64 per-core L1 tile convention.
extern "C" {

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

void gemm_bf16xbfp16(const bfloat16 *__restrict pA,
                      const bfp16ebs8 *__restrict pB,
                      bfloat16 *__restrict pC) {
  constexpr unsigned r = 8, s = 8, t = 8;
  constexpr unsigned m = DIM_M, k = DIM_K, n = DIM_N;

  static_assert(m % r == 0, "DIM_M must be a multiple of 8");
  static_assert(k % s == 0, "DIM_K must be a multiple of 8");
  static_assert(n % t == 0, "DIM_N must be a multiple of 8");

  gemm_bf16xbfp16_core<r, s, t, m / r, k / s, n / t>(pA, pB, pC);
}

} // extern "C"
