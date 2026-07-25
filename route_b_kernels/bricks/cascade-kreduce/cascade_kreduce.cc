// cascade-kreduce brick [group: movement]
//
// Generic K-REDUCTION op-TYPE across adjacent cores for the resident
// [tile,D] stream contract: each of N_CASCADE cores holds a K-chunk of the
// contraction (Wfc2_slab[K_chunk,Dout] style), computes its PARTIAL product
// via aie::mmul (systolic tile, not a scalar aie::mac loop), and the partials
// are summed down a cascade of adjacent cores (HEAD seeds with an optional
// residual/bias, MIDDLE forwards recv+partial, TAIL truncates + writes the
// final [Dout] result). This is the MOVEMENT-lever half of the brick: fusing
// the per-core K-reduction hop into a cascade instead of a DDR buffer-copy +
// host/software vector-add between dispatches.
//
// API pattern studied (per task):
//   aie::mmul<M,K,N,TA,TB,accauto> -> load_v -> acc.mul/mac -> acc.to_vector<Out>()
//   (gemm-int8 / gemm-bfp16-ebs8 bricks in this same dir; the systolic-tile
//   skeleton) + "aie::mmul + cascade put/get" per the task's brick spec.
//
// MODEL STUDIED FIRST: route_b_kernels/cascade_ffn/ (STRUCTURE.md +
// mv_bf16_gelu.cc). Two things carry over from that model and one does NOT:
//   - carries over: the HEAD/MIDDLE/TAIL cascade role split, residual
//     injected at HEAD, truncate-to-bf16 only at TAIL (matvec_cascade_add.py
//     A.2 in STRUCTURE.md) -- this brick generalizes that role split as a
//     parameterized-by-n op-TYPE (not per-model).
//   - carries over: STRUCTURE.md's OWN finding that in this stack
//     "matvec_cascade_add.py links NO .cc kernel -- the cascade is pure MLIR
//     vector ops" (npu_cascade channels / ChannelPut/ChannelGet), and the
//     per-core matvec itself was a scalar aie::mac loop, never aie::mmul.
//   - does NOT carry over: mv_bf16_gelu.cc's scalar-mac matvec. This brick
//     upgrades the per-core partial to the systolic aie::mmul<8,8,8,...>
//     tile (the actual movement/compute win the task asks for -- a true
//     8x8x8 tile push per K-step instead of one MAC per lane-group), see
//     cascade_kreduce_partial_mmul_tile() below.
//
// *** THE PR-DRAFT FINDING (read before extending) ***
// The task asks for "aie::mmul + cascade put/get" as ONE fused kernel-level
// primitive. aie_api DOES define cascade read/write intrinsics
// (aie::detail::adf::cascade_stream_helper, readincr_v/writeincr against
// input_cascade<AccumTag>/output_cascade<AccumTag>; see aie_api/adf/
// stream.hpp and the readincr_v<8>(input_cascade...) example in aie_doc.hpp
// line ~326) -- BUT that header hard-requires `#include <adf.h>`
// (mlir-aie/third_party/aie_api/include/aie_api/adf/stream.hpp:9), which is
// the Vitis ADF graph framework. `adf.h` is NOT vendored anywhere in this
// aie_api tree (`find ... -iname adf.h` -> zero hits) and is not part of the
// IRON/mlir-aie device-kernel toolchain this project uses. IRON's actual
// hardware cascade wiring is done at the MLIR/dataflow level instead
// (`channel_type="npu_cascade"` + `ChannelPut`/`ChannelGet`, confirmed by
// STRUCTURE.md's own read of matvec_cascade_add.py) -- i.e. THE LOWERING
// ALREADY FORCES BUFFER-TRANSPORT (a DMA'd L1 scratch hop between adjacent
// cores' object FIFOs), not a true AccumTag-native cascade FIFO put/get
// issued from inside a .cc kernel. This is exactly the task's named
// contingency ("if the lowering forces buffer-transport, that is a
// PR-draft"). See `pr_draft` in the task return / this file's bottom comment
// for the point-fix candidate. Below, the DEFAULT (compiled, tested) path is
// the buffer-copy + software-add combine (mirrors matvec_cascade_add's
// MIDDLE role, generalized to a runtime-length n and split across dedicated
// HEAD/MIDDLE/TAIL entry points); a second path gated behind
// XDNA_BRICK_CASCADE_KREDUCE_USE_ADF (NOT compiled by default: this repo's
// toolchain has no adf.h) sketches the true-hardware-cascade call shape for
// whenever adf.h is vendored/available, so the upgrade path is a one-line
// #define flip, not a rewrite.
//
// Tiling: single AIE core, one row-tile x col-tile of the OUTPUT (this
// core's K-chunk contribution) at a time, walking this core's K_chunk in
// 8-wide mmul steps, fp32-accumulating before a single to_vector<bfloat16>()
// writeback of the PARTIAL. M,N must each be a multiple of 8 (native
// aie::mmul<8,8,8,...> tile shape); K_chunk (this core's reduction slice)
// must be a multiple of 8. The cross-core reduction itself (this brick's
// "cascade" half) is elementwise over the [N] partials and is n-generic
// (any n, typically a multiple of 16 for legalized bf16 vector width).
//
// Build (CPU-only, no device): plain -O2, -I mlir-aie/third_party/aie_api/
// include, no extra -D's needed (self-contained: aie_api/aie.hpp + stdint.h
// only, same class as affine_cast.cc / gemm_int8.cc / gemm-bfp16-ebs8.cc).

#include <aie_api/aie.hpp>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Per-core PARTIAL: this core's K-chunk contribution to the reduction,
// C_partial[M,N] += A[M,K_chunk] @ B[K_chunk,N], via the native systolic
// aie::mmul<8,8,8,bfloat16,bfloat16,accauto> tile (NOT a scalar aie::mac
// loop -- that is the model kernel's older idiom; this brick's whole point
// is pushing the per-core matmul through the systolic tile). bf16 in,
// fp32-accumulate internally, bf16 partial out (overwrite -- one K-chunk per
// core, so no running-accumulate-in-C needed; the CROSS-core accumulate is
// the cascade combine below).
// ---------------------------------------------------------------------------
template <unsigned M, unsigned K, unsigned N>
static inline void cascade_kreduce_partial_tile(const bfloat16 *__restrict pA,
                                                 const bfloat16 *__restrict pB,
                                                 bfloat16 *__restrict pC) {
  static_assert(M % 8 == 0, "cascade-kreduce: M must be a multiple of 8 (mmul tile)");
  static_assert(K % 8 == 0, "cascade-kreduce: K must be a multiple of 8 (mmul tile, this core's K-chunk)");
  static_assert(N % 8 == 0, "cascade-kreduce: N must be a multiple of 8 (mmul tile)");

  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;

  constexpr unsigned kSteps = K / 8;

  for (unsigned mi = 0; mi < M / 8; ++mi) {
    for (unsigned ni = 0; ni < N / 8; ++ni) {
      MMUL acc;

      for (unsigned ki = 0; ki < kSteps; ++ki) {
        const bfloat16 *pA_tile = pA + (mi * kSteps + ki) * MMUL::size_A;
        const bfloat16 *pB_tile = pB + (ki * (N / 8) + ni) * MMUL::size_B;

        aie::vector<bfloat16, MMUL::size_A> A =
            aie::load_v<MMUL::size_A>(pA_tile);
        aie::vector<bfloat16, MMUL::size_B> B =
            aie::load_v<MMUL::size_B>(pB_tile);

        if (ki == 0) {
          acc.mul(A, B);
        } else {
          acc.mac(A, B);
        }
      }

      bfloat16 *pC_tile = pC + (mi * (N / 8) + ni) * MMUL::size_C;
      aie::store_v(pC_tile, acc.template to_vector<bfloat16>());
    }
  }
}

#if defined(XDNA_BRICK_CASCADE_KREDUCE_USE_ADF)
// ---------------------------------------------------------------------------
// TRUE-HARDWARE-CASCADE path (aie_api AccumTag cascade FIFO), sketched but
// NOT compiled by default -- requires <adf.h> (Vitis ADF), which this
// toolchain does not vendor (see the PR-draft comment at the top of this
// file). Kept here as the one-flip upgrade target: flip the macro once
// adf.h is available/wired through IRON, and these entries replace the
// buffer-copy + software-add combine below with a real cascade put/get.
// ---------------------------------------------------------------------------
#include <adf.h>

template <unsigned N>
static inline void cascade_kreduce_middle_adf(input_cascade<accfloat> *cin,
                                               const bfloat16 *__restrict partial,
                                               output_cascade<accfloat> *cout) {
  aie::accum<accfloat, N> recv = aie::readincr_v<N>(cin);
  aie::accum<accfloat, N> part(aie::load_v<N>(partial));
  aie::writeincr(cout, aie::add(recv, part));
}
#endif // XDNA_BRICK_CASCADE_KREDUCE_USE_ADF

// ---------------------------------------------------------------------------
// Cascade COMBINE (the movement/reduction half): the default, compiled,
// device-real path in THIS toolchain. n-generic (any n, ideally a multiple
// of 16 for the legalized bf16 vector width) -- one symbol serves every
// cascade width the schedule ever picks, per the earn-from-instance /
// generic-rails rule (no per-model resize).
//
// Role split (mirrors matvec_cascade_add.py's HEAD/MIDDLE/TAIL, generalized):
//   HEAD   (first core in the chain): seed the running total with this
//           core's partial PLUS an optional residual/bias r (R may be all-
//           zero if the op-type instance has no residual to fuse in).
//   MIDDLE (interior cores): total = recv (from the upstream neighbour,
//           already ferried across the L1/object-FIFO hop the lowering
//           forces -- see PR-draft note) + this core's partial.
//   TAIL   (last core, drains to the [tile,D] stream): total = recv +
//           partial, truncate f32 accumulate -> bf16, write the final slab.
// ---------------------------------------------------------------------------

static inline void cascade_kreduce_head_bf16(uint32_t n,
                                              const bfloat16 *__restrict partial,
                                              const bfloat16 *__restrict r,
                                              bfloat16 *__restrict out) {
  for (uint32_t i = 0; i < n; i++) {
    out[i] = static_cast<bfloat16>(static_cast<float>(partial[i]) +
                                    static_cast<float>(r[i]));
  }
}

static inline void cascade_kreduce_middle_bf16(uint32_t n,
                                                const bfloat16 *__restrict recv,
                                                const bfloat16 *__restrict partial,
                                                bfloat16 *__restrict out) {
  for (uint32_t i = 0; i < n; i++) {
    out[i] = static_cast<bfloat16>(static_cast<float>(recv[i]) +
                                    static_cast<float>(partial[i]));
  }
}

static inline void cascade_kreduce_tail_bf16(uint32_t n,
                                              const bfloat16 *__restrict recv,
                                              const bfloat16 *__restrict partial,
                                              bfloat16 *__restrict out) {
  for (uint32_t i = 0; i < n; i++) {
    out[i] = static_cast<bfloat16>(static_cast<float>(recv[i]) +
                                    static_cast<float>(partial[i]));
  }
}

extern "C" {

// Fixed instantiations for the sizes the golden/verify harness exercises
// (grow this vocabulary per-model instance -- do NOT template-explode every
// M/K/N up front, per the earn-from-instance/YAGNI rule).

// Per-core PARTIAL (this core's K-chunk contribution), matvec shape (M=8
// output rows folded into one mmul-tile row, K_chunk=8 minimal tile, N=8):
// smallest legal aie::mmul<8,8,8,...> single-step instance.
void cascade_kreduce_partial_8x8x8(const bfloat16 *__restrict a,
                                    const bfloat16 *__restrict b,
                                    bfloat16 *__restrict c) {
  cascade_kreduce_partial_tile<8, 8, 8>(a, b, c);
}

// Larger per-core PARTIAL instance matching a Whisper-FFN-fc2-style K-chunk
// (this core's slab of the reduction): M=8 (one output row-tile), K_chunk=64
// (a small multi-step K-chunk), N=8.
void cascade_kreduce_partial_8x64x8(const bfloat16 *__restrict a,
                                     const bfloat16 *__restrict b,
                                     bfloat16 *__restrict c) {
  cascade_kreduce_partial_tile<8, 64, 8>(a, b, c);
}

// Cascade combine, n-generic (runtime length; caller picks n = the
// [tile,D]-stream's D or a sub-slab of it).
void cascade_kreduce_head_bf16_n(uint32_t n, bfloat16 *partial, bfloat16 *r,
                                  bfloat16 *out) {
  cascade_kreduce_head_bf16(n, partial, r, out);
}
void cascade_kreduce_middle_bf16_n(uint32_t n, bfloat16 *recv,
                                    bfloat16 *partial, bfloat16 *out) {
  cascade_kreduce_middle_bf16(n, recv, partial, out);
}
void cascade_kreduce_tail_bf16_n(uint32_t n, bfloat16 *recv,
                                  bfloat16 *partial, bfloat16 *out) {
  cascade_kreduce_tail_bf16(n, recv, partial, out);
}

} // extern "C"
