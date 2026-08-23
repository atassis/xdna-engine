//===- mm_mode_lnaffcast.cc -------------------------------------*- C++ -*-===//
//
// lnaffcast as a MODE of the modal GEMM xclbin, so the op runs inside the
// resident's hardware context instead of behind a program boundary. The prize
// is the boundary, not the op: 90% of the merge's -238.0 ms/clip sits on the 96
// `lnaffcast -> modal` transitions, not on lnaffcast itself.
//
// It declares no buffer. The bf16-out resident already owns a per-core
// `cacc` of 32x128xf32 = 16384 B, which is EXACTLY the four 1024-column rows a
// C tile holds, so the staging this needs is the accumulator the GEMM was
// going to zero anyway. One x row is two A tiles, gb is exactly one B tile.
//
// NO INDEX MAP. Every fifo shuffle is undone by the mode's own shim taps, all
// three at exactly 4 dims against the shim's 4 (A [32,8,4,4]/[128,4,32,1], B
// [4,8,16,4]/[512,4,32,1], C [4,8,16,8]/[1024,8,64,1]), so the operands arrive
// in column order and every access below is contiguous at 16 lanes. Do not
// "fix" an ordering here -- if a tap is wrong the fix is the tap, and the A one
// is 4 dims only because its outer slab dimension merges with the leading digit
// (512 == 4 x 128). See lnaffcast-mode-taps-fit-the-shim-in-four-dims.
//
// Self-contained because AIEAssignCoreLinkFiles traces only direct func.call
// edges from the core, so an object this one CALLED would never reach the link
// line. Same shape mm_mode_body.cc uses.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// The lnaffcast embedding dim. A build parameter, not a property of the tile:
// the tile dims fix how many rows a C tile carries, this fixes how long one is.
#ifndef LNA_COLS
#define LNA_COLS 1024
#endif

// f32 elements in one A tile, and output rows in one C tile. Derived from the
// GEMM's own declared shapes so they cannot drift from the buffers they index.
static constexpr int kSlabF32 = EPI_M * EPI_K / 2;
static constexpr int kRowsPerC = EPI_M * EPI_N / LNA_COLS;

static_assert(EPI_M * EPI_N == kRowsPerC * LNA_COLS,
              "C tile must hold a whole number of output rows");
static_assert(LNA_COLS % kSlabF32 == 0,
              "an x row must be a whole number of A tiles");
static_assert(LNA_COLS % 16 == 0, "the write pass walks 16 lanes at a time");

// Stage one A tile into the accumulator. The A tile is DECLARED bf16 (it is the
// GEMM's operand buffer) but a mode stream fills it with f32 activations, so it
// is read through a reinterpret -- the same borrow the accadd mode already
// makes. Both buffers are objectFIFO/aie.buffer allocations, which are at least
// 32-byte aligned, so the f32 view is aligned by construction.
template <int N>
static void lnaff_stage(const bfloat16 *restrict a, float *restrict acc) {
  event0();
  const float *src = reinterpret_cast<const float *>(a);
  for (int i = 0; i < kSlabF32; i += N)
    ::aie::store_v(acc + i, ::aie::load_v<N>(src + i));
  event1();
}

// LN + affine + f32->bf16 cast over the whole staged C tile.
//
// The math is ln_affine_cast.cc's, unchanged and deliberately so: this has to
// be bit-comparable to the op it replaces or the merge cannot be gated on
// parity. That includes the rounding split -- the reductions run under the AIE
// DEFAULT mode (ctxLN never sets one) and conv_even is swapped in ONLY around
// the affine+cast write. Setting conv_even up front changes the reduction
// rounding and regressed WER 8.2 -> 8.8, compounded over 24 layers.
template <int N>
static void lnaff_apply(const bfloat16 *restrict gb, float *restrict acc,
                        bfloat16 *restrict out) {
  event0();
  constexpr float epsilon = 1e-5f;
  constexpr int chunks = LNA_COLS / N;
  // gb is [gamma | beta], one B tile, f32 like the activations above.
  const float *gbf = reinterpret_cast<const float *>(gb);
  const float *gamma = gbf;
  const float *beta = gbf + LNA_COLS;

  for (int row = 0; row < kRowsPerC; row++) {
    const float *x = acc + row * LNA_COLS;
    bfloat16 *y = out + row * LNA_COLS;

    // pass 1: mean = Sx / cols
    ::aie::vector<float, N> sum_v = ::aie::zeros<float, N>();
    for (int i = 0; i < chunks; i++)
      sum_v = ::aie::add(sum_v, ::aie::load_v<N>(x + i * N));
    float mean = ::aie::reduce_add(sum_v) / float(LNA_COLS);
    ::aie::vector<float, N> mean_v = ::aie::broadcast<float, N>(mean);

    // pass 2: var = S(x-mean)^2 / cols (centered, matching the host reference)
    ::aie::vector<float, N> var_v = ::aie::zeros<float, N>();
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(x + i * N), mean_v);
      // The temporary is load-bearing, not style: aie::mul yields an accum, and
      // naming it as a vector is what narrows it back. Inlining the call makes
      // add() see an accum and fail to deduce. Same spelling as ln_affine_cast.cc.
      ::aie::vector<float, N> sq = ::aie::mul(d, d);
      var_v = ::aie::add(var_v, sq);
    }
    float var = ::aie::reduce_add(var_v) / float(LNA_COLS);
    ::aie::vector<float, N> inv_v =
        ::aie::broadcast<float, N>(::aie::invsqrt(var + epsilon));

    // write: out = ((x - mean) * inv) * gamma + beta -> bf16.
    // The core's rounding mode is one sticky register shared by every kernel on
    // the core, so it is saved and restored rather than left swapped -- here
    // that matters more than in a standalone kernel, because the GEMM epilogue
    // runs on this same core from a different dispatch.
    const auto saved_rounding =
        ::aie::swap_rounding(::aie::rounding_mode::conv_even);
    for (int i = 0; i < chunks; i++) {
      ::aie::vector<float, N> d = ::aie::sub(::aie::load_v<N>(x + i * N), mean_v);
      ::aie::vector<float, N> norm = ::aie::mul(d, inv_v);
      ::aie::vector<float, N> ng =
          ::aie::mul(norm, ::aie::load_v<N>(gamma + i * N));
      ::aie::accum<accfloat, N> acc_y;
      acc_y.from_vector(::aie::add(ng, ::aie::load_v<N>(beta + i * N)));
      ::aie::store_v(y + i * N, acc_y.template to_vector<bfloat16>());
    }
    ::aie::set_rounding(saved_rounding);
  }
  event1();
}

extern "C" {
// One A tile -> the accumulator at `slot`. Called once per A tile; the slot is
// a compile-time constant from an unrolled generator loop, so the mode needs no
// counter and no loop-index cast.
void mm_lnaff_stage_f32(const bfloat16 *__restrict a, float *__restrict acc,
                        int32_t slot) {
  lnaff_stage<16>(a, acc + slot * kSlabF32);
}

// gb + the staged accumulator -> the bf16 C tile.
void mm_lnaff_apply_f32(const bfloat16 *__restrict gb, float *__restrict acc,
                        bfloat16 *__restrict out) {
  lnaff_apply<16>(gb, acc, out);
}
}
