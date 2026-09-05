//===- residual_add.cc ----------------------------------------*- C++ -*-===//
//
// Device-side f32 scaled residual add (whole-block-resident fusion). Keeps the block
// residual `x = a + scale*b` (Macaron FFN: a=x, b=ff, scale=0.5; full residual: scale=1.0)
// ON-CHIP so the residual never round-trips to host. out[t,c] = a[t,c] + scale*b[t,c], f32.
//
// `scale` is BAKED at IRON-generation time (passed as a compile-time literal from core_body,
// like `cols`): an AIE2 tile has only 2 input DMA channels, both consumed by the row-tiled a
// and b, so there is no channel left for a runtime scale param -> one xclbin per scale value
// (s050 = 0.5, s100 = 1.0). f32 mul-by-0.5/1.0 + one add is near-exact (rel-L2 ~0 vs host).
//
// 2-input ABI: a (g3), b (g4), out (g5). One call: ONE row of `cols`. cols % N == 0.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// Three wire formats, exactly ONE compiled per build (not three entry points) so the L1 figure a
// build reports is its own.
//   default        a/b/out f32
//   RESADD_BF16    a/b/out bf16 -- six streaming buffers halve, 24588 B -> 12300 B of the 64 KB core
//   RESADD_B_BF16  a/out f32, b bf16
//
// B_BF16 is the arm a bf16-out modal GEMM needs: only the sub-layer addend comes off that drain, and
// the residual stream a/out is read as f32 by every other brick on the seam, so narrowing it too
// would move the defect rather than fix it. Widening b is exact (bf16 is a truncated f32), so the
// arithmetic is the f32 arm's with a coarser addend.
#if defined(RESADD_B_BF16)
template <int N>
void residual_add_row(const float *restrict a, const bfloat16 *restrict b,
                      float *restrict out, float scale, int32_t cols) {
  event0();
  const auto saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const ::aie::vector<float, N> sv = ::aie::broadcast<float, N>(scale);
  for (int i = 0; i < cols; i += N) {
    ::aie::accum<accfloat, N> ba;
    ba.from_vector(::aie::load_v<N>(b + i), 0);
    ::aie::vector<float, N> sb = ::aie::mul(ba.template to_vector<float>(), sv);
    ::aie::store_v(out + i, ::aie::add(::aie::load_v<N>(a + i), sb));
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void residual_add_row(float *a, bfloat16 *b, float *out, float scale,
                      int32_t cols) {
  residual_add_row<16>(a, b, out, scale, cols);
}
}
#elif defined(RESADD_BF16)
template <int N>
void residual_add_row(const bfloat16 *restrict a, const bfloat16 *restrict b,
                      bfloat16 *restrict out, float scale, int32_t cols) {
  event0();
  // Same conv_even discipline as the f32 arm, and it matters more here: the narrow back to
  // bf16 is a rounding site the f32 arm does not have.
  const auto saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const ::aie::vector<float, N> sv = ::aie::broadcast<float, N>(scale);
  for (int i = 0; i < cols; i += N) {
    // bf16 -> f32 is exact (bf16 is a truncated f32), so widening costs nothing numerically
    // and the add/mul run at f32 exactly as in the f32 arm.
    ::aie::accum<accfloat, N> aa;
    aa.from_vector(::aie::load_v<N>(a + i), 0);
    ::aie::accum<accfloat, N> ba;
    ba.from_vector(::aie::load_v<N>(b + i), 0);
    ::aie::vector<float, N> sb = ::aie::mul(ba.template to_vector<float>(), sv);
    ::aie::vector<float, N> y = ::aie::add(aa.template to_vector<float>(), sb);
    ::aie::accum<accfloat, N> ya;
    ya.from_vector(y);
    ::aie::store_v(out + i, ya.template to_vector<bfloat16>());
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void residual_add_row(bfloat16 *a, bfloat16 *b, bfloat16 *out, float scale,
                      int32_t cols) {
  residual_add_row<16>(a, b, out, scale, cols);
}
}
#else
template <int N>
void residual_add_row(const float *restrict a, const float *restrict b,
                      float *restrict out, float scale, int32_t cols) {
  event0();
  // Round-to-nearest-even so the on-chip f32 add matches the host round-nearest add bit-for-bit
  // (default AIE rounding is truncation -> 1-ULP drift that accumulates over blocks; the WER-path
  // rounding rule, same as glu.cc / affine_cast.cc).
  // crRnd is a single sticky register shared by every kernel that runs on this core, so the mode
  // has to be handed back: leaving it set makes the NEXT kernel's narrowing depend on whether this
  // one happened to run first, which is invisible in every artifact we diff.
  const auto saved_rounding =
      ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  const ::aie::vector<float, N> sv = ::aie::broadcast<float, N>(scale);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> av = ::aie::load_v<N>(a + i);
    ::aie::vector<float, N> bv = ::aie::load_v<N>(b + i);
    ::aie::vector<float, N> sb = ::aie::mul(bv, sv);  // scale*b
    ::aie::store_v(out + i, ::aie::add(av, sb));      // a + scale*b
  }
  ::aie::set_rounding(saved_rounding);
  event1();
}

extern "C" {
void residual_add_row(float *a, float *b, float *out, float scale, int32_t cols) {
  residual_add_row<16>(a, b, out, scale, cols);
}
}
#endif  // wire-format arm
