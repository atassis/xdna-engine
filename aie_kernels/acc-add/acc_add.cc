//===- acc_add.cc ---------------------------------------------*- C++ -*-===//
//
// Device-side f32 elementwise accumulate-add (resident-FFN fc2 on-device K-split
// accumulation). out[t, c] = a[t, c] + b[t, c], per row over `cols`, all f32.
//
// The resident FFN fc2 is a K-split: DFF/KRES modal (identity) partials, each
// [PAD_M, KRES] f32. Today those partials are summed into a HOST Array2 (so the
// FFN output is host-assembled and cannot stay device-resident). This brick sums
// the running accumulator (a) with the next partial (b) into a third BO (out), so
// the fc2 output lands in ONE device BO -- deleting the host accumulation. f32 add
// is DETERMINISTIC (not exact -- it rounds), so seeding acc=0 and adding the partials
// IN THE SAME ORDER as the host is bit-identical to the host sequential f32 K-split
// (WER-neutral). The ORDER is what licenses that, not exactness -- a tree or parallel
// reduction over the same partials is not a safe substitute.
//
// 2-input ABI: a (g3), b (g4), out (g5) -- an AIE2 tile has only 2 input DMA
// channels, so a + b sits exactly at the limit (mirrors affine_cast's in + gb).
// One call: ONE row of `cols` (the per-row core_body contract). cols % N == 0.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// ACCADD_B_BF16: the partial (b) arrives on a bf16-out GEMM drain while the running accumulator
// (a/out) stays f32. Widening b is exact, but the bit-identity above does NOT carry over -- the
// partials are already rounded when they reach the core, so this arm differs from the host K-split
// by the drain's rounding, not by accumulation order. Exactly one arm compiles per build.
#ifdef ACCADD_B_BF16
template <int N>
void acc_add_row(const float *restrict a, const bfloat16 *restrict b,
                 float *restrict out, int32_t cols) {
  event0();
  for (int i = 0; i < cols; i += N) {
    ::aie::accum<accfloat, N> ba;
    ba.from_vector(::aie::load_v<N>(b + i), 0);
    ::aie::store_v(out + i, ::aie::add(::aie::load_v<N>(a + i),
                                       ba.template to_vector<float>()));
  }
  event1();
}

extern "C" {
void acc_add_row(float *a, bfloat16 *b, float *out, int32_t cols) {
  acc_add_row<16>(a, b, out, cols);
}
}
#else
template <int N>
void acc_add_row(const float *restrict a, const float *restrict b,
                 float *restrict out, int32_t cols) {
  event0();
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> av = ::aie::load_v<N>(a + i);
    ::aie::vector<float, N> bv = ::aie::load_v<N>(b + i);
    ::aie::store_v(out + i, ::aie::add(av, bv)); // f32 add is exact (no narrowing)
  }
  event1();
}

extern "C" {
void acc_add_row(float *a, float *b, float *out, int32_t cols) {
  acc_add_row<16>(a, b, out, cols);
}
}
#endif  // ACCADD_B_BF16
