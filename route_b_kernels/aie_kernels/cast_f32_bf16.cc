//===- cast_f32_bf16.cc ---------------------------------------*- C++ -*-===//
//
// Device-side f32 -> bf16 elementwise cast (resident-rails seam primitive).
//
// The resident-stream frontier hands activations device-to-device between xclbin
// dispatches. Producers that emit f32 (e.g. the ctxLN ln_2pass kernel, "never
// re-expand") must be cast to bf16 before the whole_array matmul (bf16 in) can
// consume them WITHOUT a host round-trip. This is that cast, on-chip: one row of
// `cols` f32 in -> `cols` bf16 out, matching the host npu_xrt::pack_f32_to_bf16
// (round-to-nearest-even, the aie to_vector<bfloat16> default).
//
// One call casts ONE row of `cols` elements (the ml/layernorm per-row core_body
// contract, reused verbatim). cols must be a multiple of N (16).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

template <int N>
void cast_f32_bf16_row(const float *restrict input, bfloat16 *restrict output,
                       int32_t cols) {
  event0();
  // Round-to-nearest-even to match the host AVX512 pack (default accum narrow truncates).
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  for (int i = 0; i < cols; i += N) {
    // f32 -> bf16 narrow via an accumulator (the aie_api idiom, see
    // mm_silu_epilogue.cc / norm_gemv_prologue.cc): vector<float> has no
    // to_vector<bfloat16>; accum<accfloat> does.
    ::aie::vector<float, N> v = ::aie::load_v<N>(input + i);
    ::aie::accum<accfloat, N> a;
    a.from_vector(v);
    ::aie::store_v(output + i, a.template to_vector<bfloat16>());
  }
  event1();
}

extern "C" {
void cast_f32_bf16_row(float *input, bfloat16 *output, int32_t cols) {
  cast_f32_bf16_row<16>(input, output, cols);
}
}
