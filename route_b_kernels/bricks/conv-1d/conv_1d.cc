//===- conv_1d.cc ----------------------------------------------*- C++ -*-===//
//
// CAUSAL dilated 1-D convolution brick (route_b_kernels/bricks, group: conv).
//
//   y[co, t] = bias[co] + sum_ci sum_j w[ci, co, j] * x[ci, t - (k-1-j)*dilation]
//
// with x[.., <0] reading as zero. Reference semantics: s2.cpp/src/s2_codec.cpp:224-235. Every
// residual unit in the DAC codec decoder uses two of these -- one dilated (1, 3, 9) and one 1x1
// (s2_codec.cpp:381-393) -- so this brick plus snake is the whole residual unit.
//
// s2.cpp expresses it as left-pad-then-valid-conv; at stride 1 that is exactly (k-1)*dilation zeros
// on the left, so out_len == in_len. Evaluated here in the direct form instead: a kernel that
// materialised the padded copy would spend L1 on zeros, and the bound check is one comparison.
// golden.py asserts the two forms agree, and asserts causality directly, because a sign error in
// the shift is invisible to a shape check.
//
// ONE OUTPUT CHANNEL PER CALL. The residual-unit weights are [c_in, c_out, k] = 96*96*7 f32 =
// 258 KB and no core tile holds that, so the caller streams one output channel's slice at a time
// past a resident activation -- the same split the fused upsample stage uses, and for the same
// reason. The caller owns that decomposition; this kernel just does one channel.
//
// SCALAR, like the conv-transpose-1d brick it sits beside. This is the correctness phase, and a
// scalar kernel is the reference the vectorised form gets gated against later. The vector form here
// is genuinely available (unlike in conv_transpose_1d, where the stride-offset scatter would be
// unaligned) since the gather is contiguous in t -- worth taking, but only once this is green.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

// w_row: [c_in * k] weights for ONE output channel, laid out [ci*k + j].
// x:     [c_in, t] activation.
// out:   [t] for that output channel.
static inline void conv_1d_causal_core(const float *restrict x, const float *restrict w_row,
                                       float bias, float *restrict out, int32_t c_in, int32_t k,
                                       int32_t t, int32_t dilation) {
  event0();
  for (int32_t p = 0; p < t; p++) {
    out[p] = bias;
  }
  for (int32_t ci = 0; ci < c_in; ci++) {
    const float *xr = x + (int32_t)(ci * t);
    const float *wr = w_row + (int32_t)(ci * k);
    for (int32_t j = 0; j < k; j++) {
      const float wv = wr[j];
      const int32_t shift = (k - 1 - j) * dilation;
      // Everything before `shift` reads padding, which is zero, so it contributes nothing --
      // start at the first output that has a real input rather than testing inside the loop.
      for (int32_t p = shift; p < t; p++) {
        out[p] += wv * xr[p - shift];
      }
    }
  }
  event1();
}

} // namespace route_b_bricks
