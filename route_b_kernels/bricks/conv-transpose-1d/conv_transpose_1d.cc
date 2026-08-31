//===- conv_transpose_1d.cc ------------------------------------*- C++ -*-===//
//
// Causal transposed 1-D convolution brick (route_b_kernels/bricks, group: conv).
//
//   y[co, ti*stride + j] += w[ci, co, j] * x[ci, ti],  then per-output-channel bias.
// Cropping is a caller-side view (drop the last `crop_right` columns), so it costs no copy here.
// Reference semantics: s2.cpp/src/s2_codec.cpp:236-250. The fish-audio DAC decoder runs one of these
// per upsample stage at decoder_rates [8, 8, 4, 2].
//
// SCALAR ON PURPOSE. The vectorised scatter form writes at `ti * stride` offsets, and the codec's
// strides are 8/8/4/2 -- none a multiple of the 16-float vector width -- so every store after the
// first would be unaligned, and unaligned vector access truncates on aie2p
// (kb/aie2p-unaligned-vector-load-truncation). This is the CORRECTNESS phase: a scalar kernel is
// obviously right and becomes the reference the vectorised form is later gated against. Making it
// fast is a separate, deliberate step (aligned scratch row + overlap-add, or GEMM + overlap-add).
//
// Caller owns zeroing `out` (c_out * out_len floats) before the call.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

inline void conv_transpose_1d_core(const float *restrict x, const float *restrict w,
                                   const float *restrict bias, float *restrict out, int32_t c_in,
                                   int32_t c_out, int32_t k, int32_t t, int32_t stride,
                                   int32_t out_len) {
  event0();
  for (int32_t co = 0; co < c_out; co++) {
    float *o_row = out + (int32_t)(co * out_len);
    for (int32_t p = 0; p < out_len; p++) {
      o_row[p] = bias[co];
    }
    for (int32_t ci = 0; ci < c_in; ci++) {
      const float *w_row = w + (int32_t)((ci * c_out + co) * k);
      const float *x_row = x + (int32_t)(ci * t);
      for (int32_t ti = 0; ti < t; ti++) {
        const float xv = x_row[ti];
        const int32_t base = ti * stride;
        for (int32_t j = 0; j < k; j++) {
          o_row[base + j] += w_row[j] * xv;
        }
      }
    }
  }
  event1();
}

} // namespace route_b_bricks

extern "C" {

// x: [c_in, t], w: [c_in, c_out, k], bias: [c_out], out: [c_out, (t-1)*stride + k].
void conv_transpose_1d_f32(float *x, float *w, float *bias, float *out, int32_t c_in, int32_t c_out,
                           int32_t k, int32_t t, int32_t stride, int32_t out_len) {
  route_b_bricks::conv_transpose_1d_core(x, w, bias, out, c_in, c_out, k, t, stride, out_len);
}

} // extern "C"
