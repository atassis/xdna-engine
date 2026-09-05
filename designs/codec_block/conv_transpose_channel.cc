//===- conv_transpose_channel.cc -------------------------------*- C++ -*-===//
//
// Causal transposed 1-D convolution for ONE output channel, with NO static state.
//
// WHY THIS EXISTS. The fused upsample stage (upsample_stage.cc) keeps a static [c_in, t] scratch so
// snake is computed once per stream instead of once per output channel. It was measured green twice
// -- 3.722e-07 synthetic, 5.337e-07 on a real window -- and then, with its source unchanged and the
// device healthy (a copy kernel is still bit-exact on every rail), it began returning NaN on every
// run. Nothing in its translation unit moved.
//
// That fits a pattern rather than being a one-off. Across this codec's kernels, every one that keeps
// a static scratch has been fragile and every one that keeps none has been repeatedly stable:
//
//     conv-1d, snake                 no static buffer   green all session, many rebuilds
//     upsample_stage (fused)         static scratch     green twice, then NaN, source unchanged
//     residual_unit (fused)          two static         green only at t=16
//     residual_unit_bf16             static             NaN wherever it links
//
// `.bss` is not zeroed on these cores (the fused stage's first version proved that by returning NaN
// from an uninitialised `static int` guard), and the buffers are placed by the allocator, so any
// rebuild can move them. Correctness that depends on static L1 state is correctness that a comment
// change can revoke. So the driver does not depend on it: this kernel holds nothing between calls,
// and the caller pays for snake as its own pass instead.
//
//   y[co, ti*stride + j] += w[ci, co, j] * x[ci, ti],  then bias, then crop_right = stride.
//
// Reference semantics: s2.cpp/src/s2_codec.cpp:236-250, same as the conv-transpose-1d brick, which
// this is the one-output-channel form of. Kept out of that brick's file so its own green is not
// perturbed -- on this toolchain, adding an unused function to a translation unit is not free.
//
// SCALAR, for the reason the brick documents: the scatter writes at `ti * stride` offsets and the
// codec's strides are 8/8/4/2, none a multiple of the 16-float vector width, so every store after
// the first would be unaligned and unaligned vector access truncates on aie2p.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

// x:     [c_in, t] activation, already snake-activated by the caller.
// w_col: [c_in * k] weights for ONE output channel, laid out [ci*k + j] -- that is w[:, co, :] of
//        the conv_transpose layout [c_in, c_out, k], which is the TRANSPOSE of what conv_1d takes.
// out:   [t * stride], already cropped: crop_right == stride drops exactly the tail, so the first
//        t*stride columns of the uncropped result are the answer.
static inline void conv_transpose_channel_core(const float *restrict x, const float *restrict w_col,
                                               float bias, float *restrict out, int32_t c_in,
                                               int32_t k, int32_t t, int32_t stride) {
  event0();
  const int32_t out_len = t * stride;
  for (int32_t p = 0; p < out_len; p++) {
    out[p] = bias;
  }
  for (int32_t ci = 0; ci < c_in; ci++) {
    const float *w_row = w_col + (int32_t)(ci * k);
    const float *x_row = x + (int32_t)(ci * t);
    for (int32_t ti = 0; ti < t; ti++) {
      const float xv = x_row[ti];
      const int32_t base = ti * stride;
      for (int32_t j = 0; j < k; j++) {
        const int32_t q = base + j;
        // Guard rather than crop afterwards: the uncropped length is (t-1)*stride + k, so the last
        // few taps of the final input fall past the cropped window and are simply not written.
        if (q < out_len) {
          out[q] += w_row[j] * xv;
        }
      }
    }
  }
  event1();
}

} // namespace route_b_bricks
