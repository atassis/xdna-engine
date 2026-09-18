//===- conv_transpose_channel_bf16.cc --------------------------*- C++ -*-===//
//
// BF16 arm of conv_transpose_channel.cc, same math, same scalar structure, NO static state:
//
//   y[co, ti*stride + j] += w[ci, co, j] * x[ci, ti],  then bias, then crop_right = stride.
//
// STAYS SCALAR, and this is the honest limit of the format lever for this one kernel. conv-1d's
// bf16 win is TWO independent levers: native MAC (no aie2p f32-emulation tax) and double the vector
// lanes at the same 512-bit register (bf16 N=32 vs f32 N=16). This kernel gets neither: the scatter
// writes at `ti * stride` offsets and the codec's strides are 8/8/4/2 -- none a multiple of EITHER
// vector width, 16 (f32) or 32 (bf16, which needs MORE alignment, not less) -- so every store after
// the first would still be unaligned, and unaligned vector access truncates on aie2p
// (kb/aie2p-unaligned-vector-load-truncation). conv_1d.cc's own header notes the brick this is built
// from (conv-transpose-1d) reformulates the scatter into per-phase contiguous gathers to get a
// vector form anyway (conv_transpose_channel_core_vec); that reformulation is dtype-independent and
// is NOT ported here, to keep this file's blast radius to the format swap alone -- it is future work
// for whoever fuses this into a resident-stream kernel, not part of this change.
//
// So what bf16 buys THIS kernel is the OTHER half of the format lever this project's doctrine
// separates explicitly: not compute throughput, but WEIGHT-STREAM BYTES -- half the DMA bytes per
// streamed weight tile, at unchanged (scalar) compute cost. Report it as that, not as a MAC-cycle
// win: see the accompanying device gate script and its cycle-ratio note.
//
// NO STATIC L1 STATE, matching conv_transpose_channel.cc's own reason for existing (its header
// carries the failure catalog this codec paid for: a static scratch that was green twice then NaN
// with source unchanged, and residual_unit_bf16.cc's NaN restructure) -- every buffer here is
// caller-owned, nothing persists across calls, and this is ONE OUTPUT CHANNEL PER CALL, the same
// contract as the f32 original.
//
// ACCUMULATION STAYS F32 (kb/bf16-norm-numerics-and-accumulation-guards): `acc` is a local float
// array regardless of x/w's bf16 storage, narrowed to bfloat16 once on the way into `out`.
//
// ROUNDING on the f32->bfloat16 narrow: this is a SCALAR cast (`(bfloat16)acc[p]`), not the vector
// accfloat->bfloat16 `to_vector<bfloat16>()` path conv_1d_bf16.cc calls `set_rounding` before -- so
// that file's citation (gemm_bf16xbfp16.cc's device A/B) does not directly cover this site; aie_api's
// scalar-vs-vector to_fixed asymmetry documented in kb/magic-number-rounding-is-a-noop-on-aie2p
// (both overloads there turned out inert, but that was established BY MEASUREMENT, not by assuming
// scalar and vector behave the same) means this scalar narrow's rounding behaviour is UNVERIFIED by
// us and should not be assumed identical to the vector site. Left unset here rather than cited
// speculatively; the device gate script's rel-L2 numbers are what actually bound its effect at this
// gate's ~1e-2 accuracy budget.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

// x:     [c_in, t] bf16 activation, already snake-activated by the caller.
// w_col: [c_in * k] bf16 weights for ONE output channel, laid out [ci*k + j] -- w[:, co, :] of the
//        conv_transpose layout [c_in, c_out, k] (the TRANSPOSE of conv_1d's layout).
// out:   [t * stride] bf16, already cropped (crop_right == stride drops exactly the tail).
template <int C_IN, int K, int T, int STRIDE>
static inline void conv_transpose_channel_core_bf16(const bfloat16 *restrict x,
                                                     const bfloat16 *restrict w_col, float bias,
                                                     bfloat16 *restrict out) {
  static_assert(K == 2 * STRIDE, "codec invariant: k == 2*stride (same assert as upsample_stage.cc)");
  event0();
  constexpr int out_len = T * STRIDE;
  float acc[out_len];
  for (int p = 0; p < out_len; p++) {
    acc[p] = bias;
  }
  for (int ci = 0; ci < C_IN; ci++) {
    const bfloat16 *w_row = w_col + ci * K;
    const bfloat16 *x_row = x + ci * T;
    for (int ti = 0; ti < T; ti++) {
      const float xv = (float)x_row[ti];
      const int base = ti * STRIDE;
      for (int j = 0; j < K; j++) {
        const int q = base + j;
        // Guard rather than crop afterwards: the uncropped length is (t-1)*stride + k, so the last
        // few taps of the final input fall past the cropped window and are simply not written.
        if (q < out_len) {
          acc[q] += (float)w_row[j] * xv;
        }
      }
    }
  }
  for (int p = 0; p < out_len; p++) {
    out[p] = (bfloat16)acc[p];
  }
  event1();
}

} // namespace route_b_bricks
