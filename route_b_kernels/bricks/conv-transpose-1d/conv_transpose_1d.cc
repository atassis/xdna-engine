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
// fast is a separate, deliberate step -- taken below not by aligning the scatter but by deleting
// it (conv_transpose_channel_core_vec, a per-output-PHASE reformulation with no scatter at all).
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

// VECTORISED form of ONE OUTPUT CHANNEL of a transposed conv -- the granularity
// route_b_kernels/codec_block/conv_transpose_channel.cc's scalar `conv_transpose_channel_core`
// already runs at (that is what window_driver.py and quantizer_driver.py actually dispatch; this
// brick's own `conv_transpose_1d_core` above is multi-channel and used only by this file's own
// oneshot gate). Kept in THIS file rather than added to conv_transpose_channel.cc: that file's own
// header documents why it stays untouched (adding an unused function to a translation unit is not
// free on this toolchain, and it is the file the driver links) -- same reasoning conv_1d.cc's
// vector core followed by living beside its own scalar reference rather than in a third file.
//
// THE SCATTER, DELETED, NOT ALIGNED. The naive vector form stores at `ti * stride` offsets, which
// for this codec's strides (8/8/4/2, never a multiple of N=16) makes every store after the first
// unaligned, and an unaligned vector STORE truncates silently on this toolchain, same failure class
// as an unaligned load (kb/aie2p-unaligned-vector-load-truncation, the hazard conv_1d_causal_core_vec's
// own header names). A transposed conv is the textbook "gather" decomposition into `stride`
// ordinary causal convolutions, one per output PHASE s = p mod stride: writing p = q*stride + s,
//
//   y[co, q*stride+s] = bias[co] + sum_ci sum_m w_s[ci,co,m] * x[ci, q-m],   x[<0] := 0
//   w_s[m] = w[ci,co, s + m*stride],   m = 0 .. M_s-1,   M_s = ceil((k-s)/stride)
//
// which is EXACTLY conv_1d_causal_core_vec's shape (a causal FIR, contiguous gather in q, dilation
// 1, this phase's own M_s taps) -- so every store this kernel makes is a plain N-aligned, fully
// contiguous vector store into a PER-PHASE run. There is no scatter left to align; the read-side
// misalignment (read_base not a multiple of N) is handled the same in-register way
// conv_1d_causal_core_vec does, with aligned loads plus shuffle_down_fill, and the same bounds
// argument applies unchanged: shift >= 0 and oc <= t-N together keep every load's window inside
// x[0, c_in*t), by the identical reasoning conv_1d_causal_core_vec's own header spells out.
//
// Both crop_right conventions this codec uses fall out for free, with NO crop_right parameter.
// The natural (uncropped) q-range per phase is q = 0..t-1 when k <= stride (quantizer,
// crop_right=0) and q = 0..t when k == 2*stride (decoder, crop_right=stride); the decoder's crop
// drops exactly q=t -- every phase's own last sample, all `stride` of them together, which IS the
// crop_right=stride tail. So computing only q = 0..t-1 per phase (this kernel's whole loop range)
// already is the cropped answer under EITHER convention. Cross-checked against
// conv_transpose_channel_core's own `if (q < out_len)` guard, which crops the same way for the
// same reason, and against verify_conv_transpose_1d.py's VEC gate, which exercises both.
//
// OUTPUT LAYOUT is the one contract change from the scalar core: `out` is PHASE-MAJOR, [stride, t]
// flattened as out[s*t + q], NOT the interleaved [t*stride] the scalar core produces. Interleaving
// (out[s*t+q] -> final[q*stride+s], a reshape(stride,t).T) is a pure host reshape -- zero device
// cost -- and it is the caller's job, exactly as window_driver.py already does the window-stitch on
// host for every op in this file. Keeping the interleave off-chip is what keeps every on-chip store
// aligned; it is not a limitation forced by anything else.
//
// PRECONDITIONS: t must be a multiple of N, same contract as conv_1d_causal_core_vec and every
// vectorised f32 brick in this tree. k <= 2*stride (both codec cases: k==stride, k==2*stride) --
// the "no crop_right parameter, always compute q=0..t-1" simplification above is where that
// enters: at k==2*stride the natural (uncropped) top-of-range is q==t exactly one row past what
// this kernel computes, which crop_right==stride is defined to discard, but at k==3*stride it is
// two rows past, and this kernel does not compute either of them. A numpy transliteration of this
// exact algorithm confirms the general per-phase math (arbitrary k, including k > 2*stride) against
// golden.conv_transpose_1d_ref(..., crop_right=k-stride); this kernel just never needs to compute
// beyond q=t-1, since nothing in this codec calls it with k > 2*stride.
//
// c_in/k/t/stride stay RUNTIME int32_t parameters, matching conv_transpose_channel_core's own
// signature. That is syntactically the shape flagged as a miscompile hazard
// (docs/kb/kernel-internal-loops-miscompile-put-volume-in-the-worker.md: a genuinely runtime trip
// count driving a VECTOR loop can silently corrupt output) -- but every call site this brick is
// gated from (verify_conv_transpose_1d.py's generated shims, mirroring conv_1d_causal_core_vec's
// own gate) substitutes them as C++ literal constants, and this function is `static inline` in the
// same translation unit as its caller, so -O2 constant-folds the loop trip counts during inlining.
// That is the exact shape conv_1d_causal_core_vec already ships green in, and the shape snake's
// (4-chunk) and softmax's (4-iteration) shipped internal loops resolve to as well -- a
// COMPILE-TIME-bounded loop after inlining, not a genuinely runtime one. Not proven for THIS kernel
// until its own device gate runs; see the report's UNVERIFIED list.
template <int N>
static inline void conv_transpose_channel_core_vec(const float *restrict x, const float *restrict w_col,
                                                    float bias, float *restrict out, int32_t c_in,
                                                    int32_t k, int32_t t, int32_t stride) {
  event0();
  const ::aie::vector<float, N> zero_v = ::aie::zeros<float, N>();
  const ::aie::vector<float, N> bias_bv = ::aie::broadcast<float, N>(bias);

  for (int32_t s = 0; s < stride; s++) {
    float *o_phase = out + (int32_t)(s * t);
    const int32_t m_s = (s < k) ? (k - s + stride - 1) / stride : 0; // ceil((k-s)/stride), clamped
    for (int32_t oc = 0; oc < t; oc += N) {
      ::aie::accum<accfloat, N> acc;
      acc.from_vector(bias_bv);
      for (int32_t ci = 0; ci < c_in; ci++) {
        const float *xr = x + (int32_t)(ci * t);
        const float *wr = w_col + (int32_t)(ci * k);
        for (int32_t m = 0; m < m_s; m++) {
          const int32_t shift = m; // dilation 1: tap m reads x[ci, q-m]
          if (oc + N <= shift) continue; // entirely left of this tap's causal window: skip
          const ::aie::vector<float, N> wbv = ::aie::broadcast<float, N>(wr[s + m * stride]);
          const int32_t read_base = oc - shift;
          if (read_base < 0) {
            const int32_t r = read_base + N; // in [1, N-1]
            const ::aie::vector<float, N> b = ::aie::load_v<N>(xr);
            acc = ::aie::mac(acc, ::aie::shuffle_down_fill(zero_v, b, (unsigned)r), wbv);
          } else {
            const int32_t ra = (read_base / N) * N; // floor to N (read_base >= 0)
            const int32_t r = read_base - ra;
            if (r == 0) {
              acc = ::aie::mac(acc, ::aie::load_v<N>(xr + ra), wbv);
            } else {
              const ::aie::vector<float, N> a0 = ::aie::load_v<N>(xr + ra);
              const ::aie::vector<float, N> a1 = ::aie::load_v<N>(xr + ra + N);
              acc = ::aie::mac(acc, ::aie::shuffle_down_fill(a0, a1, (unsigned)r), wbv);
            }
          }
        }
      }
      ::aie::store_v(o_phase + oc, acc.template to_vector<float>());
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
