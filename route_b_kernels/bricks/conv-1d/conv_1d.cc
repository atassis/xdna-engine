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
// ONE OUTPUT CHANNEL PER CALL. The residual-unit weights are [c_out, c_in, k] = 96*96*7 f32 =
// 258 KB and no core tile holds that, so the caller streams one output channel's slice at a time
// past a resident activation -- the same split the fused upsample stage uses, and for the same
// reason. The caller owns that decomposition; this kernel just does one channel.
//
// conv_1d_causal_core (below) is SCALAR and stays the correctness reference the vectorised form is
// gated against -- unlike conv-transpose-1d, whose stride-offset SCATTER on write is genuinely
// unaligned, this brick's gather is contiguous in t, so a vector form (conv_1d_causal_core_vec,
// below) is available. See its own comment for how it stays alignment-safe.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

// w_row: [c_in * k] weights for ONE output channel, laid out [ci*k + j]. With ggml_conv_1d's
//        [c_out, c_in, k] layout that slice is w[co] and is already contiguous -- no gather.
//        Note conv_transpose_1d uses the TRANSPOSED layout; see conv-1d/golden.py.
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

// VECTORISED form of conv_1d_causal_core. Same signature, same math, ADDITIVE -- the scalar core
// above is untouched and stays the reference this is gated against (see _verify/verify_conv_1d.py).
//
// Vectorises over t (output position p), N lanes at a time. The gather is contiguous in t (dilation
// only changes the scalar `shift`, never the read stride -- see file header), but `shift` is almost
// never a multiple of N, so a naive aligned-output/misaligned-input load would issue an unaligned
// load_v. On this toolchain that SNAPS to the aligned base and silently returns the wrong data
// instead of erroring (the same failure dwconv1d_shift's header records for its own FIR window,
// route_b_kernels/dwconv1d/dwconv1d.cc) -- so every load_v here is on an N-aligned address, and the
// misaligned window is built in-register with shuffle_down_fill, exactly that kernel's technique.
//
// dwconv1d_shift affords a padded scratch buffer because T/K/P are compile-time there; this core
// keeps c_in/k/t/dilation runtime, matching conv_1d_causal_core, so there is no compile-time bound
// to size a buffer from -- and it turns out not to need one. Per output chunk [oc, oc+N) the read
// window is x[oc-shift .. oc-shift+N), and since 0 <= oc <= t-N and shift >= 0, that window starts
// at most N-1 elements before x's own base and ends at most at x's last element. One shared,
// always-zero length-N vector stands in for that left slack; nothing is ever read outside
// x[0, c_in*t).
//
// Three cases per (ci, j, oc), decided by comparing oc/oc+N against shift:
//   oc + N <= shift : chunk is entirely left of this tap's causal window -> contributes 0, skip.
//   oc     >= shift : chunk is entirely inside it -> read_base = oc-shift >= 0: one aligned load if
//                      read_base is itself N-aligned, else an aligned pair + shuffle_down_fill.
//   otherwise       : the chunk straddles `shift` -> read_base in [-(N-1), -1]; splice the shared
//                      zero vector against x's own first N-wide chunk.
// The aligned-pair case never reads past x's t elements: read_base <= t - N always. A second vector
// is only read when read_base isn't already N-aligned, and since t % N == 0, the one read_base that
// IS N-aligned at the top of the range is read_base == t - N itself -- so whenever a second vector
// is needed, read_base < t - N, which places its aligned floor at or before t - 2N, keeping the pair
// inside [0, t). PRECONDITION: t must be a multiple of N (the same contract every vectorised f32
// brick in this tree carries -- softmax/layernorm/rmsnorm/relu2 all require cols % N == 0).
template <int N>
static inline void conv_1d_causal_core_vec(const float *restrict x, const float *restrict w_row,
                                            float bias, float *restrict out, int32_t c_in,
                                            int32_t k, int32_t t, int32_t dilation) {
  event0();
  const ::aie::vector<float, N> zero_v = ::aie::zeros<float, N>();
  const ::aie::vector<float, N> bias_bv = ::aie::broadcast<float, N>(bias);

  for (int32_t oc = 0; oc < t; oc += N) {
    ::aie::accum<accfloat, N> acc;
    acc.from_vector(bias_bv);
    for (int32_t ci = 0; ci < c_in; ci++) {
      const float *xr = x + (int32_t)(ci * t);
      const float *wr = w_row + (int32_t)(ci * k);
      for (int32_t j = 0; j < k; j++) {
        const int32_t shift = (k - 1 - j) * dilation;
        if (oc + N <= shift) continue; // entirely left of the causal window: no contribution
        const ::aie::vector<float, N> wbv = ::aie::broadcast<float, N>(wr[j]);
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
    ::aie::store_v(out + oc, acc.template to_vector<float>());
  }
  event1();
}

} // namespace route_b_bricks
