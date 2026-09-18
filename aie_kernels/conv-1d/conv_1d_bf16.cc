//===- conv_1d_bf16.cc -------------------------------------------*- C++ -*-===//
//
// BF16 arm of the causal dilated conv-1d brick (aie_kernels/conv-1d), same math as
// conv_1d.cc's conv_1d_causal_core[_vec]:
//
//   y[co, t] = bias[co] + sum_ci sum_j w[ci, co, j] * x[ci, t - (k-1-j)*dilation]
//
// WHY A SEPARATE FILE. Same reason conv_1d.cc keeps its scalar core as the reference the vector
// core gates against: this is a NEW dtype arm, not a rewrite. conv_1d.cc's own f32 vector core is
// untouched.
//
// WHY THIS EXISTS -- the FORMAT lever, not the fusion lever. aie2p has no f32 vector MAC:
// `aie_api/detail/aie2p/config.hpp` hardcodes `__AIE_API_FP32_SUPPORT__ = 0`, and
// conv_1d_causal_core_vec's own header measured what that costs -- `aie::mac` on `float` expands
// to a 3-term bf16 decomposition, ~28 real vector ops for one source-level MAC, ~86 cycles/tap
// device-measured (0.185 MAC/cyc). bf16 operands need none of that: `aie::mac` on
// `aie::vector<bfloat16,N>` issues the hardware MAC directly. This file swaps operand/storage dtype
// only -- same gather math, same alignment discipline, same accumulate-in-f32 rule -- so a
// side-by-side compile against conv_1d_causal_core_vec isolates the format effect from everything
// else (see the accompanying device gate script's op-count note for the derived ratio).
//
// COMPILE-TIME BOUNDS, DELIBERATELY, NOT conv_1d_causal_core_vec's runtime `int32_t t`/`c_in`/`k`.
// A vector loop whose trip count is a genuine RUNTIME value is a known aie2p miscompile hazard in
// this tree -- `sin`/`gelu-erf` were red across multiple sessions for exactly this shape, fixed only
// by removing the runtime bound (kb/kernel-internal-loops-miscompile-put-volume-in-the-worker; two
// shipped bricks, snake and softmax, still carry a runtime-bound internal loop and are green, but
// that is an UNEXPLAINED, toolchain-fragile exception per that note's own kill_if, not a precedent
// to build on). C_IN/K/T/DILATION are template ints here so every loop bound the compiler sees is a
// literal, the same guarantee `sin_core<N>` gets from templating on N. This is new code with no
// device history of its own, so it does not get the benefit of the doubt snake/softmax have earned.
//
// NO STATIC L1 STATE. Every buffer here is caller-owned (tile/resident/out pointers); nothing
// persists across calls. This is the OTHER lesson this codec paid for twice --
// conv_transpose_channel.cc's header has the failure catalog (upsample_stage's static scratch: green
// twice, then NaN, source unchanged; residual_unit_bf16.cc: NaN wherever it links) and the operating
// rule (kb/static-l1-state-makes-kernels-unreproducible). ONE OUTPUT CHANNEL PER CALL, same contract
// as conv_1d_causal_core: the caller streams tiles; this kernel never loops over them.
//
// ROUNDING ON THE accfloat -> bfloat16 NARROW. `aie::set_rounding(conv_even)` is called explicitly,
// defensively -- NOT because it is known to fix anything here. It measured INERT for this exact
// conversion elsewhere in this tree: gemm_bf16xbfp16.cc's own accfloat->bfloat16 narrow (same
// `acc.template to_vector<bfloat16>()` call shape) forced floor vs forced conv_even and got
// BIT-IDENTICAL output on device (rel-L2 6.777424096e-03 either way, 4 arms,
// _verify/verify_rounding_ab.py) even though `mov crrnd, #0xc` is verifiably emitted -- the op does
// not consult the register. That is a DIFFERENT conversion from the one `to_fixed` is inert for
// (float->int, kb/magic-number-rounding-is-a-noop-on-aie2p) and from the one that DOES honor it
// (bf16->bfp16ebs8 inside emulated aie::mmul, 1.3x accuracy, kb/log/bfp16-emulation-inherits-floor-
// rounding) -- so nothing here is assumed by analogy; the accfloat->bfloat16 case is CITED from an
// existing device measurement of the identical call shape, not re-derived (this is a non-device
// session; see the gate script for the A/B that would re-confirm it on THIS kernel specifically).
// Kept for one instruction per call and because the register is GLOBAL AND STICKY -- whatever a
// prior kernel on this core left set otherwise leaks in; setting it here removes that dependency
// regardless of whether the hardware currently honors it.
//
// ACCUMULATION STAYS F32. `aie::accum<accfloat, N>` is the wide accumulator for both the vector and
// scalar cores below regardless of bf16 storage -- same rule as
// kb/bf16-norm-numerics-and-accumulation-guards (f32-accumulate the SUM; bf16 accumulation is the
// known way to lose a norm) and the same choice residual_unit_bf16.cc's header already made for its
// (unrelated, broken) restructure.
//
// BIAS: packed bf16 IN THE TILE, widened to float before use. bricklib's streamed rail
// (`_build_streamed`) types a whole tile with ONE dtype, so a tile cannot mix bf16 weights with an
// f32 bias without a byte-reinterpret packing trick; bias is one scalar per output channel and its
// quantization cost is negligible against the gate, so it rides in bf16 like the weights rather than
// adding that complexity.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace route_b_bricks {

// SCALAR reference. Stateless, no vector intrinsics, so it carries none of this file's own
// hazards -- it is what the vector core below is gated against, same relationship
// conv_1d_causal_core has to conv_1d_causal_core_vec in the f32 brick.
//
// w_row: [c_in * k] bf16 weights for ONE output channel, laid out [ci*k + j] (ggml_conv_1d layout,
//        see conv_1d.cc's own comment -- unchanged by dtype).
// x:     [c_in, t] bf16 activation.
// out:   [t] bf16, for that output channel.
template <int C_IN, int K, int T, int DILATION>
static inline void conv_1d_causal_core_bf16_scalar(const bfloat16 *restrict x,
                                                    const bfloat16 *restrict w_row, float bias,
                                                    bfloat16 *restrict out) {
  event0();
  float acc[T];
  for (int p = 0; p < T; p++) {
    acc[p] = bias;
  }
  for (int ci = 0; ci < C_IN; ci++) {
    const bfloat16 *xr = x + ci * T;
    const bfloat16 *wr = w_row + ci * K;
    for (int j = 0; j < K; j++) {
      const float wv = (float)wr[j];
      const int shift = (K - 1 - j) * DILATION;
      for (int p = shift; p < T; p++) {
        acc[p] += wv * (float)xr[p - shift];
      }
    }
  }
  for (int p = 0; p < T; p++) {
    out[p] = (bfloat16)acc[p];
  }
  event1();
}

// VECTORISED, native bf16 aie::mac. Same 3-case alignment logic as conv_1d_causal_core_vec (see
// that function's own comment for why each case is alignment-safe -- the reasoning is dtype-
// independent, only the lane count N changes): a naive aligned-output/misaligned-input load would
// issue an unaligned load_v, which on this toolchain snaps to the aligned base and silently returns
// the wrong data instead of erroring. So every load_v here is on an N-aligned address and the
// misaligned window is built in-register with shuffle_down_fill.
//
// N=32: aie2p's vector register is 512 bits: 512/16 = 32 bf16 lanes, double conv_1d_causal_core_vec's
// N=16 f32 lanes (512/32 = 16) at the SAME register width -- the other half of the format lever
// (more lanes per instruction), independent of the MAC-decomposition saving described above.
//
// PRECONDITION: T must be a multiple of N (32), same contract every vectorised brick in this tree
// carries (softmax/layernorm/rmsnorm/relu2/conv_1d_causal_core_vec all require cols % N == 0).
template <int N, int C_IN, int K, int T, int DILATION>
static inline void conv_1d_causal_core_bf16_vec(const bfloat16 *restrict x,
                                                 const bfloat16 *restrict w_row, float bias,
                                                 bfloat16 *restrict out) {
  static_assert(T % N == 0, "T must be a multiple of N");
  event0();
#ifndef CONV1D_BF16_NO_SET_ROUNDING
  ::aie::set_rounding(::aie::rounding_mode::conv_even); // defensive; see file header
#endif
  const ::aie::vector<bfloat16, N> zero_v = ::aie::zeros<bfloat16, N>();
  const ::aie::vector<float, N> bias_bv = ::aie::broadcast<float, N>(bias);

  for (int oc = 0; oc < T; oc += N) {
    ::aie::accum<accfloat, N> acc;
    acc.from_vector(bias_bv);
    for (int ci = 0; ci < C_IN; ci++) {
      const bfloat16 *xr = x + ci * T;
      const bfloat16 *wr = w_row + ci * K;
      for (int j = 0; j < K; j++) {
        const int shift = (K - 1 - j) * DILATION;
        if (oc + N <= shift) continue; // entirely left of the causal window: no contribution
        const ::aie::vector<bfloat16, N> wbv = ::aie::broadcast<bfloat16, N>(wr[j]);
        const int read_base = oc - shift;
        if (read_base < 0) {
          const int r = read_base + N; // in [1, N-1]
          const ::aie::vector<bfloat16, N> b = ::aie::load_v<N>(xr);
          acc = ::aie::mac(acc, ::aie::shuffle_down_fill(zero_v, b, (unsigned)r), wbv);
        } else {
          const int ra = (read_base / N) * N; // floor to N (read_base >= 0)
          const int r = read_base - ra;
          if (r == 0) {
            acc = ::aie::mac(acc, ::aie::load_v<N>(xr + ra), wbv);
          } else {
            const ::aie::vector<bfloat16, N> a0 = ::aie::load_v<N>(xr + ra);
            const ::aie::vector<bfloat16, N> a1 = ::aie::load_v<N>(xr + ra + N);
            acc = ::aie::mac(acc, ::aie::shuffle_down_fill(a0, a1, (unsigned)r), wbv);
          }
        }
      }
    }
    ::aie::store_v(out + oc, acc.template to_vector<bfloat16>());
  }
  event1();
}

} // namespace route_b_bricks
