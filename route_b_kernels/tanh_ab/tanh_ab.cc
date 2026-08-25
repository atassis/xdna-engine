//===- tanh_ab.cc -------------------------------------------------*- C++ -*-===//
//
// A/B device probe: the hw aie::tanh<bfloat16> SFU LUT vs a software tanh built on
// the exp2f_vec degree-5 poly, and the two GELU epilogues they induce.
//
// PURPOSE: mm_silu_epilogue.cc's mm_gelu_epilogue_f32o is, after the conv_even fix,
// dominated by aie::tanh itself -- the recovered (1+t) scores 1.126e-2 against exact
// tanh where a correctly-rounded bf16 would give 1.422e-3, ~7.9x, concentrated near
// inner ~ +/-0.5 rather than in the saturating tail. That profile says APPROXIMATION
// defect, not representation, and it matches what the exp2 A/B already measured on
// the sibling SFU op: aie::exp2<bfloat16> is a coarse piecewise-linear interpolator
// whose error is 720x-5771x the same poly's (see the 2026-07-29 exp2 hw-vs-sw note).
// aie::tanh lowers to `::tanh(acc)` on the same block (detail/aie2p/elementary.hpp:343,
// the ONLY Tanh specialization aie2p has), so the hypothesis is one mechanism, two ops.
//
// This measures it in ISOLATION rather than inferring it from the encoder, and it
// answers the second question in the same run: whether the software form can be
// BUILT and RUN in an epilogue-shaped loop at all. Five aligned columns per input:
//   raw     : the input, passed through (alignment / sanity)
//   hwtanh  : f32(aie::tanh<bfloat16>(x))     -- the exact call shape the epilogue uses
//   swtanh  : tanh via exp2f_vec + aie::inv   -- the candidate primitive
//   gelu_hw : the SHIPPED epilogue chain (bf16 cube, hw tanh, conv_even)
//   gelu_sw : the same chain with ONLY the tanh swapped for swtanh
// so hwtanh-vs-swtanh isolates the primitive and gelu_hw-vs-gelu_sw prices the swap
// end-to-end, with gelu_hw acting as the control that the chain is reproduced here.
//
// WHAT THIS MEASURED (2026-08-25), because two defects sat on top of the numbers:
//
// (1) STACK. aie.core's stack_size defaults to 0x400 and the generated ld.script put
//     tout_buff_0 at the very next address, zero clearance. This kernel's frame is
//     bigger than exp2_ab.cc's, so it overflowed straight into the output buffer:
//     the full form zeroed every column including the raw passthrough, and an
//     intermediate arm corrupted exactly 64 of 512 raw lanes = 256 bytes, i.e. the
//     overflow, byte for byte. exp2_ab hides the same layout because the buffer
//     ABOVE its stack is an unused ping-pong slot. Build with STACK=8192 (the
//     Makefile knob) or the measurement is of the stack, not of tanh.
//
// (2) A SECOND-CALL MISCOMPILE, still open. With the software path present and the
//     stack raised, the FIRST aie::tanh in the loop (the hwtanh column) stays
//     bit-identical to a build without that path, while the SECOND one (inside the
//     gelu_hw chain) returns 0 -- gelu_hw degenerates to exactly 0.5*x on 510/512
//     lanes. So gelu_hw is only trustworthy in the NOSW=1 build, and the gelu_hw vs
//     gelu_sw comparison is NOT sound in the arm that has both. The hwtanh vs
//     swtanh comparison IS: both come from one build whose hwtanh column is proven
//     bit-identical to the clean one.
//
// exp2f_vec is `noinline` (load-bearing -- forcing it inline miscompiles), and
// probes/ra_spill_repro.cc documents an aie2p spill-around-noinline-call defect with
// this shape. The NOSW ladder below was built to separate that from the stack, and
// found the stack first: the poly ALONE (mode 2) runs clean and reproduces its own
// 8.44e-5 accuracy, so the call is not by itself the problem.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr float LOG2E = 1.4426950408889634f;
static constexpr int VL = 16;

// Input length, compile-time (matches -DTANHAB_N in the Makefile). out[] is
// 5*TANHAB_N f32 words in one objectFifo buffer and the aie2p BD length cap is
// 16383 32-bit words per descriptor, so TANHAB_N <= 3276.
#ifndef TANHAB_N
#define TANHAB_N 1024
#endif
// Bisect arm for the software path, because "it does not run" has three candidate
// causes and they need separating: the noinline poly CALL itself, the reciprocal /
// abs / select wrapped around it, or the pair together.
//   0 = full swtanh_vec (poly + inv + abs + select)
//   1 = no software path at all -- exp2f_vec is not even linked (the control)
//   2 = the poly ALONE: store exp2f_vec(x) raw, exactly the shape exp2_ab.cc runs
//   3 = the full swtanh_vec, but only ONE call site (gelu_sw stays zeroed)
//   4 = swtanh_vec's WRAPPER with the noinline call removed: same abs/min/inv/select,
//       same value live across the spot, but an INLINE surrogate where the poly was.
//       3-vs-4 is what separates "the cross-call spill" from "one of these ops".
// The hw columns and the shipped chain are identical in all three.
#ifndef TANHAB_NOSW
#define TANHAB_NOSW 0
#endif
static_assert(TANHAB_N % VL == 0, "TANHAB_N must be a multiple of the vector width");

// ---------------------------------------------------------------------------
// VERBATIM copy of relpos_mha.cc's exp2f_vec (incl. the load-bearing noinline),
// via exp2_ab.cc. SOFTWARE f32 2^x. Do not hand-edit without re-diffing against
// route_b_kernels/relpos_mha/relpos_mha.cc.
// ---------------------------------------------------------------------------
static __attribute__((noinline)) aie::vector<float, VL> exp2f_vec(aie::vector<float, VL> x) {
  x = aie::max(x, aie::broadcast<float, VL>(-100.0f));
  aie::vector<int32_t, VL> ki = aie::to_fixed<int32_t>(x);          // round-to-nearest on aie2p
  aie::vector<float, VL> kf = aie::to_float<float>(ki);
  aie::vector<int32_t, VL> one = aie::broadcast<int32_t, VL>(1);
  aie::vector<int32_t, VL> zero = aie::broadcast<int32_t, VL>(0);
  ki = aie::sub(ki, aie::select(zero, one, aie::lt(x, kf)));
  aie::vector<float, VL> f = aie::sub(x, aie::to_float<float>(ki)); // f in [0,1)
  aie::vector<float, VL> p = aie::broadcast<float, VL>(0.0013333558f);
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.0096181291f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.0555041087f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.2402265069f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.6931471805f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(1.0f));
  aie::vector<int32_t, VL> ebits =
      aie::upshift(aie::add(ki, aie::broadcast<int32_t, VL>(127)), 23);
  aie::vector<float, VL> p2k = ebits.cast_to<float>();
  return aie::mul(p, p2k).to_vector<float>();
}

// Software tanh, f32 throughout, on top of exp2f_vec.
//   tanh(z) = sign(z) * (1 - 2/(2^(2|z|*log2e) + 1))
// The |z| fold is what keeps this in range: 2^(2z*log2e) overflows f32 for z > ~44,
// and exp2f_vec's own guard is only the LOWER clamp at -100, so the raw
// 1 - 2/(exp+1) form would return garbage on the positive tail rather than
// saturating. Folded, the argument is >= 0 and clamped at +100, where the
// reciprocal underflows to 0 and tanh saturates to exactly +/-1, which is correct.
static inline aie::vector<float, VL> swtanh_vec(aie::vector<float, VL> z) {
  const aie::vector<float, VL> zero = aie::zeros<float, VL>();
  const aie::vector<float, VL> one = aie::broadcast<float, VL>(1.0f);
  const aie::vector<float, VL> two = aie::broadcast<float, VL>(2.0f);
  aie::vector<float, VL> az = aie::abs(z);
  aie::vector<float, VL> arg =
      aie::min(aie::mul(az, aie::broadcast<float, VL>(2.0f * LOG2E)).to_vector<float>(),
               aie::broadcast<float, VL>(100.0f));
  aie::vector<float, VL> e = exp2f_vec(arg);
  aie::vector<float, VL> r = aie::inv(aie::add(e, one));
  aie::vector<float, VL> t = aie::sub(one, aie::mul(two, r).to_vector<float>());
  // restore the sign: tanh is odd.
  return aie::select(t, aie::sub(zero, t), aie::lt(z, zero));
}

// swtanh_vec with the noinline poly swapped for an inline surrogate of similar shape.
// Everything else is byte-identical: the same abs/min, the same aie::inv, the same
// aie::select sign restore, and z still live across the point where the call was. It
// does NOT compute tanh (the surrogate is not 2^x) and is never scored -- its only job
// is to answer whether the CALL or the surrounding ops are what stops the kernel.
static inline aie::vector<float, VL> swtanh_inline_surrogate(aie::vector<float, VL> z) {
  const aie::vector<float, VL> zero = aie::zeros<float, VL>();
  const aie::vector<float, VL> one = aie::broadcast<float, VL>(1.0f);
  const aie::vector<float, VL> two = aie::broadcast<float, VL>(2.0f);
  aie::vector<float, VL> az = aie::abs(z);
  aie::vector<float, VL> arg =
      aie::min(aie::mul(az, aie::broadcast<float, VL>(2.0f * LOG2E)).to_vector<float>(),
               aie::broadcast<float, VL>(100.0f));
  aie::vector<float, VL> e = aie::add(aie::mul(arg, arg).to_vector<float>(), one); // inline stand-in
  aie::vector<float, VL> r = aie::inv(aie::add(e, one));
  aie::vector<float, VL> t = aie::sub(one, aie::mul(two, r).to_vector<float>());
  return aie::select(t, aie::sub(zero, t), aie::lt(z, zero));
}

// The shipped epilogue's bf16 cube, factored out so gelu_hw and gelu_sw differ in
// EXACTLY one call. Returns `inner` = c0*(x + c1*x^3) as f32 (what the epilogue
// feeds tanh), computed through the same bf16 narrows the epilogue uses.
static inline aie::vector<float, VL> gelu_inner_bf16(aie::vector<float, VL> accf,
                                                     aie::vector<bfloat16, VL> &xv_out) {
  const aie::vector<bfloat16, VL> c0 = aie::broadcast<bfloat16, VL>(0.7978845608f); // sqrt(2/pi)
  const aie::vector<bfloat16, VL> c1 = aie::broadcast<bfloat16, VL>(0.044715f);
  aie::accum<accfloat, VL> a;
  a.from_vector(accf);
  aie::vector<bfloat16, VL> xv = a.to_vector<bfloat16>();
  aie::vector<bfloat16, VL> x2 = aie::mul(xv, xv);
  aie::vector<bfloat16, VL> x3 = aie::mul(x2, xv);
  aie::vector<bfloat16, VL> c1x3 = aie::mul(c1, x3);
  aie::vector<bfloat16, VL> inner_b = aie::add(xv, c1x3);
  xv_out = xv;
  return aie::mul(c0, inner_b).to_vector<float>();
}

extern "C" void tanh_ab(float *restrict in, float *restrict out) {
  event0();
  float *raw = out + 0 * TANHAB_N;
  float *hwtanh = out + 1 * TANHAB_N;
  float *swtanh = out + 2 * TANHAB_N;
  float *gelu_hw = out + 3 * TANHAB_N;
  float *gelu_sw = out + 4 * TANHAB_N;

  // conv_even for the whole probe: the shipped epilogue swaps to it, and crRnd is one
  // sticky per-core register, so leaving it at the hardware default would measure a
  // chain the kernel no longer runs. Restored on exit (the matmul shares the register).
  const auto saved_rounding = aie::swap_rounding(aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, VL> one_b = aie::broadcast<bfloat16, VL>(1.0f);
  const aie::vector<bfloat16, VL> half_b = aie::broadcast<bfloat16, VL>(0.5f);
  const aie::vector<float, VL> one_f = aie::broadcast<float, VL>(1.0f);
  const aie::vector<float, VL> half_f = aie::broadcast<float, VL>(0.5f);

  for (int i = 0; i < TANHAB_N; i += VL) {
    aie::vector<float, VL> x = aie::load_v<VL>(in + i);
    aie::store_v(raw + i, x);

    // (1) the hw SFU LUT, called exactly as mm_gelu_epilogue_f32o calls it:
    // template arg bfloat16, a vector<float,16> in, a vector<bfloat16,16> out.
    aie::vector<bfloat16, VL> hw = aie::tanh<bfloat16>(x);
    aie::store_v(hwtanh + i, aie::add(aie::zeros<accfloat, VL>(), hw).to_vector<float>());

    // (2) the software candidate, f32 in and out.
#if TANHAB_NOSW == 1
    aie::store_v(swtanh + i, aie::zeros<float, VL>());
#elif TANHAB_NOSW == 2
    aie::store_v(swtanh + i, exp2f_vec(x));   // poly alone, no inv/abs/select
#elif TANHAB_NOSW == 4
    aie::store_v(swtanh + i, swtanh_inline_surrogate(x));
#else                                          // 0 and 3 both take the full form
    aie::store_v(swtanh + i, swtanh_vec(x));
#endif

    // (3)/(4) the two epilogue chains. Same bf16 cube, same conv_even, one call apart.
    aie::vector<bfloat16, VL> xv;
    aie::vector<float, VL> inner = gelu_inner_bf16(x, xv);

    aie::vector<bfloat16, VL> t_hw = aie::tanh<bfloat16>(inner);
    aie::vector<bfloat16, VL> tp1 = aie::add(t_hw, one_b);
    aie::vector<bfloat16, VL> xt = aie::mul(xv, tp1);
    aie::vector<bfloat16, VL> gx = aie::mul(half_b, xt);
    aie::accum<accfloat, VL> oacc;
    oacc.from_vector(gx);
    aie::store_v(gelu_hw + i, oacc.to_vector<float>());

    // gelu_sw keeps the tail in f32 as well -- swtanh already returns f32, so
    // narrowing (1+t) back to bf16 purely to match the hw chain would throw away
    // the precision this arm exists to test. xv (bf16 x) is reused so the two arms
    // share an identical x; only the transcendental differs.
#if TANHAB_NOSW   // 1, 2 and 3 all leave gelu_sw out; only mode 0 builds both call sites
    aie::store_v(gelu_sw + i, aie::zeros<float, VL>());
#else
    aie::vector<float, VL> t_sw = swtanh_vec(inner);
    aie::vector<float, VL> xf = aie::add(aie::zeros<accfloat, VL>(), xv).to_vector<float>();
    aie::vector<float, VL> g_sw =
        aie::mul(aie::mul(half_f, xf).to_vector<float>(),
                 aie::add(one_f, t_sw)).to_vector<float>();
    aie::store_v(gelu_sw + i, g_sw);
#endif
  }
  aie::set_rounding(saved_rounding);
  event1();
}
