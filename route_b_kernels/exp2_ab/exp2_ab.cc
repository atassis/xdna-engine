//===- exp2_ab.cc -------------------------------------------------*- C++ -*-===//
//
// A/B device probe: hw aie::exp2<bfloat16> LUT vs the relpos_mha software f32
// degree-5 poly (exp2f_vec), both against a float64 numpy 2**x host reference.
//
// PURPOSE: relpos_mha.cc (route_b_kernels/relpos_mha/relpos_mha.cc:117-122) carries
// a comment claiming the hw exp2 LUT is "~2-4% inaccurate" on aie2p and the sw poly
// is "~1e-4 accurate (full 2^x rel-err 8.5e-5)". That claim is from a previous
// session and is about to become the justification for an upstream mlir-aie PR, so
// it must be re-measured on THIS toolchain pin / THIS device, not trusted. This
// kernel produces, for the SAME f32 input vector, four aligned output streams so a
// single host run can score both paths unambiguously:
//   raw   : the input, passed through (alignment / sanity)
//   hw2x  : f32(aie::exp2<bfloat16>(x))            -- the hw LUT computing 2^x
//   sw2x  : exp2f_vec(x)                            -- the sw poly computing 2^x
//   hwex  : f32(bf16_exp.cc's EXACT pipeline(x))    -- bonus, see note below
//
// DESIGN NOTE on "fed the same way bf16_exp.cc feeds it" (upstream
// mlir-aie/aie_kernels/aie2p/bf16_exp.cc): that kernel computes e^x = 2^(log2e*x)
// by pre-multiplying its bf16 input by log2e before calling aie::exp2<bfloat16>.
// If hw2x reproduced that log2e pre-multiply, it would compute e^x, not 2^x, and
// comparing e^x against the requested float64 "2**x" reference would report a huge
// "error" that is really just e^x != 2^x -- not a measurement of the LUT's own
// approximation error. So hw2x calls aie::exp2<bfloat16> DIRECTLY on the f32 input
// vector, matching upstream's exact call shape (template arg bfloat16, a
// vector<float,16> in, a vector<bfloat16,16> out -- literally
// `aie::exp2<bfloat16>(exp_in.to_vector<float>())` in bf16_exp.cc, and
// `aie::exp2<bfloat16>(sl)` in relpos_mha.cc's own production softmax) without the
// log2e step, so it is testing the SAME hw primitive on the SAME target function
// (2^x) as sw2x/exp2f_vec. hwex is the bonus column that DOES reproduce
// bf16_exp.cc's literal pipeline byte-for-byte (bf16-cast input, log2e-multiply,
// accfloat-accumulate, to_vector<float>, exp2<bfloat16>) for a direct e^x fidelity
// check against float64 np.exp(x); the host runner scores it separately, not
// blended into the primary hw2x/sw2x/2**x table.
//
// exp2f_vec below is a VERBATIM copy (including __attribute__((noinline))) of
// route_b_kernels/relpos_mha/relpos_mha.cc L123-145. Do not hand-edit it here
// without re-diffing against that source of truth -- this file exists precisely so
// relpos_mha.cc itself is never touched by this measurement.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr float LOG2E = 1.4426950408889634f;
static constexpr int VL = 16;

// Input/output vector length, compile-time (matches -DEXP2AB_N in the Makefile).
// Must be a multiple of VL. out[] is laid out [raw|hw2x|sw2x|hwex], each EXP2AB_N.
// NOTE: out[] is 4*EXP2AB_N f32 words in one objectFifo buffer; the aie2p BD length
// cap is 16383 32-bit words per descriptor, so 4*EXP2AB_N must stay <= 16383
// (EXP2AB_N <= 4095). 4080 = 255*16 is the largest VL-multiple that fits with margin.
#ifndef EXP2AB_N
#define EXP2AB_N 1024
#endif
static_assert(EXP2AB_N % VL == 0, "EXP2AB_N must be a multiple of the vector width");

// ---------------------------------------------------------------------------
// VERBATIM copy of relpos_mha.cc's exp2f_vec (L123-145, incl. the load-bearing
// noinline -- see that file's header comment for why). SOFTWARE f32 2^x (x <= 0
// nominally; clamped at -100 same as upstream).
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

// ---------------------------------------------------------------------------
// bf16_exp.cc's EXACT pipeline (mlir-aie/aie_kernels/aie2p/bf16_exp.cc), inlined
// here so the a/b probe can reproduce e^x with byte-for-byte fidelity to the
// upstream reference kernel (bonus hwex column). Input is rounded to bf16 first
// (matching bf16_exp.cc's `bfloat16 *a_in` signature), multiplied by log2e in
// bf16, accumulated in accfloat, widened to a float vector, then fed to the same
// hw LUT.
// ---------------------------------------------------------------------------
static inline aie::vector<float, VL> hw_ex_vec(aie::vector<float, VL> xf32) {
  // widen f32 -> bf16 -> (accum) -> f32 the same way cast_f32_bf16.cc does the
  // narrow half of this round-trip: an accfloat accumulator, not a raw cast.
  aie::accum<accfloat, VL> narrow_acc;
  narrow_acc.from_vector(xf32);
  aie::vector<bfloat16, VL> input_bf16 = narrow_acc.to_vector<bfloat16>();

  aie::vector<bfloat16, VL> log2e_vec = aie::broadcast<bfloat16, VL>((bfloat16)LOG2E);
  aie::accum<accfloat, VL> exp_in = aie::mul(input_bf16, log2e_vec);
  aie::vector<bfloat16, VL> exp_val = aie::exp2<bfloat16>(exp_in.to_vector<float>());
  // widen bf16 -> f32 for output, via the same zero-add idiom relpos_mha.cc uses
  // (relpos_scores_softmax, "widen bf16 -> f32 accumulate").
  return aie::add(aie::zeros<accfloat, VL>(), exp_val).to_vector<float>();
}

extern "C" void exp2_ab(float *restrict in, float *restrict out) {
  event0();
  float *raw = out + 0 * EXP2AB_N;
  float *hw2x = out + 1 * EXP2AB_N;
  float *sw2x = out + 2 * EXP2AB_N;
  float *hwex = out + 3 * EXP2AB_N;
  for (int i = 0; i < EXP2AB_N; i += VL) {
    aie::vector<float, VL> x = aie::load_v<VL>(in + i);
    aie::store_v(raw + i, x);

    // (1) hw path: aie::exp2<bfloat16> fed a float32 vector directly -- SAME call
    // shape as bf16_exp.cc's `aie::exp2<bfloat16>(exp_in.to_vector<float>())` and
    // relpos_mha.cc's own `aie::exp2<bfloat16>(sl)`. Computes 2^x (see header note).
    aie::vector<bfloat16, VL> hw = aie::exp2<bfloat16>(x);
    aie::vector<float, VL> hw_f32 =
        aie::add(aie::zeros<accfloat, VL>(), hw).to_vector<float>(); // widen bf16->f32
    aie::store_v(hw2x + i, hw_f32);

    // (2) sw path: the verbatim relpos_mha.cc poly, direct 2^x contract (no
    // caller-side log2e prescale here -- matches how exp2f_vec itself is defined).
    aie::store_v(sw2x + i, exp2f_vec(x));

    // (3) bonus: bf16_exp.cc's literal e^x pipeline (scored vs np.exp(x) on host).
    aie::store_v(hwex + i, hw_ex_vec(x));
  }
  event1();
}
