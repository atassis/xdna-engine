//===- gdn_gates.cc ----------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Gate activation for the gated delta rule (op-type `gates{delta}`), the producer the gatedeltanet
// brick expects its already-activated (alpha, beta) from:
//
//   alpha = exp(neg_a * softplus(a + dt_bias))      neg_a = -exp(A_log), folded on the host
//   beta  = sigmoid(b) = exp(-softplus(-b))
//
// All in f32 vector arithmetic. exp is exp2f_vec's polynomial (mlir-aie
// aie_kernels/aie2p/exp2f_vec.cc); softplus(u) = max(u, 0) + log1p(exp(-|u|)) with log1p a degree-8
// fit on [0, 1] (max rel error 1.9e-7 in f32 Horner). Writing sigmoid through softplus avoids a
// reciprocal. The SFU tanh/exp2 LUT is not used: alpha compounds through the recurrent state
// (kb/aie-tanh-is-the-same-coarse-sfu-lut-as-exp2 measured that LUT at 5e-3..5e-2).
//
// ab = [a | b] bf16 (the projection's output), params = [neg_a | dt_bias] f32, gates = interleaved
// (alpha, beta) f32 per head -- the gatedeltanet brick's [T,2] layout. N is compile-time.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GG_N
#define GG_N 32
#endif

static constexpr unsigned V = 16;
static_assert(GG_N % V == 0, "head count must be a multiple of the 16-lane f32 vector");
using vf = aie::vector<float, V>;

static inline vf vmul(vf a, vf b) { return aie::mul(a, b).template to_vector<float>(); }
static inline vf vfma(vf a, vf b, float c) { return aie::add(vmul(a, b), aie::broadcast<float, V>(c)); }

// exp2f_vec (mlir-aie #3467): 2^k from the exponent field, degree-6 polynomial on the fraction.
static inline vf exp2v(vf x) {
  x = aie::max(x, aie::broadcast<float, V>(-111.0f));
  x = aie::min(x, aie::broadcast<float, V>(127.999f));
  aie::vector<int32_t, V> ki = aie::to_fixed<int32_t>(x);
  ki = aie::sub(ki, aie::select(aie::broadcast<int32_t, V>(0), aie::broadcast<int32_t, V>(1),
                                aie::lt(x, aie::to_float<float>(ki))));
  vf f = aie::sub(x, aie::to_float<float>(ki));
  vf p = aie::broadcast<float, V>(0.0013333558f);
  p = vfma(p, f, 0.0096181291f);
  p = vfma(p, f, 0.0555041087f);
  p = vfma(p, f, 0.2402265069f);
  p = vfma(p, f, 0.6931471805f);
  p = vfma(p, f, 1.0f);
  aie::vector<int32_t, V> e = aie::upshift(aie::add(ki, aie::broadcast<int32_t, V>(127)), 23);
  return vmul(p, e.template cast_to<float>());
}

// Peano cannot legalize G_FNEG on <16 x float> (aie::neg), so negation is a subtract from zero.
static inline vf negv(vf x) { return aie::sub(aie::broadcast<float, V>(0.0f), x); }

static inline vf expv(vf x) { return exp2v(vmul(x, aie::broadcast<float, V>(1.4426950409f))); }

static inline vf log1pv(vf t) {  // t in [0, 1]
  vf p = aie::broadcast<float, V>(0.005253457929939032f);
  p = vfma(p, t, -0.02958850748836994f);
  p = vfma(p, t, 0.07836166769266129f);
  p = vfma(p, t, -0.13674770295619965f);
  p = vfma(p, t, 0.19111430644989014f);
  p = vfma(p, t, -0.24844369292259216f);
  p = vfma(p, t, 0.33319270610809326f);
  p = vfma(p, t, -0.49999502301216125f);
  p = vfma(p, t, 1.0f);
  return vmul(t, p);
}

static inline vf softplusv(vf u) {
  vf neg_abs = negv(aie::abs(u));
  return aie::add(aie::max(u, aie::broadcast<float, V>(0.0f)), log1pv(expv(neg_abs)));
}

static inline vf load_bf16_as_f32(const bfloat16 *p) {
  aie::accum<accfloat, V> acc;
  acc.from_vector(aie::load_v<V>(p));
  return acc.template to_vector<float>();
}

template <unsigned N>
static inline void gdn_gates_core(const bfloat16 *__restrict ab, const float *__restrict params,
                                  float *__restrict gates) {
  event0();
  for (unsigned h = 0; h < N; h += V) {
    vf a = load_bf16_as_f32(&ab[h]);
    vf b = load_bf16_as_f32(&ab[N + h]);
    vf neg_a = aie::load_v<V>(&params[h]);
    vf dtb = aie::load_v<V>(&params[N + h]);
    vf alpha = expv(vmul(neg_a, softplusv(aie::add(a, dtb))));
    vf beta = expv(negv(softplusv(negv(b))));
    auto [lo, hi] = aie::interleave_zip(alpha, beta, 1);
    aie::store_v(&gates[2 * h], lo);
    aie::store_v(&gates[2 * h + V], hi);
  }
  event1();
}

extern "C" {

void gdn_gates(bfloat16 *ab, float *params, float *gates) { gdn_gates_core<GG_N>(ab, params, gates); }
}
