//===- silu_brick.cc ------------------------------------------*- C++ -*-===//
//
// Device-side SiLU brick (Conformer conv-module post-dwconv activation).
//   out[i] = silu(in[i]) = in[i] * sigmoid(in[i])
// Per-row: one row of `cols` f32 in -> f32 out. SEPARATE single-op-loop brick (NOT a
// dwconv epilogue) -- immune to the fused-epilogue per-channel-loop miscompile (KB log
// dwconv-fused-epilogue-alt-channel-miscompile). Fed the dwconv output host-side.
//
// SILU_MODE (opt-in probes; default 0 = the SHIPPED brick, behavior unchanged):
//   0  hybrid bf16-tanh sigmoid (tanh<bfloat16>, everything else f32). WORKS, RU 8.5
//      (bf16-tanh precision floor). The committed default.
//   1  EXP2-ONLY probe: out = 2^(-|x|*log2e). Isolates the software f32 exp2f in this
//      kernel context (does exp2f alone hang here?).
//   2  FULL f32-poly sigmoid (exact, no bf16 LUT): sigmoid via exp2f + inv + 1 Newton
//      step. This is the exact-8.2 path AND the candidate CRVO-7950 emulation, but it
//      HANGS on device ("kernel run did not complete") -- bisect target.
//
// Runtime ABI (mirrors glu_iron / ctx_ln): 1=instr, 3=in, 4=out, 5=tmp, 6=ctrl, 7=trace.
// TRACKED COPY -- installed into mlir-aie/.../layernorm by sync_kernels.sh.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef SILU_MODE
#define SILU_MODE 0
#endif

#if SILU_MODE == 7
// MINIMAL HANG REPRO (no exp2f): a noinline vector helper + a value held live across the
// call + a reciprocal/select composition after it. Reproduces the same 23-slot spill
// pressure as the exp2f path -> isolates the codegen bug from exp2f/int<->float.
static __attribute__((noinline)) ::aie::vector<float, 16>
heavy16(::aie::vector<float, 16> x) {
  ::aie::vector<float, 16> a = ::aie::mul(x, x).to_vector<float>();
  ::aie::vector<float, 16> b = ::aie::add(a, x);
  ::aie::vector<float, 16> c = ::aie::mul(b, a).to_vector<float>();
  return ::aie::add(c, b);
}
#endif

#if SILU_MODE >= 1 && SILU_MODE != 7
// SOFTWARE f32 2^x for x <= 0 (the hw aie::exp2 is a bf16-output LUT, ~2-4% inaccurate;
// this poly is ~1e-4). From relpos_mha.cc exp2f_vec, fixed at 16 lanes. NOINLINE is
// LOAD-BEARING: inlining it makes Peano -O2 miscompile to NaN (register-pressure codegen bug).
static __attribute__((noinline)) ::aie::vector<float, 16>
exp2f_neg16(::aie::vector<float, 16> x) {
  x = ::aie::max(x, ::aie::broadcast<float, 16>(-100.0f));
  ::aie::vector<int32_t, 16> ki = ::aie::to_fixed<int32_t>(x);
  ::aie::vector<float, 16> kf = ::aie::to_float<float>(ki);
  ::aie::vector<int32_t, 16> one = ::aie::broadcast<int32_t, 16>(1);
  ::aie::vector<int32_t, 16> zero = ::aie::broadcast<int32_t, 16>(0);
  ki = ::aie::sub(ki, ::aie::select(zero, one, ::aie::lt(x, kf)));
  ::aie::vector<float, 16> f = ::aie::sub(x, ::aie::to_float<float>(ki));
  ::aie::vector<float, 16> p = ::aie::broadcast<float, 16>(0.0013333558f);
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(0.0096181291f));
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(0.0555041087f));
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(0.2402265069f));
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(0.6931471805f));
  p = ::aie::add(::aie::mul(p, f).to_vector<float>(), ::aie::broadcast<float, 16>(1.0f));
  ::aie::vector<int32_t, 16> ebits =
      ::aie::upshift(::aie::add(ki, ::aie::broadcast<int32_t, 16>(127)), 23);
  ::aie::vector<float, 16> p2k = ebits.cast_to<float>();
  return ::aie::mul(p, p2k).to_vector<float>();
}
#endif

template <int N>
void silu_row(const float *restrict input, float *restrict output, int32_t cols) {
  event0();
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
#if SILU_MODE == 0
  // --- shipped hybrid: bf16 tanh, everything else f32 (RU 8.5) ---
  const ::aie::vector<float, N> halff = ::aie::broadcast<float, N>(0.5f);
  const ::aie::vector<float, N> onef = ::aie::broadcast<float, N>(1.0f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> half_x = ::aie::mul(xv, halff);
    ::aie::vector<bfloat16, N> th_b = ::aie::tanh<bfloat16>(half_x);
    ::aie::accum<accfloat, N> tacc;
    tacc.from_vector(th_b);
    ::aie::vector<float, N> th = tacc.template to_vector<float>();
    ::aie::vector<float, N> tp1 = ::aie::add(th, onef);
    ::aie::vector<float, N> sig = ::aie::mul(tp1, halff);
    ::aie::vector<float, N> outv = ::aie::mul(xv, sig);
    ::aie::store_v(output + i, outv);
  }
#elif SILU_MODE == 1
  // --- EXP2-ONLY probe: out = 2^(-|x|*log2e). isolates exp2f in this kernel ---
  const ::aie::vector<float, N> zero = ::aie::broadcast<float, N>(0.0f);
  const ::aie::vector<float, N> neg_log2e = ::aie::broadcast<float, N>(-1.44269504089f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> absx = ::aie::max(xv, ::aie::sub(zero, xv));
    ::aie::vector<float, N> s = exp2f_neg16(::aie::mul(absx, neg_log2e).template to_vector<float>());
    ::aie::store_v(output + i, s);
  }
#elif SILU_MODE == 3
  // --- INV-ISOLATION probe: out = 1/(1+s), s=exp2f(...). tests aie::inv (no Newton/select) ---
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> zero = ::aie::broadcast<float, N>(0.0f);
  const ::aie::vector<float, N> neg_log2e = ::aie::broadcast<float, N>(-1.44269504089f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> absx = ::aie::max(xv, ::aie::sub(zero, xv));
    ::aie::vector<float, N> s = exp2f_neg16(::aie::mul(absx, neg_log2e).template to_vector<float>());
    ::aie::vector<float, N> r = ::aie::inv(::aie::add(one, s));
    ::aie::store_v(output + i, r);
  }
#elif SILU_MODE == 4
  // --- FULL f32-poly sigmoid, NO Newton step (aie::inv is already f32-accurate). ---
  // silu = x*sigmoid(x); sigmoid = (x<0 ? s : 1)/(1+s), s = 2^(-|x|*log2e).
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> zero = ::aie::broadcast<float, N>(0.0f);
  const ::aie::vector<float, N> neg_log2e = ::aie::broadcast<float, N>(-1.44269504089f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> absx = ::aie::max(xv, ::aie::sub(zero, xv));
    ::aie::vector<float, N> s = exp2f_neg16(::aie::mul(absx, neg_log2e).template to_vector<float>());
    ::aie::vector<float, N> r = ::aie::inv(::aie::add(one, s));
    ::aie::vector<float, N> num = ::aie::select(one, s, ::aie::lt(xv, zero));
    ::aie::vector<float, N> sig = ::aie::mul(num, r).template to_vector<float>();
    ::aie::vector<float, N> outv = ::aie::mul(xv, sig).template to_vector<float>();
    ::aie::store_v(output + i, outv);
  }
#elif SILU_MODE == 5
  // --- FULL exact f32 sigmoid + Newton, but UNROLL-DISABLED (the relpos precedent:
  // exp2f-heavy bodies miscompile/hang under Peano -O2 loop unrolling). ---
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> two = ::aie::broadcast<float, N>(2.0f);
  const ::aie::vector<float, N> zero = ::aie::broadcast<float, N>(0.0f);
  const ::aie::vector<float, N> neg_log2e = ::aie::broadcast<float, N>(-1.44269504089f);
  #pragma clang loop unroll(disable)
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> absx = ::aie::max(xv, ::aie::sub(zero, xv));
    ::aie::vector<float, N> s = exp2f_neg16(::aie::mul(absx, neg_log2e).template to_vector<float>());
    ::aie::vector<float, N> denom = ::aie::add(one, s);
    ::aie::vector<float, N> r0 = ::aie::inv(denom);
    ::aie::vector<float, N> dr0 = ::aie::mul(denom, r0).template to_vector<float>();
    ::aie::vector<float, N> r = ::aie::mul(r0, ::aie::sub(two, dr0)).template to_vector<float>();
    ::aie::vector<float, N> num = ::aie::select(one, s, ::aie::lt(xv, zero));
    ::aie::vector<float, N> sig = ::aie::mul(num, r).template to_vector<float>();
    ::aie::vector<float, N> outv = ::aie::mul(xv, sig).template to_vector<float>();
    ::aie::store_v(output + i, outv);
  }
#elif SILU_MODE == 6
  // --- TWO-PASS exact f32 silu: NOTHING held across the noinline exp2f call. ---
  // pass1: s = 2^(-|x|*log2e) -> scratch (xv used then dead before the call returns).
  // pass2: reload x + s, do reciprocal+select+silu -- NO call in this body (no spill-around-call).
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> two = ::aie::broadcast<float, N>(2.0f);
  const ::aie::vector<float, N> zero = ::aie::broadcast<float, N>(0.0f);
  const ::aie::vector<float, N> neg_log2e = ::aie::broadcast<float, N>(-1.44269504089f);
  alignas(64) float scratch[400]; // cols baked at 400 (DW_T)
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> absx = ::aie::max(xv, ::aie::sub(zero, xv));
    ::aie::vector<float, N> s = exp2f_neg16(::aie::mul(absx, neg_log2e).template to_vector<float>());
    ::aie::store_v(scratch + i, s);
  }
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> s = ::aie::load_v<N>(scratch + i);
    ::aie::vector<float, N> denom = ::aie::add(one, s);
    ::aie::vector<float, N> r0 = ::aie::inv(denom);
    ::aie::vector<float, N> dr0 = ::aie::mul(denom, r0).template to_vector<float>();
    ::aie::vector<float, N> r = ::aie::mul(r0, ::aie::sub(two, dr0)).template to_vector<float>();
    ::aie::vector<float, N> num = ::aie::select(one, s, ::aie::lt(xv, zero));
    ::aie::vector<float, N> sig = ::aie::mul(num, r).template to_vector<float>();
    ::aie::vector<float, N> outv = ::aie::mul(xv, sig).template to_vector<float>();
    ::aie::store_v(output + i, outv);
  }
#elif SILU_MODE == 7
  // --- MINIMAL HANG REPRO (dummy heavy16, no exp2f) ---
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> two = ::aie::broadcast<float, N>(2.0f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> s = heavy16(xv);                    // xv held across the call
    ::aie::vector<float, N> r0 = ::aie::inv(::aie::add(one, s));
    ::aie::vector<float, N> num = ::aie::select(one, s, ::aie::lt(xv, two));
    ::aie::vector<float, N> sig = ::aie::mul(num, r0).template to_vector<float>();
    ::aie::vector<float, N> outv = ::aie::mul(xv, sig).template to_vector<float>();
    ::aie::store_v(output + i, outv);
  }
#elif SILU_MODE == 8
  // --- LOW-PRESSURE inline Horner-poly tanh sigmoid (no call, no exp2f, no int-bit tricks) ---
  // tanh(t) ~ t*Horner(t^2), deg-11 odd fit on [-3,3] (clamped). silu rel-L2 ~1e-3 (beats bf16-tanh
  // 6.7e-3). Horner reuses one accumulator -> few live vectors -> small frame -> avoids the
  // cross-call spill fault AND the spill/objectfifo-buffer collision. sigmoid=0.5*(1+tanh(x/2)).
  const ::aie::vector<float, N> half = ::aie::broadcast<float, N>(0.5f);
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> c3 = ::aie::broadcast<float, N>(3.0f);
  const ::aie::vector<float, N> cm3 = ::aie::broadcast<float, N>(-3.0f);
  const ::aie::vector<float, N> a0 = ::aie::broadcast<float, N>(0.98904683f);
  const ::aie::vector<float, N> a1 = ::aie::broadcast<float, N>(-0.28778211f);
  const ::aie::vector<float, N> a2 = ::aie::broadcast<float, N>(0.072722501f);
  const ::aie::vector<float, N> a3 = ::aie::broadcast<float, N>(-0.01140079f);
  const ::aie::vector<float, N> a4 = ::aie::broadcast<float, N>(0.00095127754f);
  const ::aie::vector<float, N> a5 = ::aie::broadcast<float, N>(-3.1999843e-05f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> t = ::aie::min(::aie::max(::aie::mul(xv, half).template to_vector<float>(), cm3), c3);
    ::aie::vector<float, N> u = ::aie::mul(t, t).template to_vector<float>();
    ::aie::vector<float, N> acc = a5;
    acc = ::aie::add(::aie::mul(acc, u).template to_vector<float>(), a4);
    acc = ::aie::add(::aie::mul(acc, u).template to_vector<float>(), a3);
    acc = ::aie::add(::aie::mul(acc, u).template to_vector<float>(), a2);
    acc = ::aie::add(::aie::mul(acc, u).template to_vector<float>(), a1);
    acc = ::aie::add(::aie::mul(acc, u).template to_vector<float>(), a0);
    ::aie::vector<float, N> th = ::aie::mul(t, acc).template to_vector<float>();      // tanh(x/2)
    ::aie::vector<float, N> sig = ::aie::mul(half, ::aie::add(one, th)).template to_vector<float>();
    ::aie::vector<float, N> outv = ::aie::mul(xv, sig).template to_vector<float>();
    ::aie::store_v(output + i, outv);
  }
#elif SILU_MODE == 2
  // --- FULL f32-poly sigmoid (exact; HANGS -- bisect target) ---
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> two = ::aie::broadcast<float, N>(2.0f);
  const ::aie::vector<float, N> zero = ::aie::broadcast<float, N>(0.0f);
  const ::aie::vector<float, N> neg_log2e = ::aie::broadcast<float, N>(-1.44269504089f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> absx = ::aie::max(xv, ::aie::sub(zero, xv));
    ::aie::vector<float, N> s = exp2f_neg16(::aie::mul(absx, neg_log2e).template to_vector<float>());
    ::aie::vector<float, N> denom = ::aie::add(one, s);
    ::aie::vector<float, N> r0 = ::aie::inv(denom);
    ::aie::vector<float, N> dr0 = ::aie::mul(denom, r0).template to_vector<float>();
    ::aie::vector<float, N> r = ::aie::mul(r0, ::aie::sub(two, dr0)).template to_vector<float>();
    ::aie::vector<float, N> num = ::aie::select(one, s, ::aie::lt(xv, zero));
    ::aie::vector<float, N> sig = ::aie::mul(num, r).template to_vector<float>();
    ::aie::vector<float, N> outv = ::aie::mul(xv, sig).template to_vector<float>();
    ::aie::store_v(output + i, outv);
  }
#endif
  event1();
}

extern "C" {
void silu_row(float *input, float *output, int32_t cols) {
  silu_row<16>(input, output, cols);
}
}
