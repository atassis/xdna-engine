//===- ra_spill_repro.cc --------------------------------------*- C++ -*-===//
//
// MINIMAL, NON-PARAKEET reproducer for the aie2p (Peano/llvm-aie) register-
// allocation / spill-around-noinline-call codegen defect that gates the resident
// conv module's DEFAULT flip.
//
// Full characterization: xdna-engine-private journal log
//   dwconv-fused-epilogue-alt-channel-miscompile.md ("UNIFICATION" section).
//
// The defect (one bug, two faces) on an objectfifo-driven per-tile loop with a
// HEAVY f32 body:
//   * HANG    when a vector value is held live ACROSS a `noinline` call -- the
//             value must be spilled around the call, and that spill miscompiles.
//   * CORRUPT alternate (even) tile iterations otherwise (even == ping-pong
//             buffer 0), or emit a wrong loop-invariant constant register.
//
// This file carries NO Parakeet weights and NO exp2f -- just a `noinline` heavy
// f32 vector helper and one live-across-call value, the smallest shape that still
// forces a spill-around-`jl`. It is SELF-VALIDATING via RA_HOLD:
//   RA_HOLD=1 (default)  xv held live across heavy16() -> spill-around-call  -> REPRO
//   RA_HOLD=0 (control)  nothing held across heavy16() -> no cross-call spill -> CLEAN
// Build both, run both: only the RA_HOLD=1 variant should hang / corrupt evens.
//
// Runtime ABI (mirrors silu_iron / glu_iron): opcode 3, in=gid3, out=gid4.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef RA_HOLD
#define RA_HOLD 1
#endif

// A `noinline` heavy f32 vector helper. Forces an out-of-line `jl` call taking a
// 512-bit vector in a vector reg (x0) and returning via a hidden pointer (p0).
// Heavy enough to raise register pressure so the caller MUST spill live values
// around the call. Byte-for-byte the SILU_MODE==7 helper in silu_brick.cc.
static __attribute__((noinline)) ::aie::vector<float, 16>
heavy16(::aie::vector<float, 16> x) {
  ::aie::vector<float, 16> a = ::aie::mul(x, x).to_vector<float>();
  ::aie::vector<float, 16> b = ::aie::add(a, x);
  ::aie::vector<float, 16> c = ::aie::mul(b, a).to_vector<float>();
  return ::aie::add(c, b);
}

template <int N>
void ra_spill_row(const float *restrict input, float *restrict output,
                  int32_t cols) {
  event0();
  const ::aie::vector<float, N> one = ::aie::broadcast<float, N>(1.0f);
  const ::aie::vector<float, N> two = ::aie::broadcast<float, N>(2.0f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    ::aie::vector<float, N> s = heavy16(xv); // out-of-line jl call
#if RA_HOLD
    // xv held LIVE ACROSS the call (used below) -> spill-around-call = the trigger.
    ::aie::vector<float, N> r0 = ::aie::inv(::aie::add(one, s));
    ::aie::vector<float, N> num = ::aie::select(one, s, ::aie::lt(xv, two));
    ::aie::vector<float, N> sig = ::aie::mul(num, r0).template to_vector<float>();
    ::aie::vector<float, N> outv = ::aie::mul(xv, sig).template to_vector<float>();
    ::aie::store_v(output + i, outv);
#else
    // CONTROL: xv dead before the call returns -> nothing to spill across the call.
    ::aie::vector<float, N> r0 = ::aie::inv(::aie::add(one, s));
    ::aie::store_v(output + i, r0);
#endif
  }
  event1();
}

extern "C" {
void ra_spill_row(float *input, float *output, int32_t cols) {
  ra_spill_row<16>(input, output, cols);
}
}
