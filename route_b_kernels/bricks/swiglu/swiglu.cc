//===- swiglu.cc ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// SwiGLU brick [group: act]:
//   out[i] = silu(gate[i]) * up[i] = gate[i] * sigmoid(gate[i]) * up[i]
//
// GENERIC op-TYPE against the resident [tile, D] stream contract -- this is
// NOT a per-model clone. The op reads two same-shaped f32 operand streams
// (gate, up -- typically the two halves of a fused up-projection GEMM output,
// [tile, D] each) and writes one f32 [tile, D] stream. D/tile-count are
// caller-supplied at compile time via the SWIGLU_SIZE macro (element count of
// the flat streamed region = tile * D); the elementwise body itself is
// layout-independent (mmul-blocked vs row-major is moot for a per-element op,
// same reasoning as mm_silu_epilogue.cc / silu_brick.cc), so this same object
// serves any (tile, D) whose product is a multiple of 16 lanes.
//
// SIGMOID PATH: uses the XDNA2 hardware exp2 SFU (aie::exp2<bfloat16>, a
// bf16-output LUT for 2^x) rather than the tanh identity, per the brick spec.
//   sigmoid(x) = 1 / (1 + 2^(-x*log2(e)))
// The tanh path (mm_silu_epilogue.cc, silu_brick.cc SILU_MODE==0) is the
// shipped default elsewhere in this tree because it is proven WER-neutral;
// this brick intentionally exercises the OTHER hardware SFU (exp2) to give
// the brick catalog both activation-SFU primitives. Precision follows the
// f32o_hiprec pattern from mm_silu_epilogue.cc: keep x and the final
// multiplies in f32, only the SFU output (bounded in (0,1]) is bf16, and the
// reciprocal denominator is refined with one Newton step in f32 (mirrors
// silu_brick.cc SILU_MODE==5/6, the exact-but-safe f32 sigmoid).
//
// KNOWN TOOLCHAIN HAZARD (see silu_brick.cc, dwconv-fused-epilogue-alt-
// channel-miscompile KB note): a *software* f32 exp2f polynomial held live
// across a noinline call has caused Peano -O2 register-pressure miscompiles
// and worker-stack overflows on this target in OTHER kernels. This brick
// sidesteps that class of bug entirely by using the HARDWARE exp2 SFU
// (aie::exp2, a single intrinsic, no noinline helper, no software polynomial,
// nothing held live across a call) -- there is no known instance of this
// hazard hitting the hardware-SFU intrinsics themselves (tanh/exp2/invsqrt),
// only the earlier from-scratch *software* exp2f polynomial in silu_brick.cc.
// Flagging this reasoning explicitly for the leaf that does the on-device
// pass: if a hang/NaN DOES show up here, bisect against that KB note first.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// Elementwise SwiGLU over a flat `size`-element [tile, D] region:
//   out = gate * sigmoid(gate) * up,   all f32 in/out.
// sigmoid via the hardware exp2 SFU: sigmoid(x) = 1 / (1 + 2^(-x*log2e)).
template <int size>
static inline void swiglu(const float *__restrict pGate_in,
                          const float *__restrict pUp_in,
                          float *__restrict pOut) {
  event0();
  static_assert(size % 16 == 0, "tile*D must be a multiple of 16");

  const aie::vector<float, 16> neg_log2e = aie::broadcast<float, 16>(-1.44269504089f);
  const aie::vector<float, 16> one = aie::broadcast<float, 16>(1.0f);
  const aie::vector<float, 16> two = aie::broadcast<float, 16>(2.0f);

  const float *__restrict gate_ptr = pGate_in;
  const float *__restrict up_ptr = pUp_in;
  float *__restrict out_ptr = pOut;

  for (int off = 0; off < size; off += 16) {
    aie::vector<float, 16> gv = aie::load_v<16>(gate_ptr);
    gate_ptr += 16;
    aie::vector<float, 16> uv = aie::load_v<16>(up_ptr);
    up_ptr += 16;

    // s = 2^(-x*log2e) via the hardware exp2 SFU (bf16-output LUT).
    aie::vector<float, 16> arg = aie::mul(gv, neg_log2e).to_vector<float>();
    aie::vector<bfloat16, 16> s_bf = aie::exp2<bfloat16>(arg);

    // Up-convert the SFU result to f32 (mirrors the tanh-hybrid up-convert in
    // mm_silu_epilogue_f32o_hiprec), then do the reciprocal + one Newton
    // refinement step in f32 (mirrors silu_brick.cc SILU_MODE==5/6).
    aie::accum<accfloat, 16> s_acc;
    s_acc.from_vector(s_bf);
    aie::vector<float, 16> s = s_acc.to_vector<float>();

    aie::vector<float, 16> denom = aie::add(one, s);
    aie::vector<float, 16> r0 = aie::inv(denom);
    aie::vector<float, 16> dr0 = aie::mul(denom, r0).to_vector<float>();
    aie::vector<float, 16> r = aie::mul(r0, aie::sub(two, dr0)).to_vector<float>(); // Newton step

    aie::vector<float, 16> silu = aie::mul(gv, r).to_vector<float>();
    aie::vector<float, 16> outv = aie::mul(silu, uv).to_vector<float>();

    aie::store_v(out_ptr, outv);
    out_ptr += 16;
  }

  event1();
}

extern "C" {

// Flat element count of the resident [tile, D] region, provided at compile
// time by the caller (kernel-generation side picks tile/D; only the product
// needs to be a multiple of 16). Defaults kept in step with the sibling
// bricks in this tree (mm_silu_epilogue.cc EPI_M*EPI_N == 32*32).
#ifndef SWIGLU_M
#define SWIGLU_M 32
#endif
#ifndef SWIGLU_N
#define SWIGLU_N 32
#endif

void swiglu_f32_f32(const float *__restrict gate_in,
                    const float *__restrict up_in,
                    float *__restrict out) {
  swiglu<SWIGLU_M * SWIGLU_N>(gate_in, up_in, out);
}

} // extern "C"
