//===- silu_brick.cc ------------------------------------------*- C++ -*-===//
//
// Device-side SiLU brick (Conformer conv-module post-dwconv activation).
//   out[i] = silu(in[i]) = in[i] * sigmoid(in[i]),  sigmoid(x) = 0.5*(1 + tanh(x/2))
// Per-row: one row of `cols` f32 in -> f32 out. hiprec recipe (tanh ARG kept in f32,
// final x*sigmoid mul in f32; sigmoid bf16-tanh) -- byte-for-byte glu.cc's numerics
// and mm_silu_epilogue_f32o_hiprec, both WER-neutral.
//
// This is a SEPARATE single-op-loop brick, NOT fused into the dwconv per-channel loop.
// That is deliberate: fusing SiLU (or any multi-op epilogue) into dwconv1d_shift's
// objectfifo-driven per-channel loop miscompiles alternate (even) channel iterations
// on the current toolchain (mlir-aie fb1f7095, persists on latest Peano 2026071401) --
// see the KB log dwconv-fused-epilogue-alt-channel-miscompile. A simple memory-fed
// single-op loop like this one (as GLU / ctx_ln / affine_cast) is immune.
//
// Runtime ABI (mirrors glu_iron / ctx_ln): 1=instr, 3=in, 4=out, 5=tmp, 6=ctrl, 7=trace
// (output read from the b/out slot), driven from Rust by run_matmul8(3, instr, n, in_bo,
// out_bo, dummy_c, dummy_tmp, dummy_tr).
//
// TRACKED COPY -- installed into mlir-aie/.../layernorm by sync_kernels.sh.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

template <int N>
void silu_row(const float *restrict input, float *restrict output, int32_t cols) {
  event0();
  // Round-to-nearest-even on the bf16 narrowings (banked: WER-path bf16 casts MUST
  // round-nearest; default truncation biases toward zero and regressed WER).
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const ::aie::vector<float, N> halff = ::aie::broadcast<float, N>(0.5f);
  const ::aie::vector<float, N> onef = ::aie::broadcast<float, N>(1.0f);
  for (int i = 0; i < cols; i += N) {
    ::aie::vector<float, N> xv = ::aie::load_v<N>(input + i);
    // HYBRID max-precision sigmoid: the ONLY working tanh on this toolchain is the bf16
    // one (aie::tanh<float> mis-compiles -> garbage/100% WER); the FFN's all-bf16-sigmoid
    // recipe regressed RU 8.1->8.9 here (conv activation is approximation-sensitive). So
    // keep tanh at bf16 (unavoidable) but do EVERYTHING else in f32: up-convert the tanh
    // output immediately, then (1+tanh)*0.5 and x*sigmoid in f32 -- removing the two bf16
    // roundings the FFN recipe keeps. sigmoid(x)=0.5*(1+tanh(x/2)); silu=x*sigmoid(x).
    ::aie::vector<float, N> half_x = ::aie::mul(xv, halff);
    ::aie::vector<bfloat16, N> th_b = ::aie::tanh<bfloat16>(half_x); // bf16 tanh (only working tanh)
    ::aie::accum<accfloat, N> tacc;
    tacc.from_vector(th_b);
    ::aie::vector<float, N> th = tacc.template to_vector<float>();   // up-convert to f32 NOW
    ::aie::vector<float, N> tp1 = ::aie::add(th, onef);              // f32
    ::aie::vector<float, N> sig = ::aie::mul(tp1, halff);            // f32 sigmoid in [0,1]
    ::aie::vector<float, N> outv = ::aie::mul(xv, sig);             // f32 silu
    ::aie::store_v(output + i, outv);
  }
  event1();
}

extern "C" {
void silu_row(float *input, float *output, int32_t cols) {
  silu_row<16>(input, output, cols);
}
}
