//===- conformer_epilogues.cc ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Fused GEMM-epilogue kernels for the FastConformer block (Parakeet-TDT-0.6b-v3,
// d_model=1024, d_ff=4096). Each consumes the f32 accumulator C-tile produced by
// the matmul/GEMV (mm.cc matmul_bf16_f32 to_vector<float>(), or the mv_bf16
// matvec partial) and applies one elementwise/affine node, writing a bf16 tile,
// folding the f32(acc)->bf16(out) downconvert onto the array. This deletes a host
// elementwise pass per node (the same win as mm_silu/mm_gelu_epilogue).
//
// Nodes (host reference = scripts/parakeet_ref_encoder.py):
//   SiLU         out = x * sigmoid(x)            (FFN act + conv-module post-dwconv act)
//   GLU          out = a * sigmoid(g)            (conv-module pointwise_conv1 gate;
//                a = first D cols, g = next D cols of the [T,2D] GEMM output)
//   BatchNorm    out = scale_c * x + shift_c     (inference BN = per-channel affine:
//                scale_c = gamma/sqrt(var+eps), shift_c = beta - gamma*mean/sqrt(var+eps))
//   residual-add out = residual + alpha * x      (block residual; Macaron FFN alpha=0.5)
//
// sigmoid is synthesized from the on-chip tanh SFU (there is no sigmoid SFU):
//   sigmoid(z) = 1/(1+e^-z) = 0.5*(1 + tanh(z/2))
// matching the task contract "sigmoid = 0.5(tanh+1)" and the silu.cc/mm_silu idiom.
//
// BIAS: any pre-activation bias is folded into the producing matmul via host
// K-augmentation (see mm_silu_epilogue.cc), so these kernels take NO bias arg and
// the core stays within its 2 input-DMA channels (the AIE2P compute-tile limit).
// The BN scale/shift and the residual ARE genuine second operands (delivered on
// their own objectFIFO), so the BN/residual kernels use one extra input each.
//
// LAYOUT: SiLU/BN/residual are per-element (BN per-element within a column), so the
// mmul-blocked storage order is irrelevant -- the C ObjectFifo de-shuffles to
// row-major on the way out exactly as in the plain matmul; we walk the tile flat
// in 16-wide chunks. GLU is the ONE exception: it pairs column j with column N+j of
// the SAME row, so it is NOT layout-independent. The GLU kernel below assumes a
// ROW-MAJOR [M,2N] accumulator (the GEMV/matvec producer mv_bf16 writes row-major;
// the conv-module pointwise_conv1 at M=T uses the mm.cc path -- feed it the
// un-tiled output, or split a/g into two N-wide GEMM outputs). Documented so the
// resident-block wiring picks the right producer layout.
//
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// Load 16 f32 lanes from the accumulator and narrow to bf16 via an accumulator
// (mirrors the f32->bf16 narrow path in mm_silu_epilogue.cc).
static inline aie::vector<bfloat16, 16> load_narrow_bf16(const float *p) {
  aie::accum<accfloat, 16> a;
  a.from_vector(aie::load_v<16>(p));
  return a.to_vector<bfloat16>();
}

// sigmoid(z) = 0.5*(1 + tanh(z/2)), z a bf16 vector -> bf16 vector.
static inline aie::vector<bfloat16, 16>
sigmoid_bf16(const aie::vector<bfloat16, 16> &z,
             const aie::vector<bfloat16, 16> &half,
             const aie::vector<bfloat16, 16> &one) {
  auto half_z = aie::mul(z, half);                                  // z/2 (accum)
  auto t = aie::tanh<bfloat16>(half_z.to_vector<float>());          // tanh(z/2)
  auto t_p1 = aie::add(t, one);                                     // 1 + tanh
  return aie::mul(t_p1, half);                                      // 0.5*(1+tanh)
}

// ---------------------------------------------------------------------------
// SiLU: out = x * sigmoid(x). Per-element, flat over `size` lanes.
// ---------------------------------------------------------------------------
template <int size>
static inline void silu_epilogue(const float *__restrict pin,
                                 bfloat16 *__restrict pout) {
  event0();
  static_assert(size % 16 == 0, "tile size must be a multiple of 16");
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  for (int off = 0; off < size; off += 16) {
    aie::vector<bfloat16, 16> xv = load_narrow_bf16(pin + off);
    aie::vector<bfloat16, 16> sig = sigmoid_bf16(xv, half, one);
    aie::store_v(pout + off, aie::mul(xv, sig).to_vector<bfloat16>());
  }
  event1();
}

// ---------------------------------------------------------------------------
// GLU: in [M,2N] row-major (a = cols[0,N), g = cols[N,2N)), out [M,N].
//   out[r,j] = a[r,j] * sigmoid(g[r,j])
// ---------------------------------------------------------------------------
static inline void glu_epilogue(uint32_t M, uint32_t N,
                                const float *__restrict pin,
                                bfloat16 *__restrict pout) {
  event0();
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  for (uint32_t r = 0; r < M; r++) {
    const float *arow = pin + (uint32_t)(r * 2u * N);
    const float *grow = arow + N;
    bfloat16 *orow = pout + (uint32_t)(r * N);
    for (uint32_t j = 0; j < N; j += 16) {
      aie::vector<bfloat16, 16> av = load_narrow_bf16(arow + j);
      aie::vector<bfloat16, 16> gv = load_narrow_bf16(grow + j);
      aie::vector<bfloat16, 16> sig = sigmoid_bf16(gv, half, one);
      aie::store_v(orow + j, aie::mul(av, sig).to_vector<bfloat16>());
    }
  }
  event1();
}

// ---------------------------------------------------------------------------
// BatchNorm-fold (inference): out[r,c] = scale[c]*x[r,c] + shift[c].
// scale/shift are per-channel (length N) bf16, broadcast across the M rows.
// in [M,N] row-major f32 -> out [M,N] bf16.
// ---------------------------------------------------------------------------
static inline void bn_fold_epilogue(uint32_t M, uint32_t N,
                                    const float *__restrict pin,
                                    const bfloat16 *__restrict pscale,
                                    const bfloat16 *__restrict pshift,
                                    bfloat16 *__restrict pout) {
  event0();
  for (uint32_t r = 0; r < M; r++) {
    const float *xrow = pin + (uint32_t)(r * N);
    bfloat16 *orow = pout + (uint32_t)(r * N);
    for (uint32_t c = 0; c < N; c += 16) {
      aie::vector<bfloat16, 16> xv = load_narrow_bf16(xrow + c);
      aie::vector<bfloat16, 16> sv = aie::load_v<16>(pscale + c);
      aie::vector<bfloat16, 16> bv = aie::load_v<16>(pshift + c);
      // scale*x + shift: bf16 mul -> bf16 vector, then bf16 add.
      aie::vector<bfloat16, 16> prod = aie::mul(xv, sv).to_vector<bfloat16>();
      aie::store_v(orow + c, aie::add(prod, bv));
    }
  }
  event1();
}

// ---------------------------------------------------------------------------
// residual-add: out[i] = residual[i] + alpha * x[i]. Per-element, flat.
// x is the f32 accumulator (the sub-layer output); residual is the bf16 running
// activation; alpha is the Macaron half-step (0.5) or 1.0 for a full residual.
// ---------------------------------------------------------------------------
static inline void residual_add_epilogue(uint32_t size, float alpha,
                                         const float *__restrict px,
                                         const bfloat16 *__restrict presidual,
                                         bfloat16 *__restrict pout) {
  event0();
  const aie::vector<bfloat16, 16> av = aie::broadcast<bfloat16, 16>((bfloat16)alpha);
  for (uint32_t off = 0; off < size; off += 16) {
    aie::vector<bfloat16, 16> xv = load_narrow_bf16(px + off);
    aie::vector<bfloat16, 16> rv = aie::load_v<16>(presidual + off);
    aie::vector<bfloat16, 16> sx = aie::mul(xv, av).to_vector<bfloat16>();
    aie::store_v(pout + off, aie::add(rv, sx));
  }
  event1();
}

// ---------------------------------------------------------------------------
// extern "C" entry points.
//  - compile-time fixed-size symbols (EPI_M/EPI_N) for the mmul whole_array path,
//    matching the mm_silu_epilogue_f32_bf16 (const float*, bfloat16*) signature.
//  - runtime-length symbols (one symbol serves any shape), matching the mv_bf16
//    runtime-n convention (gelu_tile_bf16 etc.).
// ---------------------------------------------------------------------------
#ifndef EPI_M
#define EPI_M 8
#endif
#ifndef EPI_N
#define EPI_N 1024
#endif

extern "C" {

// --- fixed-size (mmul path) ---
void conformer_silu_epilogue_f32_bf16(const float *__restrict c_in,
                                      bfloat16 *__restrict c_out) {
  silu_epilogue<EPI_M * EPI_N>(c_in, c_out);
}

// --- runtime-length (GEMV/matvec path) ---
// SiLU over n flat lanes (n a multiple of 16).
void conformer_silu_bf16(uint32_t n, const float *c_in, bfloat16 *c_out) {
  // Templated on a chunk-of-16 loop via the size-templated impl is compile-time;
  // for runtime n we inline the same loop here.
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>(1.0f);
  event0();
  for (uint32_t off = 0; off < n; off += 16) {
    aie::vector<bfloat16, 16> xv = load_narrow_bf16(c_in + off);
    aie::vector<bfloat16, 16> sig = sigmoid_bf16(xv, half, one);
    aie::store_v(c_out + off, aie::mul(xv, sig).to_vector<bfloat16>());
  }
  event1();
}

// GLU: in [m,2n] row-major -> out [m,n].
void conformer_glu_bf16(uint32_t m, uint32_t n, const float *c_in,
                        bfloat16 *c_out) {
  glu_epilogue(m, n, c_in, c_out);
}

// BatchNorm-fold: in [m,n] -> out [m,n] with per-channel scale[n], shift[n].
void conformer_bn_fold_bf16(uint32_t m, uint32_t n, const float *c_in,
                            const bfloat16 *scale, const bfloat16 *shift,
                            bfloat16 *c_out) {
  bn_fold_epilogue(m, n, c_in, scale, shift, c_out);
}

// residual-add: out[i] = residual[i] + alpha * c_in[i], over n flat lanes.
void conformer_residual_add_bf16(uint32_t n, float alpha, const float *c_in,
                                 const bfloat16 *residual, bfloat16 *c_out) {
  residual_add_epilogue(n, alpha, c_in, residual, c_out);
}

} // extern "C"
