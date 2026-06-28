//===- mm_movement_stub.cc ------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, xdna2-asr-engine.
//
//===----------------------------------------------------------------------===//
//
// DATA_MOVEMENT_ONLY twin of aie_kernels/aie2p/mm.cc's bf16->f32 matmul, for the
// per-op occupancy harness (brick #8, the measure-first gate). It exports the SAME
// two extern "C" symbols the whole_array IRON program links --
//   matmul_bf16_f32(bfloat16*, bfloat16*, float*)   and   zero_f32(float*)
// -- so the build links this object in place of mm_<m>x<k>x<n>.o WITHOUT changing
// the IRON dataflow graph (objectFIFO topology, BD chains, lock structure, DMA
// access patterns) one bit.
//
// THE MEASUREMENT IDEA (A/B latency diff):
//   The L3<->L2<->L1 DMA that streams A/B in and C out is driven entirely by the
//   IRON runtime sequence + objectFIFOs (whole_array_iron.py core_fn acquire/
//   release), NOT by the kernel body. So a no-op kernel keeps the per-dispatch
//   data movement BYTE-IDENTICAL to the real kernel; only the in-core COMPUTE
//   (L1->reg loads + bfp16 MACs + acc stores in the inner matmul) is removed.
//   Therefore, for a fixed (M,K,N) dispatch:
//       t_full  = movement + dispatch_overhead + stall + COMPUTE
//       t_stub  = movement + dispatch_overhead + stall            (this file)
//       compute_time   = t_full - t_stub
//       compute_frac   = (t_full - t_stub) / t_full
//   compute_frac near 1 => COMPUTE-bound (mmul/bfp16/int8 tiles pay);
//   compute_frac near 0 => MOVEMENT/dispatch-bound (the FORMAT/COMPUTE bricks do
//   NOT pay here -- attack it with the MOVEMENT bricks instead). This is exactly
//   the Phase-0 ranking the brick-honoring rebuild needs (see the spec
//   2026-06-28-parakeet-tdt-full-npu-brick-honoring.md, Phase 0).
//
// zero_f32 is kept REAL (reuses AMD's zero_vectorized) because zeroing the f32
// accumulator tile is part of every kernel's fixed per-tile overhead and is
// negligible compute -- we isolate the heavy MAC datapath, not the acc reset.
// The matmul body keeps event0()/event1() so a trace-instrumented run still shows
// a (near-zero) compute bracket for cross-checking.
//
// The stub C output is the zeroed accumulator (meaningless numerically); the
// harness asserts the stub's C ~= 0 to PROVE the compute was actually elided
// (a no-op stub that still computed would invalidate the diff).
//
// Build defines mirror the production fast-BFP16 resident kernel
// (scripts/build_parakeet_kernels.sh): -Dbf16_f32_ONLY -DDIM_M -DDIM_K -DDIM_N
// plus the BFP16/emulate flags from KERNEL_DEFINES. DIM_* only size zero_f32 here;
// the matmul twin is shape-agnostic. See scripts/build_parakeet_occupancy_stub.sh.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// AMD's zero_vectorized<T,M,N> / zero_scalar (header-guarded). Resolved via the
// kernel build's `-I ${MLIR_AIE_DIR}/include`.
#include "aie_kernels/aie2p/zero.cc"

// These match the production resident tile (m=64, k=32, n=128). Overridable via
// -DDIM_M / -DDIM_K / -DDIM_N to track whatever tile the Makefile selects.
#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 32
#endif
#ifndef DIM_N
#define DIM_N 128
#endif

extern "C" {

// Movement-only twin of matmul_bf16_f32: identical ABI, MAC datapath ELIDED.
// (void)-casts keep the parameters live for the ABI without emitting any loads.
void matmul_bf16_f32(bfloat16 *a_in, bfloat16 *b_in, float *c_out) {
  event0();
  (void)a_in;
  (void)b_in;
  (void)c_out;
  event1();
}

// Real accumulator zero (cheap, shared with the full kernel's per-tile prologue).
void zero_f32(float *c_out) { zero_vectorized<float, DIM_M, DIM_N>(c_out); }

} // extern "C"
