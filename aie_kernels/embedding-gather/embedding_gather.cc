//===- embedding_gather.cc -------------------------------------*- C++ -*-===//
//
// GENERIC AIE2P brick: N-wide bf16 vector copy, the compute-core half of a
// host-offset row gather from an L3-resident embedding table (op-TYPE:
// indexed row lookup, contract (b) -- see aie_kernels/gather-rows/
// gather_rows.cc's header for contract (a), the L1-resident sibling this is
// NOT). Design: docs/s2-embedding-gather-design.md.
//
// Row selection happens in the DMA, not here: the host writes a per-dispatch
// byte offset into an aiex.scratchpad_parameter (gen_embedding_gather.py),
// so by the time this kernel runs, `in_chunk` already IS the requested row's
// next GATHER_CHUNK_N elements. This file never sees an index and therefore
// cannot clamp one -- that is the caller's job (see the design doc's "Does
// not: clamp the index" section).
//
// ONE vector op per call, deliberately, not a D/N-chunk loop over a whole
// row: a multi-iteration loop inside an AIE kernel body has silently
// miscompiled on this toolchain pin for other bricks in this catalog
// (gelu-erf, sin), so volume belongs in the worker, not the body. The
// D/GATHER_CHUNK_N repeat count for a full row lives in the objectFifo WORKER
// loop in gen_embedding_gather.py instead.
//
// GATHER_CHUNK_N=16 matches this catalog's established bf16 row-vector width
// (rmsnorm.cc, cast_quant_bf16_int8.cc, swiglu.cc all instantiate at N=16).
//
// No `static` state, no local buffer (a bare load_v -> store_v needs no
// scratch array, so there is no `alignas` buffer to declare here either).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GATHER_CHUNK_N
#define GATHER_CHUNK_N 16
#endif

extern "C" {

void embedding_gather_chunk_bf16(const bfloat16 *restrict in_chunk,
                                  bfloat16 *restrict out_chunk) {
  event0();
  ::aie::store_v(out_chunk, ::aie::load_v<GATHER_CHUNK_N>(in_chunk));
  event1();
}

}  // extern "C"
