//===- transpose_tile.cc ----------------------------------*- C++ -*-===//
//
// COMPUTE-tile transpose brick for the Conformer conv module (Task 0.1).
//
// Transposes ONE contiguous [mb, nb] row-major tile to a contiguous [nb, mb]
// row-major tile, ENTIRELY on the compute core:
//
//     out[j*mb + i] = in[i*nb + j]     for i in [0,mb), j in [0,nb)
//
// This is the de-risking enabler for conv-module step 3b (kill the two HOST
// transposes GLU[T,D]->dwconv[D,T]->pw2[T,D]). The transposing n-D DMA path is
// KNOWN to hang when co-resident (blocker npu.rs:740), so the ELEMENT transpose
// lives here on the core; the shim DMA only does contiguous reads + a
// block-scatter write with UNIT inner stride (no DMA element transpose).
//
// Pure data movement (no arithmetic) -> BIT-EXACT vs host x.T. Element type is
// chosen by macro so ONE source serves bf16 (uint16) and f32 (uint32); the copy
// is byte-exact regardless of float semantics. mb, nb are runtime int32 args so a
// single compiled object serves any block size (mirrors silu_row's `cols` arg).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
#include <stdint.h>

#ifdef TPOSE_i32
typedef uint32_t TELEM;
#else
typedef uint16_t TELEM; // default: bf16 (2 bytes)
#endif

extern "C" {
void transpose_tile(TELEM *__restrict__ in, TELEM *__restrict__ out, int32_t mb,
                    int32_t nb) {
  for (int32_t i = 0; i < mb; i++) {
    const TELEM *in_row = in + i * nb;
    for (int32_t j = 0; j < nb; j++) {
      out[j * mb + i] = in_row[j];
    }
  }
}
}
