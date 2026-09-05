//===- transpose_dma.cc ---------------------------------------*- C++ -*-===//
//
// transpose-dma brick (movement group): transpose/relayout FOR FREE in the
// n-D strided DMA (dma_bd) -- NO COMPUTE, kills a V-transpose.
//
// WHAT THIS FILE IS: on real AIE2P hardware, the buffer descriptor (dma_bd)
// programs its own multi-dimensional address generator -- up to 4 nested
// [size, stride] pairs per side (source AND destination independently) --
// so a strided relayout (transpose, block-scatter, whatever a caller's
// stride vectors express) costs the shim/mem-tile DMA engine its ordinary
// streaming bandwidth and ZERO extra compute-core cycles. That is the whole
// point of this brick: it is a *config*, not an *op* -- there is no
// arithmetic anywhere in the loop below, only address arithmetic, which is
// exactly what a BD's dimension registers compute in hardware.
//
// This .cc is the CPU-side stand-in for that BD config:
//   (a) a golden/verification reference the device pass checks its actual
//       dma_bd wiring against (does the BD move the right bytes?),
//       matching golden.py in this directory bit-for-bit;
//   (b) a core-side FALLBACK for the case the transposing n-D DMA path
//       cannot be wired up yet -- transpose_tile.cc (aie_kernels/) already
//       documents that the DMA-transpose path is KNOWN to hang when
//       co-resident (blocker npu.rs:740) and ships a 2-D compute-tile
//       fallback for that reason. transpose_dma is the GENERIC op-TYPE
//       this brick group should converge on once/if that blocker clears:
//       any caller expresses its relayout as up to 4 [count, stride] pairs
//       (matching a real BD's dimension count) instead of a hardcoded 2-D
//       [mb, nb] transpose, so ONE brick covers transpose, block-scatter,
//       and gather-relayout alike -- never a per-model clone.
//
// Generic contract: resident [tile, D]-shaped data, arbitrary permutation
// or relayout expressed purely via strides (elements, not bytes) on each
// side. d0..d3 are iteration counts (1 = unused dim, degenerates cleanly);
// is0..is3 / os0..os3 are the matching input/output strides. A plain 2-D
// [mb, nb] -> [nb, mb] transpose (transpose_tile.cc's case) is just:
//   d0=mb, d1=nb, d2=1, d3=1;  is0=nb, is1=1, is2=0, is3=0;
//   os0=1, os1=mb, os2=0, os3=0;
//
// Element type chosen by macro so ONE source serves bf16 (uint16) and f32
// (uint32), same convention as transpose_tile.cc -- byte-exact copy
// regardless of float semantics, so this is BIT-EXACT vs any host
// relayout/transpose by construction (pure permutation, no arithmetic).
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
// Generic up-to-4-D strided relayout: out[addr_out(i0..i3)] = in[addr_in(i0..i3)]
// for i0 in [0,d0), i1 in [0,d1), i2 in [0,d2), i3 in [0,d3).
// Strides are in ELEMENTS (of TELEM), matching a dma_bd's dimension model.
// Degenerate (unused) dims: caller passes d=1, stride=0.
void transpose_dma(TELEM *__restrict__ in, TELEM *__restrict__ out,
                    int32_t d0, int32_t d1, int32_t d2, int32_t d3,
                    int32_t is0, int32_t is1, int32_t is2, int32_t is3,
                    int32_t os0, int32_t os1, int32_t os2, int32_t os3) {
  for (int32_t i0 = 0; i0 < d0; i0++) {
    const int32_t base_i0_in = i0 * is0;
    const int32_t base_i0_out = i0 * os0;
    for (int32_t i1 = 0; i1 < d1; i1++) {
      const int32_t base_i1_in = base_i0_in + i1 * is1;
      const int32_t base_i1_out = base_i0_out + i1 * os1;
      for (int32_t i2 = 0; i2 < d2; i2++) {
        const int32_t base_i2_in = base_i1_in + i2 * is2;
        const int32_t base_i2_out = base_i1_out + i2 * os2;
        for (int32_t i3 = 0; i3 < d3; i3++) {
          out[base_i2_out + i3 * os3] = in[base_i2_in + i3 * is3];
        }
      }
    }
  }
}

// Convenience 2-D entry point matching transpose_tile.cc's signature exactly
// (same op-TYPE, generic-4D-strided implementation underneath) -- makes the
// "this brick subsumes the 2-D compute-tile fallback" claim callable/testable
// directly rather than only assertable in the comment above.
void transpose_dma_2d(TELEM *__restrict__ in, TELEM *__restrict__ out,
                       int32_t mb, int32_t nb) {
  transpose_dma(in, out, mb, nb, 1, 1, nb, 1, 0, 0, 1, mb, 0, 0);
}
}
