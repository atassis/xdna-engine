//===- gemv_int8xint4.cc --------------------------------------*- C++ -*-===//
//
// BRICK  gemv-int8xint4  [group: gemv]
//
// M=1 matvec: int8 activation row x int4 weight matrix, decode-shape.
// This is the DECODE movement lever: at M=1 the workload is memory-bound
// (bytes of W moved per token), not FLOP-bound, so the win is (a) int4
// weight packing -> half the DDR/L2 bytes of an int8 weight, and (b) still
// landing on the native aie2p `mmul_8_4` systolic primitive instead of a
// scalar/LUT fallback, by PADDING the decode M=1 row up to the primitive's
// native M=4 (same trick as norm_gemv_prologue.cc pads M=1 -> M=64 for the
// bf16 norm+gemv fusion: rows 1..M_NATIVE-1 are zero, so they don't perturb
// the real row-0 output and are simply discarded on read-back).
//
// This is a GENERIC op-TYPE against the resident [tile,D] stream contract:
//   A: [M_NATIVE, K]  int8   activations (row 0 = the real decode token,
//                             rows 1..M_NATIVE-1 = zero padding)
//   B: [K, N]         int4   weights, tiled into native (K_NATIVE x N_NATIVE)
//                             = (16 x 16) sub-blocks, sub-block order
//                             [k_tile][n_tile] (k-major, n-minor), each
//                             sub-block itself row-major over (k_local,
//                             n_local) -- see NOTE below on this layout being
//                             an assumption for the device-side leaf.
//   C: [M_NATIVE, N]  int32  accumulator, one native mmul C-block per
//                             n_tile, in the mmul's own internal C-block
//                             layout (opaque -- NOT asserted row-major; see
//                             NOTE). Only row 0 (real token) is meaningful.
//
// Parameterized by GEMV_K / GEMV_N (compile-time, multiples of the native
// K_NATIVE=16 / N_NATIVE=16 tile) so this is one brick for any decode
// int8-activation x int4-weight matvec shape (e.g. an FFN down-proj or an
// lm-head slice), not a per-model hand-fused kernel.
//
// API pattern (per the model kernel + the aie_api mmul convention):
//   aie::mmul<M,K,N,TA,TB,AccumTag> acc;
//   acc.mul(A, a_sign, B, b_sign) / acc.mac(...)   -- accumulate over K tiles
//   acc.to_vector<int32>()                          -- read out the C block
//
// NOTE (open item for the device-validation leaf): aie2p's `mmul_8_4`
// specialization (mlir-aie/third_party/aie_api/include/aie_api/detail/
// aie2p/mmul_8_4.hpp) is only concretely specialized for
// <M=4, K=16, N=16, int8, int4, 32-bit acc>, backed by the
// `mac_4x16_16x16_conf` / `mul_4x16_16x16` compiler intrinsics. Those
// intrinsics' exact expected in-register sub-tile permutation for B (and
// the C-block's internal element order) is HARDWARE-DEFINED and is not
// re-derived here from the aie_api headers alone (no example in-repo calls
// mmul_8_4 directly). golden.py therefore implements the STRAIGHT
// mathematical matvec (logical row-major A/B/C) as the reference the
// device pass must reconcile against once the true B/C sub-tile packing is
// confirmed on-device (or from AMD's mmul_8_4 lowering docs) -- flag this,
// don't silently assume row-major is correct HW packing.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// ---- native aie2p mmul_8_4 tile shape (fixed by the hardware primitive) ----
#ifndef GEMV_M_NATIVE
#define GEMV_M_NATIVE 4   // native M for mmul<M,K,N,int8,int4,32> (M=1 decode padded up to this)
#endif
#ifndef GEMV_K_NATIVE
#define GEMV_K_NATIVE 16  // native K sub-tile for mmul_8_4
#endif
#ifndef GEMV_N_NATIVE
#define GEMV_N_NATIVE 16  // native N sub-tile for mmul_8_4
#endif

// ---- full (resident-tile) problem shape -- parameterize per instance ----
#ifndef GEMV_K
#define GEMV_K 768    // full reduction dim (e.g. Gemma-270M hidden size)
#endif
#ifndef GEMV_N
#define GEMV_N 256    // full output dim for this brick instance / tile
#endif

static_assert(GEMV_K % GEMV_K_NATIVE == 0, "GEMV_K must be a multiple of the native K tile");
static_assert(GEMV_N % GEMV_N_NATIVE == 0, "GEMV_N must be a multiple of the native N tile");

namespace {

constexpr int kKTiles = GEMV_K / GEMV_K_NATIVE;
constexpr int kNTiles = GEMV_N / GEMV_N_NATIVE;

// One native mmul_8_4 op-type: M=4 (padded decode row 0..3), K=16, N=16.
using MMUL = ::aie::mmul<GEMV_M_NATIVE, GEMV_K_NATIVE, GEMV_N_NATIVE, int8, int4, accauto>;

// A is signed int8 activations, B is signed int4 weights (symmetric
// per-channel-quantized weight is the common decode-quant convention);
// signedness is carried by the `int8`/`int4` element types themselves --
// the public aie::mmul wrapper's mul()/mac() take just (A, B) and forward
// the correct sign flags to the detail-level mac_4x16_16x16_conf intrinsic.

}  // namespace

extern "C" {

// a: [GEMV_M_NATIVE, GEMV_K] int8, row 0 = real decode token, rows 1..3 zero.
// b: [kKTiles, kNTiles, GEMV_K_NATIVE * GEMV_N_NATIVE] int4 packed weight,
//    sub-block (kt, nt) laid out contiguously (kt outer, nt inner).
// c: [kNTiles, GEMV_M_NATIVE * GEMV_N_NATIVE] int32 accumulator, one native
//    mmul C-block per output n_tile (native mmul internal element order).
void gemv_int8xint4(const int8 *restrict a, const int4 *restrict b, int32_t *restrict c) {
  event0();

  for (int nt = 0; nt < kNTiles; nt++) {
    MMUL acc;
    for (int kt = 0; kt < kKTiles; kt++) {
      // A sub-tile: [GEMV_M_NATIVE, GEMV_K_NATIVE] slab at column-offset kt*K_NATIVE.
      // A is resident row-major over the FULL K, so the (m, kt) sub-tile is
      // strided; load one native-M-row-worth of the mmul's A vector layout
      // by gathering per-row K_NATIVE chunks (native A vector = M*K_NATIVE
      // contiguous elements in mmul's expected order == per-row concat,
      // matching how norm_gemv_prologue treats the resident A tile linearly).
      int8 a_stage[GEMV_M_NATIVE * GEMV_K_NATIVE];
      for (int m = 0; m < GEMV_M_NATIVE; m++) {
        const int8 *row = a + m * GEMV_K + kt * GEMV_K_NATIVE;
        for (int kk = 0; kk < GEMV_K_NATIVE; kk++) a_stage[m * GEMV_K_NATIVE + kk] = row[kk];
      }
      ::aie::vector<int8, MMUL::size_A> A = ::aie::load_v<MMUL::size_A>(a_stage);

      // (K_NATIVE*N_NATIVE)/2 bytes/block: int4* advances 1 byte/lane on Peano,
      // and a contiguously packed 16x16 int4 block is 128 bytes (= size_B/2).
      // See gemm_int8xint4.cc / int4-gemm-kernel-stride-fix.
      const int4 *b_block = b + (static_cast<size_t>(kt) * kNTiles + nt) * ((GEMV_K_NATIVE * GEMV_N_NATIVE) / 2);
      ::aie::vector<int4, MMUL::size_B> B = ::aie::load_v<MMUL::size_B>(b_block);

      if (kt == 0)
        acc.mul(A, B);
      else
        acc.mac(A, B);
    }
    ::aie::store_v(c + static_cast<size_t>(nt) * (GEMV_M_NATIVE * GEMV_N_NATIVE), acc.template to_vector<int32_t>());
  }

  event1();
}

}  // extern "C"
