// aie_cascade_bare.hpp -- CANDIDATE adf-free cascade accessor for aie_api.
//
// Problem it solves: aie_api exposes the core-to-core cascade ONLY through the
// ADF graph API (aie_api/adf/stream.hpp), which does `#include <adf.h>`. The
// Vitis ADF framework is not present in bare-metal / IRON / mlir-aie flows, so
// no straight-line .cc kernel can issue a cascade put/get through aie_api today
// -- even though the hardware path is exposed adf-free by the Peano intrinsics
// __builtin_aie2p_{mcd_write,scd_read}_*.
//
// This header wraps those intrinsics directly, in the aie:: namespace, with no
// <adf.h> dependency, so a fused multi-core K-reduction (mmul + cascade) can be
// written as an ordinary kernel.
//
// STATUS: the int32-vector path below is COMPILE-VERIFIED against
//   llvm-aie clang (acc2a72c) --target=aie2p-none-unknown-elf (see fix_adf_free_cascade.cc).
// The acc32 path is the one K-reduction actually wants; it is drafted here and
// the aie::accum<->v16acc32 native conversion is the one thing to finalize on review.
#pragma once
#include <aie_api/aie.hpp>

namespace aie {

// --- int32 vector cascade (compile-verified) ---
inline void cascade_out(const vector<int32_t, 16> &v) {
  __builtin_aie2p_mcd_write_vec(v, /*conf=*/0);
}
inline vector<int32_t, 16> cascade_in_i32() {
  return __builtin_aie2p_scd_read_vec(/*conf=*/0);
}

// --- acc32 cascade (the fused-K-reduction target; finalize accum<->native on review) ---
// inline void cascade_out(const accum<acc32, 16> &a) {
//   __builtin_aie2p_mcd_write_acc32(a /* .to_native()? */, 0);
// }
// inline accum<acc32, 16> cascade_in_acc32() {
//   return __builtin_aie2p_scd_read_acc32(0);
// }

} // namespace aie
