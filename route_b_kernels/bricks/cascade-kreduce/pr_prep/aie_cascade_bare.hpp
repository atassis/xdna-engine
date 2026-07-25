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
// STATUS: BOTH paths below are COMPILE-VERIFIED against
//   llvm-aie clang (acc2a72c) --target=aie2p-none-unknown-elf (see fix_adf_free_cascade.cc
//   for the int32 path and test_wrapper.cc for the acc32 round-trip).
// The accum<->v16acc32 native conversion (the one open detail flagged in PR_DRAFT.md)
// is resolved via aie_api's own accum interface: accum<acc32,16>::to_native() yields the
// native v16acc32 the builtin takes, and the implicit accum(storage_t) constructor rebuilds
// the accum from the builtin's return. No hand-rolled reinterpret_cast is needed.
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

// --- acc32 cascade (the fused-K-reduction target; compile-verified) ---
// The partial sums a multi-core K-reduction wants to move travel as accumulators.
// accum<acc32,16>::to_native() -> v16acc32 (the builtin's V16n arg); the builtin's
// V16n return -> accum via the implicit accum(storage_t) constructor.
inline void cascade_out(const accum<acc32, 16> &a) {
  __builtin_aie2p_mcd_write_acc32(a.to_native(), /*conf=*/0);
}
inline accum<acc32, 16> cascade_in_acc32() {
  return accum<acc32, 16>(__builtin_aie2p_scd_read_acc32(/*conf=*/0));
}

} // namespace aie
