//===- gather_rows.cc ------------------------------------------*- C++ -*-===//
//
// GENERIC AIE2P brick: N-row vector gather (op-TYPE: indexed row lookup from a resident
// table), route_b_kernels/bricks/gather-rows.
//
//   out[t, :] = codebook[clip(idx[t], 0, n_rows-1), :]      for t in [0, T_TILE)
//
// Matches scripts/codec_quantizer_ref.py::rvq_lookup (~line 442):
//   idx = np.clip(codes[cb].astype(np.int64), 0, size - 1)
//   gathered = codebook[idx]                                 # [T, codebook_dim]
// and AudioCodec::decode's inline clamp_code lambda -- the clamp is not optional, an
// out-of-range code is a silent OOB read on device rather than an error, so it is baked
// into the kernel unconditionally, not left to the caller.
//
// WHY NOT aie::lut<4>::parallel_lookup (the rope-lut / sin-probe primitive). That hardware
// is a 256-ENTRY SCALAR LUT: one bf16 (or similarly narrow) value per int8 key, fetched via
// the SFU gather path probed in _verify/probe_gather_{known,width,ramp}.py. It is the wrong
// shape for this op twice over: (1) this gather reads an up-to-D-FLOAT ROW per index, not
// one scalar -- codec_quantizer_ref.py's docstring (kernel-map AWKWARD #1) is explicit that
// this is "an entirely different mechanism from an N-row VECTOR-embedding gather"; (2) the
// index range here goes up to 4096 (semantic codebook), far past the LUT's 256-entry table.
// Even if the ab/cd duplication layout that blocks rope-lut (see rope_lut.cc / sin.cc
// headers, "the layout question is OPEN") were solved tomorrow, parallel_lookup would not
// apply here. So this brick does the gather as what it actually is: a runtime-computed
// pointer offset into a resident buffer, then a plain vector load/store -- no SFU / gather-
// hardware primitive at all. That is completely ordinary AIE core addressing (any core can
// index its own local SRAM at a computed offset); it is just not a pattern any existing
// brick in this catalog happened to need before.
//
// CONTRACT CHOSEN: (a) codebook RESIDENT, indices STREAMED. The resident codebook is read
// straight out of L1 at a runtime-computed row offset; indices arrive as a stream of
// GATHER_T_TILE-sized tiles (the harness's one-streamed + one-resident-operand contract,
// bricklib._build_streamed / verify_streamed). This is ONLY valid while the whole codebook
// (n_rows * D * sizeof(resident_dt)) fits a core tile's L1 (64 KB, shared with the streamed
// tile's double buffer and the stack) -- it is NOT a general answer for an arbitrarily large
// codebook.
//   Real shapes (codec_quantizer_ref.py:205-207, kernel-map line 85): D (codebook_dim) = 8.
//   - residual codebooks (x9): n_rows=1024 -> f32 resident = 32768 B (32 KB). Fits
//     comfortably; this is the shape this brick is gated at (verify_gather_rows.py).
//   - semantic codebook (x1): n_rows=4096 -> f32 resident = 131072 B (128 KB). Does NOT fit
//     a core tile even alone. bf16 resident = 65536 B (64 KB) -- exactly one whole L1, no
//     room left for the streamed tile or stack; still not safely usable as-is. Reaching the
//     semantic codebook needs either a narrower resident dtype AND a chunked-n_rows variant
//     (stream the codebook itself in row-chunks, masked-accumulate the matching indices --
//     NOT implemented here), or contract (b) below.
//
// THE ARCHITECTURALLY-RIGHT ANSWER IS (b), NOT (a). A gather does 0% compute -- this
// project's doctrine (workspace CLAUDE.md optimization lens) is that the win is deleting
// bytes moved, not speeding up an op that has none to speed up. The right primitive is a
// MOVEMENT brick: have the DMA/BD engine read codebook rows directly in INDEX order (a
// data-dependent tap/offset pattern), so the "kernel" is a copy or nothing at all, and the
// T indices never need a compute-core round trip at all. `bricklib` cannot express this: every
// design in bricklib.py drives a fixed, index-INDEPENDENT TensorTiler2D access pattern
// (in_tap/out_tap/cst_tap are all built once from static shapes), so expressing (b) needs a new
// `bricklib._build_*` that can drive an objectFIFO from a per-tile offset.
//
// CORRECTED 2026-07-31 -- that is a bricklib limitation ONLY, and this header previously
// overstated it as a hardware unknown. The HARDWARE and the IRON API both support it:
//   * `ObjectFifoHandle.fill(..., offset_parameter=...)` takes a `ScratchpadParameter` directly
//     (instance `python/aie/iron/dataflow/objectfifo.py:629,696-700`), so no new op is needed.
//   * It is ALREADY SHIPPING in this repo: the Whisper decode path drives a per-token KV-write
//     offset this way -- `route_b_kernels/decode_fused/gen_decode.py` builds a StridedCopy with
//     `output_offset_parameter="kv_off"`, and `rust/npu-engine/src/asr/whisper_decoder.rs` writes
//     that scratchpad word per token from the host. The decode ELF is CONSTANT across tokens
//     precisely because the offset moved into a scratchpad parameter.
// So the offset is HOST-written per dispatch, which is the caveat that matters for a gather: it
// serves an embedding lookup (host knows the token id) but NOT an on-chip data-dependent gather
// where the index is computed mid-dispatch. A genuinely device-computed BD field is an unshipped
// toolchain milestone, not a hardware fact. See `docs/s2-bd-gather-feasibility.md`.
// This brick ships (a), the gateable version, and leaves (b) as the real follow-up target.
//
// HARD CONSTRAINTS OBSERVED (route_b_kernels/bricks header conventions):
//   - no `static` L1 state: the kernel keeps no buffer across calls, only the resident
//     codebook the harness itself owns and re-acquires per design, never per call.
//   - the body is a template<int...> instantiated directly into the bound extern "C"
//     symbol (sin.cc's shape), not hidden behind an opaque non-template `static inline`
//     helper.
//   - no local scratch buffer at all (straight load -> store per row), so there is no
//     `alignas` buffer to declare; `aie::load_v`/`store_v` on GATHER_D-wide sub-vectors need
//     only GATHER_D-natural alignment, not a fixed 64 B, and the per-row runtime byte offset
//     (idx * D * sizeof(float)) is always a multiple of that natural width for any D that
//     satisfies the static_assert below.
//   - live set per index is minimal (one scalar index, one row pointer, one D-wide vector) --
//     nowhere near the register-pressure spill territory sin.cc / rope_lut.cc's headers
//     document. This has NOT been device-verified (no NPU access for this task, see
//     verify_gather_rows.py's header) -- sin.cc's own streamed-rail history
//     (_verify/verify_sin.py) is a standing reminder that a clean compile says nothing
//     about spill/aliasing codegen on the streamed rail specifically.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GATHER_N_ROWS
#define GATHER_N_ROWS 1024    // codebook row count (compile-time: sizes the resident buffer)
#endif
#ifndef GATHER_D
#define GATHER_D 8             // embedding width per row (codec_quantizer_ref.py codebook_dim)
#endif
#ifndef GATHER_T_TILE
#define GATHER_T_TILE 16       // indices gathered per kernel call (must match the harness's in_tile)
#endif

namespace route_b_bricks {

// Row-copy vector width. GATHER_D must tile exactly in chunks of this -- the real D=8 needs
// exactly one chunk. A smaller GATHER_D would need a narrower kRowVec; not built here (YAGNI
// -- no model in this catalog uses codebook_dim < 8 today, see build-methodology's
// "earn generality from instances" rule).
constexpr unsigned kRowVec = 8;
static_assert(GATHER_D % kRowVec == 0, "GATHER_D must be a multiple of 8 (row-copy vector width)");
static_assert(GATHER_N_ROWS > 0 && GATHER_T_TILE > 0, "GATHER_N_ROWS/GATHER_T_TILE must be positive");

template <int N_ROWS, int D, int T_TILE>
void gather_rows_core(const int32_t *restrict idx, const float *restrict codebook,
                       float *restrict out) {
  event0();
  #pragma clang loop unroll(disable)
  for (int t = 0; t < T_TILE; ++t) {
    int32_t i = idx[t];
    // clamp to [0, N_ROWS-1] -- matches rvq_lookup's np.clip / clamp_code exactly. An
    // out-of-range index is a silent OOB read on device, not a host-side error, so this is
    // unconditional, not a debug check.
    if (i < 0) i = 0;
    else if (i >= N_ROWS) i = N_ROWS - 1;
    const float *row = codebook + (size_t)i * D;
    float *dst = out + (size_t)t * D;
    #pragma clang loop unroll(disable)
    for (int d = 0; d < D; d += kRowVec) {
      ::aie::store_v(dst + d, ::aie::load_v<kRowVec>(row + d));
    }
  }
  event1();
}

}  // namespace route_b_bricks

extern "C" {

// idx      : [GATHER_T_TILE] int32, this call's tile of runtime row indices.
// codebook : [GATHER_N_ROWS, GATHER_D] float32, RESIDENT (acquired once for the whole
//            stream by the harness / caller loop, not re-acquired per call).
// out      : [GATHER_T_TILE, GATHER_D] float32, this call's tile of gathered rows.
void gather_rows_f32(const int32_t *restrict idx, const float *restrict codebook,
                      float *restrict out) {
  route_b_bricks::gather_rows_core<GATHER_N_ROWS, GATHER_D, GATHER_T_TILE>(idx, codebook, out);
}

}  // extern "C"
