// SPDX-License-Identifier: Apache-2.0
// Per-column partial argmax for the e2e/NPU lm-head. Each of the 8 whole-array columns runs this over its
// contiguous VOCAB_PAD/8 slice of the proj_out logits: it scans the slice and emits the LOCAL index of the
// max + the max value (cast to f32). The host then does the trivial 8-way reduce (global = col*slice + local;
// pick the column with the largest value) — bit-exact with the host f32 argmax (strict `>`, first-max wins,
// matches whisper.rs argmax()).
//
// _argmax_vector is the aie_kernels/aie2/argmax.cc brick (mlir-aie fork, branch brick/aie2p-argmax,
// commit 4d20715b94a), vendored verbatim rather than linked: this engine's kernel sources are pinned
// per-file (see argmax_op.py's SourceArtifact), same as every other route_b_kernels brick, and the
// branch predates this build's toolchain.lock pin. argmax_slice_bf16 wraps it at index_offset=0 so the
// output stays the LOCAL index this file's callers (argmax_design.py, whisper_decoder.rs) already
// expect — the brick's global-index/host-reduce-elimination path is follow-on work, gated on threading
// a per-column compile-time offset through the IRON design and a metadata discriminator so an
// already-built ELF's LOCAL-index output is never misread as global.
#include <aie_api/aie.hpp>
#include <cassert>
#include <limits>
#include <stdint.h>
#include <string.h>

// aie_kernel_utils.h's two scheduling hints, inlined rather than included: this file compiles from
// route_b_kernels/decode_fused, which has no verified include path to that header (unlike
// route_b_kernels/aie_kernels/*.cc). Semantically inert either way — they only shape scheduling.
#define ARGMAX_STRINGIFY_(a) #a
#define ARGMAX_STRINGIFY(a) ARGMAX_STRINGIFY_(a)

#if defined(__chess__)
#define ARGMAX_PREPARE_FOR_PIPELINING [[chess::prepare_for_pipelining]]
#define ARGMAX_LOOP_MIN_ITERATION_COUNT(x) [[chess::min_loop_count(x)]]
#elif defined(__AIECC__)
#define ARGMAX_PREPARE_FOR_PIPELINING
#define ARGMAX_LOOP_MIN_ITERATION_COUNT(x)                                    \
  _Pragma(ARGMAX_STRINGIFY(clang loop min_iteration_count(x)))
#else
#define ARGMAX_PREPARE_FOR_PIPELINING
#define ARGMAX_LOOP_MIN_ITERATION_COUNT(x)
#endif

extern "C" {

// Record layout: out[0] the winning value (bfloat16 widened to float, bit-cast into the int32 slot),
// out[1] its index (index_offset + the position found). Matches this file's existing [f32|i32] packing.
static inline void _argmax_store(int32_t *restrict out, float value, int32_t index) {
  memcpy(out, &value, sizeof(value));
  out[1] = index;
}

// One streaming pass, N=32 lanes. Lane j only ever sees positions j, j+N, j+2N, ..., so a strict `>`
// keeps the earliest of equal values within a lane, and the global first-occurrence index is
// min(offset[j] + j) over the lanes still holding the maximum — resolved once, after the loop.
static void _argmax_vector(bfloat16 *restrict in, int32_t *restrict out, const int32_t input_size,
                           const int32_t index_offset) {
  event0();
  constexpr int32_t N = 32;
  using V = aie::vector<bfloat16, N>;
  using Idx = aie::vector<int16_t, N>;
  assert(input_size <= INT16_MAX); // lane offsets are int16; no input tile reaches 32767 bf16 (64 KB, a whole core)

  alignas(64) int16_t lane_init[N];
  for (int32_t k = 0; k < N; k++)
    lane_init[k] = (int16_t)k;
  const Idx lane = aie::load_v<N>(lane_init);

  V running_max = aie::broadcast<bfloat16, N>(std::numeric_limits<bfloat16>::lowest());
  Idx running_off = aie::zeros<int16_t, N>();
  Idx offset = aie::zeros<int16_t, N>();
  const Idx step = aie::broadcast<int16_t, N>((int16_t)N);

  int32_t i = 0;
  ARGMAX_PREPARE_FOR_PIPELINING
  ARGMAX_LOOP_MIN_ITERATION_COUNT(2)
  for (; i + N <= input_size; i += N) {
    V next = aie::load_v<N>(in + i);
    auto improved = aie::gt(next, running_max);
    running_max = aie::select(running_max, next, improved);
    running_off = aie::select(running_off, offset, improved);
    offset = aie::add(offset, step);
  }

  bfloat16 best = aie::reduce_max(running_max);
  const Idx candidates =
      aie::select(aie::broadcast<int16_t, N>(INT16_MAX),
                  aie::add(running_off, lane), aie::eq(running_max, best));
  int32_t best_index = (int32_t)aie::reduce_min(candidates);

  for (; i < input_size; i++) { // remainder: input_size need not be a multiple of N
    if (in[i] > best) {
      best = in[i];
      best_index = i;
    }
  }

  _argmax_store(out, (float)best, index_offset + best_index);
  event1();
}

// `out` is a 4×bf16 (8-byte) slot the fusion framework treats as bf16 (it is uniform-bf16); the brick
// writes a [f32|i32] record through an int32 view of the same 8 bytes.
void argmax_slice_bf16(bfloat16 *restrict in, bfloat16 *restrict out, int32_t n) {
  _argmax_vector(in, (int32_t *)out, n, /*index_offset=*/0);
}

} // extern "C"
