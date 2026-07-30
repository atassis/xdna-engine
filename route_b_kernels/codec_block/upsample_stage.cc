//===- upsample_stage.cc ---------------------------------------*- C++ -*-===//
//
// One fused upsample stage of the fish-audio DAC codec decoder, at real checkpoint shapes:
//
//     snake(x, alpha)  ->  conv_transpose_1d(w, bias, stride, crop_right = stride)
//
// Reference semantics: s2.cpp/src/s2_codec.cpp:488-495. Stage 4 is the smallest of the four
// (c_in 192 -> c_out 96, k 4, stride 2) and is the one this is gated at; the shape constants are
// -D'd so the same source serves the other three.
//
// WHY THE ACTIVATION IS RESIDENT AND THE WEIGHTS STREAM, and not the other way round. The stage's
// weights are [c_in, c_out, k] = 192*96*4 f32 = 294 KB, which no core tile holds -- a 64 KB L1
// settles the direction on its own. The activation is [c_in, t] = 192*16 f32 = 12 KB and IS
// holdable, so it stays put and the weights go past it one output channel at a time. That also
// makes each streamed tile self-contained: tile `co` carries w[:, co, :] plus its own bias, so the
// kernel never needs to know which call it is on.
//
// THE SNAKE OUTPUT NEVER LEAVES THE CORE. It is computed once into a static L1 scratch on the
// first call and read by all c_out conv calls after it -- that is the point of the stage, not an
// optimisation: the intermediate between the two ops is exactly what a per-op host round trip
// would push to DDR and read back. Recomputing snake per output channel would also mean 96x the
// sin evaluations (294912 instead of 3072).
//
// SCALAR CONV, deliberately, as in the conv-transpose-1d brick it is built from: the scatter
// writes at `ti * stride` offsets and the codec's strides are 8/8/4/2, none a multiple of the
// 16-float vector width, so every store after the first would be unaligned and unaligned vector
// access truncates on aie2p (kb/aie2p-unaligned-vector-load-truncation). Correctness first; the
// vectorised form is gated against this one later.
//
// EVERY LOOP BOUND HERE IS A COMPILE-TIME CONSTANT. The sin brick corrupts on the streamed rail
// under a runtime bound and its verdict moves under unrelated edits
// (kb/sin-brick-codegen-instability), so this stage composes `sin_v` inline the way snake does --
// snake's path is the one that is genuinely device-green -- and gives every loop a literal bound.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// snake.cc, not sin.cc: this stage calls the snake BRICK's own entry per channel rather than
// composing sin_v inline. Inlining sin_v here returned real snake values in the wrong slots even
// with the scratch aligned, which is the instability in kb/sin-brick-codegen-instability; the
// brick entry is the arrangement whose green number actually reproduces, so use it and add nothing.
#include "../bricks/snake/snake.cc"

// Stage 4 by default; -D these for the other three stages.
#ifndef STAGE_C_IN
#define STAGE_C_IN 192
#endif
#ifndef STAGE_T
#define STAGE_T 16
#endif
#ifndef STAGE_K
#define STAGE_K 4
#endif
#ifndef STAGE_STRIDE
#define STAGE_STRIDE 2
#endif

// k = 2*stride and crop_right = stride at every codec stage, so out_len is exactly t*stride and
// the chain is 8*8*4*2 = 512x with no ragged edge. Asserted rather than assumed.
static_assert(STAGE_K == 2 * STAGE_STRIDE, "codec invariant: k == 2*stride");
#define STAGE_OUT_FULL ((STAGE_T - 1) * STAGE_STRIDE + STAGE_K)
#define STAGE_OUT (STAGE_T * STAGE_STRIDE)
static_assert(STAGE_T % 16 == 0 || STAGE_T == 16, "t must cover whole 16-float vectors");

namespace route_b_bricks {

// scratch holds snake(x) for the whole resident activation: [c_in, t], L1-resident, written once.
//
// alignas(64) IS REQUIRED, not hygiene. A 16xf32 vector store is 64 B and unaligned vector access
// truncates on aie2p (kb/aie2p-unaligned-vector-load-truncation). Without it this scratch came back
// shifted by exactly 2 floats -- 8 bytes -- so the stage returned real snake values in the wrong
// slots: rel-L2 1.374, and every value present but off by two. That reads exactly like the register
// aliasing the sin brick suffers from, and was nearly filed as another instance of it. It is not;
// it is alignment, and one keyword fixes it (rel-L2 1.374 -> 3.843e-08).
alignas(64) static float stage_scratch[STAGE_C_IN * STAGE_T];

// wtile:    [c_in*k weights for ONE output channel | bias | first-tile flag]
// resident: [alpha(c_in) | x(c_in*t)]
// out:      [t*stride] -- already cropped, since crop_right == stride drops exactly the tail
//
// The "compute snake once" decision rides in the TILE rather than in a static bool, because a
// zero-initialised static is not a safe assumption here: the first version used
// `static int stage_ready;` and the stage returned NaN, which is what an uncleared .bss looks like
// -- the guard read non-zero on the very first call, snake never ran, and the conv read whatever
// was in the scratch. The flag is data the host sets, so it cannot be wrong.
static inline void upsample_stage_core(const float *restrict wtile,
                                       const float *restrict resident,
                                       float *restrict out) {
  event0();
  const float *alpha = resident;
  const float *x = resident + STAGE_C_IN;

  if (wtile[STAGE_C_IN * STAGE_K + 1] != 0.0f) {
    for (int ci = 0; ci < STAGE_C_IN; ci++) {
      snake_f32((float *)x + ci * STAGE_T, stage_scratch + ci * STAGE_T, STAGE_T, alpha[ci]);
    }
  }

  float acc[STAGE_OUT_FULL];
  const float bias = wtile[STAGE_C_IN * STAGE_K];
  for (int p = 0; p < STAGE_OUT_FULL; p++) {
    acc[p] = bias;
  }
  for (int ci = 0; ci < STAGE_C_IN; ci++) {
    const float *w_row = wtile + ci * STAGE_K;
    const float *s_row = stage_scratch + ci * STAGE_T;
    for (int ti = 0; ti < STAGE_T; ti++) {
      const float xv = s_row[ti];
      const int base = ti * STAGE_STRIDE;
      for (int j = 0; j < STAGE_K; j++) {
        acc[base + j] += w_row[j] * xv;
      }
    }
  }
  // crop_right == stride, so the cropped output is simply the first t*stride columns.
  for (int p = 0; p < STAGE_OUT; p++) {
    out[p] = acc[p];
  }
  event1();
}

} // namespace route_b_bricks
