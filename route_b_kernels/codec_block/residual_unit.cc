//===- residual_unit.cc ----------------------------------------*- C++ -*-===//
//
// One residual unit of the fish-audio DAC codec decoder, fused, at real checkpoint shapes:
//
//   y = x + conv1d_1x1( snake( conv1d_dilated( snake(x, a0), w1, d ), a2 ), w3 )
//
// Reference semantics: s2.cpp/src/s2_codec.cpp:381-393. Each decoder stage runs three of these at
// dilations 1, 3, 9 (.block.2/.3/.4), so this kernel times three is the rest of a stage.
//
// TWO PASSES, BECAUSE THE SECOND CONV NEEDS EVERY CHANNEL OF THE FIRST. Output channel co of the
// 1x1 conv reads all c_in channels of the dilated conv's output, so the unit cannot be evaluated as
// one streamed pass over output channels the way a single conv can -- the intermediate has to be
// complete first. It is [c_in, t] = 96*32 f32 = 12 KB, which a core tile holds, so it lives in L1
// and the unit runs as two phases over ONE stream of 2*c_out tiles:
//
//   phase A, tiles [0, c_out)        h[co] = conv_dilated(snake(x, a0), w1_co)   -> L1 scratch
//   phase B, tiles [c_out, 2*c_out)  y[co] = x[co] + conv_1x1(snake(h, a2), w3_co)
//
// so the whole unit is one hardware context and the intermediate never reaches DDR. Phase A's
// output tiles are ignored by the caller; only phase B's carry y.
//
// THE PHASE IS DATA, NOT STATE. Each tile carries its own phase flag, its own bias, and a
// compute-the-activation-now flag, so the kernel never infers anything from call ordering or from
// a static counter. That is not style: the fused upsample stage first used `static int` for
// exactly this and returned NaN, because an uninitialised .bss read non-zero on device.
//
// EVERY SCRATCH IS alignas(64). A 16xf32 vector store is 64 B and unaligned vector access truncates
// on aie2p; the upsample stage without it came back shifted by exactly 2 floats, every value
// present and all in the wrong slot (rel-L2 1.374 -> 3.843e-08 on the keyword).
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// snake.cc, not sin.cc: call the snake BRICK entry per channel rather than composing sin_v inline.
// Inlining it returns real values in the wrong slots even with the scratch aligned
// (kb/sin-brick-codegen-instability); the brick entry is the arrangement whose green reproduces.
#include "../bricks/snake/snake.cc"
#include "../bricks/conv-1d/conv_1d.cc"

#ifndef RU_C
#define RU_C 96
#endif
#ifndef RU_T
#define RU_T 32
#endif
#ifndef RU_K
#define RU_K 7
#endif
#ifndef RU_DILATION
#define RU_DILATION 1
#endif

namespace route_b_bricks {

// act holds snake() of whatever the current phase is activating; h holds the dilated conv's output
// across all channels. act is reused in phase B -- phase A's snake(x) is dead by then.
alignas(64) static float ru_act[RU_C * RU_T];
alignas(64) static float ru_h[RU_C * RU_T];

// tile:     [c_in*k weights | bias | phase (0 = A, nonzero = B) | activate-now | channel index]
// resident: [alpha0(c) | alpha2(c) | x(c*t)]
// out:      [t] -- meaningful only for phase-B tiles
//
// The channel index rides in the tile for the same reason the phase does: phase A has to deposit
// its result at h[co] rather than h[0], and phase B's residual add has to read x[co]. Neither can
// be inferred from call ordering without a counter, and a counter here is state the kernel has no
// safe way to initialise.
// EXPANDED INTO THE CALLER, NOT CALLED, and that is measured rather than stylistic. Written
// straight into the bound kernel symbol this body gates at 1.765e-07; the identical code behind a
// `static inline` function returned NaN at every dilation. Bisecting the function form showed each
// intermediate individually correct on device -- snake(x,a0) 5.945e-08, the dilated conv 4.444e-07,
// snake(h,a2) 4.768e-07, the 1x1 conv 4.483e-07 -- so nothing in the arithmetic was wrong; only the
// packaging was. Same class as kb/sin-brick-codegen-instability. The arrangement that measures green
// is the arrangement that ships.
//
// Also note what is NOT folded here: the two phases are written out straight rather than sharing one
// activation block behind `phase_b ? ru_h : x`. That folded version was tried and also returned NaN.
//
// A shim writes:
//     extern "C" void k(float *tile, float *resident, float *out) {
//       ROUTE_B_RESIDUAL_UNIT_BODY(tile, resident, out)
//     }
//
// No comments inside the macro: a `//` before a line-continuation eats the continuation and the
// rest of the body with it.
#define ROUTE_B_RESIDUAL_UNIT_BODY(tile_, resident_, out_)                                        \
  event0();                                                                                       \
  {                                                                                               \
    const float *alpha0_ = (resident_);                                                           \
    const float *alpha2_ = (resident_) + RU_C;                                                    \
    const float *x_ = (resident_) + 2 * RU_C;                                                     \
    const float bias_ = (tile_)[RU_C * RU_K];                                                     \
    const int phase_b_ = ((tile_)[RU_C * RU_K + 1] != 0.0f);                                      \
    const int activate_ = ((tile_)[RU_C * RU_K + 2] != 0.0f);                                     \
    const int chan_ = (int)(tile_)[RU_C * RU_K + 3];                                              \
    if (!phase_b_) {                                                                              \
      if (activate_) {                                                                            \
        for (int ci = 0; ci < RU_C; ci++) {                                                       \
          snake_f32((float *)x_ + ci * RU_T, ru_act + ci * RU_T, RU_T, alpha0_[ci]);              \
        }                                                                                         \
      }                                                                                           \
      conv_1d_causal_core(ru_act, (tile_), bias_, ru_h + chan_ * RU_T,                            \
                          RU_C, RU_K, RU_T, RU_DILATION);                                         \
      for (int p = 0; p < RU_T; p++) {                                                            \
        (out_)[p] = 0.0f;                                                                         \
      }                                                                                           \
    } else {                                                                                      \
      if (activate_) {                                                                            \
        for (int ci = 0; ci < RU_C; ci++) {                                                       \
          snake_f32(ru_h + ci * RU_T, ru_act + ci * RU_T, RU_T, alpha2_[ci]);                     \
        }                                                                                         \
      }                                                                                           \
      conv_1d_causal_core(ru_act, (tile_), bias_, (out_), RU_C, 1, RU_T, 1);                      \
      const float *x_row_ = x_ + chan_ * RU_T;                                                    \
      for (int p = 0; p < RU_T; p++) {                                                            \
        (out_)[p] += x_row_[p];                                                                   \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
  event1();

} // namespace route_b_bricks
