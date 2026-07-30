//===- residual_unit_bf16.cc -----------------------------------*- C++ -*-===//
//
// The codec residual unit with its two L1 intermediates held in BF16 instead of f32.
//
// WHY: capacity, not speed. Time-blocking needs a window longer than the unit's reach, which is
// (k-1)*dilation = 54 at dilation 9. The f32 version links only at t=16, so it cannot express a
// dilation-9 window at ANY size.
//
// TWO measurements shaped this, and the first one refuted the obvious fix. Simply halving both
// [c, t] intermediates to bf16 does NOT help: at t=64 it still overflowed .bss, by 17280 B, which
// is WORSE than f32 at t=32 (10944 B), because the resident activation buffer scales with t too and
// dominates. What isolated the real constraint was the conv-1d brick, which has no static scratches
// at all and PASSES at t=64 (rel-L2 4.592e-07, t=128 exceeds memory). So the rail happily delivers a
// [96, 64] resident; the scratches on top of it are what bust the budget.
//
// Hence this design: ONE [c, t] buffer, not two, with snake recomputed per output channel instead
// of materialised. That got the overflow at t=64 down from 17280 B to 4992 B -- and still did not
// fit. THE MEASURED FRONTIER, all at c=96, dilation 1, on the streamed rail:
//
//     scratches                          t=16    t=32     t=48    t=64          t=128
//     two f32   (residual_unit.cc)       links   +10944   -       -             -
//     two bf16                           -       links    -       +17280        -
//     one bf16  (this file)              -       links    links   +4992         -
//     NONE      (conv-1d brick)          links   links    links   links 4.6e-07 exceeds memory
//
// (+N = .bss overflowed by N bytes.) Read the last two rows together: with no static buffer at all
// the rail delivers t=64 comfortably, and ONE [96,64] bf16 buffer -- 12288 B -- is enough to break
// it. So the static budget at t=64 is only about 7.8 KB, while dilation 9 needs t > 54 for its
// context alone, before any useful window.
//
// CONCLUSION, and it is an architectural one rather than a tiling one: no arrangement that keeps
// the residual unit's intermediate in core L1 can hold a dilation-9 window. Chunking c_in does not
// change this -- the 1x1 conv mixes all channels, so one full [c, t] intermediate is live no matter
// how the weights are chunked; chunking only moves it between h and y. The intermediate has to
// leave L1, either staged through the MemTile (512 KB/column, which needs the bricklib rail to grow
// a MemTile staging path) or written out and re-read as a second pass. That is the fork the
// time-blocked driver has to take.
//
// ACCUMULATION STAYS F32. Only storage is bf16: every dot product accumulates into an f32 local and
// is rounded once on the way into the scratch. Accumulating in bf16 is the known way to lose a norm
// (kb/bf16-norm-numerics-and-accumulation-guards) and there is no reason to pay it here -- the
// accumulator is one row, 256 B at t=64.
//
// STATUS: THIS VARIANT DOES NOT WORK AND IS NOT ON THE CRITICAL PATH. It is kept because the
// measurements it produced are the answer to the design question, not because the code is wanted.
//
// It returns NaN wherever it links (t=32 and t=48, both dilations 1 and 3). The fault is in this
// file's RESTRUCTURE, not in bf16: forcing the intermediate back to f32 with -DRU_H_F32 NaNs
// identically, so recomputing snake per output channel is what broke it, and the storage type is
// innocent. Not debugged further, because even a correct version does not fit -- see the frontier
// above. Fixing it would buy a kernel that still cannot hold a dilation-9 window.
//
// Kept as a SEPARATE file from residual_unit.cc rather than a -D switch on it: that one is
// device-green at 2.000e-07 / 1.113e-07 / 9.660e-08 and its body is macro-expanded because a
// function-call boundary made the identical code return NaN. Perturbing it to add a dtype switch
// risks that green for a variant that did not pan out.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../bricks/snake/snake.cc"

#ifndef RU_C
#define RU_C 96
#endif
#ifndef RU_T
#define RU_T 64
#endif
#ifndef RU_K
#define RU_K 7
#endif
#ifndef RU_DILATION
#define RU_DILATION 9
#endif

namespace route_b_bricks {

// ONE [c, t] intermediate: the dilated conv's output. snake is never materialised -- it is
// recomputed a row at a time inside each conv, which is what removes the second [c, t] buffer.
// -DRU_H_F32 keeps the same restructure but stores the intermediate in f32, which isolates a
// bf16-conversion fault from a fault in the recompute-snake restructure itself.
#ifdef RU_H_F32
typedef float ru_h_t;
#else
typedef bfloat16 ru_h_t;
#endif
alignas(64) static ru_h_t ru16_h[RU_C * RU_T];
// One-row f32 staging. snake reads and writes f32 and the dot products accumulate in f32; only
// storage is bf16 (kb/bf16-norm-numerics-and-accumulation-guards -- never accumulate in bf16).
alignas(64) static float ru16_rowa[RU_T];
alignas(64) static float ru16_rowb[RU_T];

// tile:     [c*k weights | bias | phase | activate-now (unused here) | channel index]
// resident: [alpha0(c) | alpha2(c) | x(c*t)]
// out:      [t], meaningful only for phase-B tiles
//
// The activate-now flag is ignored: with snake recomputed per channel there is no once-per-stream
// precompute to guard. The field stays in the layout so the same host packing drives both variants.
//
// Macro-expanded into the bound kernel symbol for the measured reason in residual_unit.cc: behind a
// function call the identical f32 body returned NaN at every dilation.
#define ROUTE_B_RESIDUAL_UNIT_BF16_BODY(tile_, resident_, out_)                                   \
  event0();                                                                                       \
  {                                                                                               \
    const float *alpha0_ = (resident_);                                                           \
    const float *alpha2_ = (resident_) + RU_C;                                                     \
    const float *x_ = (resident_) + 2 * RU_C;                                                     \
    const float bias_ = (tile_)[RU_C * RU_K];                                                     \
    const int phase_b_ = ((tile_)[RU_C * RU_K + 1] != 0.0f);                                      \
    const int chan_ = (int)(tile_)[RU_C * RU_K + 3];                                              \
    float acc_[RU_T];                                                                             \
    for (int p = 0; p < RU_T; p++) {                                                              \
      acc_[p] = bias_;                                                                            \
    }                                                                                             \
    if (!phase_b_) {                                                                              \
      for (int ci = 0; ci < RU_C; ci++) {                                                         \
        snake_f32((float *)x_ + ci * RU_T, ru16_rowa, RU_T, alpha0_[ci]);                         \
        const float *w_ = (tile_) + ci * RU_K;                                                    \
        for (int j = 0; j < RU_K; j++) {                                                          \
          const float wv_ = w_[j];                                                                \
          const int sh_ = (RU_K - 1 - j) * RU_DILATION;                                           \
          for (int p = sh_; p < RU_T; p++) {                                                      \
            acc_[p] += wv_ * ru16_rowa[p - sh_];                                                  \
          }                                                                                       \
        }                                                                                         \
      }                                                                                           \
      for (int p = 0; p < RU_T; p++) {                                                            \
        ru16_h[chan_ * RU_T + p] = (ru_h_t)acc_[p];                                             \
        (out_)[p] = 0.0f;                                                                         \
      }                                                                                           \
    } else {                                                                                      \
      for (int ci = 0; ci < RU_C; ci++) {                                                         \
        for (int p = 0; p < RU_T; p++) {                                                          \
          ru16_rowb[p] = (float)ru16_h[ci * RU_T + p];                                            \
        }                                                                                         \
        snake_f32(ru16_rowb, ru16_rowa, RU_T, alpha2_[ci]);                                       \
        const float wv_ = (tile_)[ci];                                                            \
        for (int p = 0; p < RU_T; p++) {                                                          \
          acc_[p] += wv_ * ru16_rowa[p];                                                          \
        }                                                                                         \
      }                                                                                           \
      const float *xr_ = x_ + chan_ * RU_T;                                                       \
      for (int p = 0; p < RU_T; p++) {                                                            \
        (out_)[p] = acc_[p] + xr_[p];                                                             \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
  event1();

} // namespace route_b_bricks
