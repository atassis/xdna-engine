//===- mm_mode_resadd2a.cc --------------------------------------*- C++ -*-===//
//
// `out = a + scale*b` as a MODE of the modal GEMM, with BOTH operands riding the
// GEMM's own A fifo. lnaffcast could put one operand on B because gb is a WEIGHT
// and wants the A broadcast; resadd's a and b are both [M,N] activations, so
// neither does, and the broadcast is paid in acquires instead -- 64 pairs per C
// tile, with the operand pairing free because one map applies to both.
// See two-activation-mode-broadcast-acquire-cost.
//
// NO INDEX MAP AND NO TAP DERIVATION. A_L2L1 and C_L2L3 compose into a closed
// form: the value that arrives at A slot p belongs at C slot
//
//     (p / (r*k)) * (n*r) + (p % (r*k)) + j*(r*k)
//
// -- m/r runs of r*k contiguous f32 at stride n*r, based at j*r*k, where j picks
// which of the n/k A tiles spanning a C tile this one is. Verified against the
// composed map over 112 (shape x blocking x j) configurations, and it holds for
// every s == t (elementwise-mode-c-drain-index-map-depth). So the shim taps are
// plain row-major block reads and the un-permute costs one address expression.
//
// The C tile is the staging buffer: `a` is widened into it and `b` accumulated
// on top, which is why this needs the f32-out resident -- the bf16-out sibling's
// C tile is not an accumulator. n/k stage calls plus n/k apply calls fill it
// exactly, because m*k elements per A tile times n/k tiles is m*n.
//
// Self-contained because AIEAssignCoreLinkFiles traces only direct func.call
// edges from the core, so an object this one CALLED would never reach the link
// line. Same shape mm_mode_lnaffcast.cc uses.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <aie_api/aie.hpp>
#include <stdint.h>

// Route an f32 sum through the vector file before it reaches memory. A store
// issued out of the accumulator file bundles `vst bmlh0` with a VALU write to
// `dm0` in one instruction word and HANGS the core -- present in exactly the
// hanging arms and absent from exactly the completing ones, 5/5
// (same-bundle-valu-write-of-the-stored-accumulator). max(s, s-1) is an identity
// for every finite s and Peano does not fold it. Set to 0 once the toolchain
// stops emitting that bundle; the disassembly, not this flag, is the authority.
#ifndef RESADD2A_VFILE_DETOUR
#define RESADD2A_VFILE_DETOUR 1
#endif

// Which entry points are compiled EMPTY. The core still acquires and releases all 64 A tiles
// whichever value this takes, so an arm that completes where another times out separates the
// stream contract from the body, and the two halves of the body from each other.
//   0 = both real   1 = both empty   2 = stage only   3 = apply only
//   4 = stage writes a CONSTANT through the same address expression, apply empty. Separates
//       "the stores do not land" from "the stores land carrying the wrong value".
#ifndef RESADD2A_NOOP
#define RESADD2A_NOOP 0
#endif
#define RESADD2A_HAS_STAGE (RESADD2A_NOOP == 0 || RESADD2A_NOOP == 2 || RESADD2A_NOOP == 4)
#define RESADD2A_HAS_APPLY (RESADD2A_NOOP == 0 || RESADD2A_NOOP == 3)

static constexpr int kRun = EPI_R * EPI_K;      // contiguous f32 one C-drain run carries
static constexpr int kRuns = EPI_M / EPI_R;     // runs per A tile
static constexpr int kStride = EPI_N * EPI_R;   // C slots between two runs
static constexpr int kLanes = 16;

static_assert(EPI_M % EPI_R == 0, "a C tile must hold a whole number of runs");
static_assert(kRun % kLanes == 0, "a run must vectorise");
static_assert(EPI_N % EPI_K == 0, "n/k A tiles must span a C tile exactly");

template <unsigned N>
static inline ::aie::vector<float, N> vfile(const ::aie::vector<float, N> &s) {
#if RESADD2A_VFILE_DETOUR
  return ::aie::max(s, ::aie::sub(s, ::aie::broadcast<float, N>(1.0f)));
#else
  return s;
#endif
}

extern "C" {

// Widen A tile `j` into the C tile at the slots the drain will read it from.
void mm_resadd2a_stage_f32(const bfloat16 *__restrict a, float *__restrict c,
                           int32_t j) {
  event0();
#if RESADD2A_HAS_STAGE
  float *__restrict dst = c + j * kRun;
  for (int i = 0; i < kRuns; ++i)
    for (int q = 0; q < kRun; q += kLanes) {
#if RESADD2A_NOOP == 4
      ::aie::store_v(dst + i * kStride + q, ::aie::broadcast<float, kLanes>(42.0f));
#else
      ::aie::accum<accfloat, kLanes> av;
      av.from_vector(::aie::load_v<kLanes>(a + i * kRun + q), 0);
      ::aie::store_v(dst + i * kStride + q, vfile(av.to_vector<float>()));
#endif
    }
#endif
  event1();
}

// Accumulate scale*b onto the staged tile. `scale_bits` is the f32 bit pattern:
// an AIE2 tile has no input channel left for a scale operand, but rtp is a
// side-channel and costs none, so one build serves every scale.
void mm_resadd2a_apply_f32(const bfloat16 *__restrict b, float *__restrict c,
                           int32_t j, int32_t scale_bits) {
  event0();
#if RESADD2A_HAS_APPLY
  float scale;
  __builtin_memcpy(&scale, &scale_bits, sizeof(scale));
  const ::aie::vector<float, kLanes> sv = ::aie::broadcast<float, kLanes>(scale);
  float *__restrict dst = c + j * kRun;
  for (int i = 0; i < kRuns; ++i)
    for (int q = 0; q < kRun; q += kLanes) {
      ::aie::accum<accfloat, kLanes> bv;
      bv.from_vector(::aie::load_v<kLanes>(b + i * kRun + q), 0);
      float *__restrict p = dst + i * kStride + q;
      ::aie::vector<float, kLanes> sb =
          ::aie::mul(bv.to_vector<float>(), sv).to_vector<float>();
      ::aie::store_v(p, vfile(::aie::add(::aie::load_v<kLanes>(p), sb)));
    }
#endif
  event1();
}

}  // extern "C"
