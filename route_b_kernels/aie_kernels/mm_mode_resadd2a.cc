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
// Apply differs from stage in TWO ways -- it multiplies, and it reads the C tile back -- so 3
// implicates both at once. 5 and 6 are that 2x2's remaining cells, stage empty in both:
//   5 = apply MULTIPLIES but does not read C (store scale*b)
//   6 = apply READS C but does not multiply (store C + b)
// Arm 2 is the cell with neither and it passes, so whichever of 5/6 hangs carries the fault.
#ifndef RESADD2A_NOOP
#define RESADD2A_NOOP 0
#endif
#ifndef RESADD2A_BF16_MUL
#define RESADD2A_BF16_MUL 0
#endif

// Accumulate with ONE fused `aie::mac` instead of a separate `aie::mul` and `aie::add`. Two
// reasons, and the brick one holds even if the hang one does not: (a) mul-then-add is the generic
// pair where the hardware has the fused op, which is the recurring mistake the brick catalog
// exists to stop; (b) MEASURED, the apply hang on the bf16 datapath tracks a store bundled in one
// instruction word with a VALU write -- `vst x9, [p4]; mov p4, p3; vmul.f dm1, x5, x4, r4` -- and
// the three bf16 arms split 3/3 on whether that bundle is present. Removing the separate vmul
// removes the operation that gets packed there. Note the store is from the VECTOR file (x9), so
// the RESADD2A_VFILE_DETOUR above did its job and the core hung anyway: the hazard is a VALU write
// bundled WITH the store, not only a store sourced from the accumulator file.
// Requires RESADD2A_BF16_MUL (mac takes the bf16 operands directly).
#ifndef RESADD2A_MAC
#define RESADD2A_MAC 0
#endif
#define RESADD2A_HAS_STAGE (RESADD2A_NOOP == 0 || RESADD2A_NOOP == 2 || RESADD2A_NOOP == 4)
#define RESADD2A_HAS_APPLY \
  (RESADD2A_NOOP == 0 || RESADD2A_NOOP == 3 || RESADD2A_NOOP == 5 || RESADD2A_NOOP == 6)
#define RESADD2A_APPLY_MULS (RESADD2A_NOOP != 6)

// Put `scale*b` on the bf16 datapath instead of the f32 one. MEASURED: the f32 `aie::mul` is what
// hangs the core -- multiply-without-reading-C is 9/9 TIMEOUT, read-C-without-multiplying is 7/9
// COMPLETED (resadd2a-apply-hang-is-the-f32-multiply-not-the-rmw). Peano has no f32 vector
// multiply, so it emulates one as a bf16 product chain, 8 vmul.f + 7 vadd.f per 16 lanes; b is
// already bf16 and scale is a build constant, so that chain is reconstructing a product neither
// operand needs widened.
//
// NOT an approximation for the scales that ship. Both factors are bf16, so their product carries
// at most 16 mantissa bits and is EXACT in f32 -- the accumulate is accfloat, so nothing rounds.
// The one requirement is that `scale` itself be bf16-representable; the encoder's two resadds use
// 1.0 and 0.5, which are. A scale that is not is silently rounded, so it is checked on the host.
#define RESADD2A_APPLY_READS_C (RESADD2A_NOOP != 5)

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
  [[maybe_unused]] const ::aie::vector<float, kLanes> sv =
      ::aie::broadcast<float, kLanes>(scale);
  [[maybe_unused]] const ::aie::vector<bfloat16, kLanes> svb =
      ::aie::broadcast<bfloat16, kLanes>((bfloat16)scale);
  float *__restrict dst = c + j * kRun;
  for (int i = 0; i < kRuns; ++i)
    for (int q = 0; q < kRun; q += kLanes) {
      ::aie::accum<accfloat, kLanes> bv;
      bv.from_vector(::aie::load_v<kLanes>(b + i * kRun + q), 0);
      float *__restrict p = dst + i * kStride + q;
#if RESADD2A_MAC && RESADD2A_APPLY_MULS && RESADD2A_APPLY_READS_C
      // C + scale*b as ONE accumulate: load C into the accumulator, mac the bf16 operands onto
      // it, narrow once. No separate product to schedule next to the store.
      ::aie::accum<accfloat, kLanes> acc;
      acc.from_vector(::aie::load_v<kLanes>(p), 0);
      acc = ::aie::mac(acc, ::aie::load_v<kLanes>(b + i * kRun + q), svb);
      ::aie::store_v(p, vfile(acc.to_vector<float>()));
#else
#if RESADD2A_APPLY_MULS && RESADD2A_BF16_MUL
      ::aie::vector<float, kLanes> sb =
          ::aie::mul(::aie::load_v<kLanes>(b + i * kRun + q), svb).to_vector<float>();
#elif RESADD2A_APPLY_MULS
      ::aie::vector<float, kLanes> sb =
          ::aie::mul(bv.to_vector<float>(), sv).to_vector<float>();
#else
      ::aie::vector<float, kLanes> sb = bv.to_vector<float>();
#endif
#if RESADD2A_APPLY_READS_C
      ::aie::store_v(p, vfile(::aie::add(::aie::load_v<kLanes>(p), sb)));
#else
      ::aie::store_v(p, vfile(sb));
#endif
#endif
    }
#endif
  event1();
}

}  // extern "C"
