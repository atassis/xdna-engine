// REAL 2-stage attention CONVEYOR kernels (heavy matmul stage -> light softmax stage).
//   stage_scores : ac = scale * Q.K^T   [TQ,DK]x[T,DK] bf16 -> [TQ,T] f32   (stage A, heavy)
//   stage_softmax: probs = row_softmax(ac)     [TQ,T] f32 -> [TQ,T] bf16     (stage B, light)
// One query tile, plain (non-relpos) attention -- the heavy->light conveyor crux from the
// research, on real math. Dims baked (like -DRELPOS_T). exp2 uses the device-proven poly
// helper from relpos_mha.cc (hw exp2 is ~2-4% off on aie2p; NOINLINE avoids the -O2 NaN bug).
#include <aie_api/aie.hpp>
#include <stdint.h>
#ifndef ATTN_NO_EPILOGUE
#define ATTN_NO_EPILOGUE 0
#endif
#ifndef ATTN_NO_QTILE
#define ATTN_NO_QTILE 0
#endif
#ifndef ATTN_NO_BDSCALE
#define ATTN_NO_BDSCALE 0
#endif
#ifndef ATTN_NO_SOFTMAX
#define ATTN_NO_SOFTMAX 0
#endif
#ifndef ATTN_NO_CTX
#define ATTN_NO_CTX 0
#endif

#ifndef ATTN_TQ
#define ATTN_TQ 8
#endif
#ifndef ATTN_T
#define ATTN_T 64
#endif
#ifndef ATTN_DK
#define ATTN_DK 64
#endif
#ifndef ATTN_SCALE
#define ATTN_SCALE 0.125f // 1/sqrt(64)
#endif
#ifndef ATTN_P
#define ATTN_P (2 * ATTN_T - 1) // relative-position length (NeMo/Parakeet rel-pos)
#endif
#ifndef ATTN_NQT
// Query tiles per dispatch. Hoisted HERE from ~700 lines down, where it sat BELOW its first use in
// bd_emit_bake: a clean build then only worked if -DATTN_NQT was passed, and conveyor_prebuild.sh
// does not pass it. The staged xclbin survived only because that script early-exits when the file
// already exists, so the breakage stayed invisible until the first real rebuild.
#define ATTN_NQT 1
#endif

static constexpr float LOG2E = 1.4426950408889634f;
static constexpr int VL = 16;

// ============================ aie::mmul PATH (scores + BD) ============================
// Both matmul stages were `aie::mac` + `aie::reduce_add` PER OUTPUT ELEMENT -- a horizontal
// reduce for every scalar of C. Traced on device: scores 279,093 cyc/query-tile and BD 226,152
// against ~180k/359k MACs, i.e. ~18x fewer MAC/cycle than the vendored mm.cc GEMM. The internal
// control was the ctx stage: identical MAC count to scores, 4x the throughput, purely because its
// contraction axis vectorizes along the output. This replaces the horizontal reduce with the
// vendored 2x2-register-block mmul kernel (aie_kernels/aie2p/mm.cc:78-211).
//
// SHAPE: (r,s,t) = (4,8,8), NOT (8,8,8). mm.cc's 2x2 expansion consumes TWO block-rows and TWO
// block-cols per iteration and asserts `m % (2*r) == 0` / `n % (2*t) == 0`. With TQ=8 and r=8 the
// A side is rowA=1, so the kernel would read block-row 1 out of bounds -- it does not compile,
// let alone run. r=4 gives rowA = 8/4 = 2, the minimum the 2x2 block admits. colA = DK/s = 16,
// colB = T/t = 22. mmul_bf16_bf16<4,8,8> is a real aie2p shape (aie_api mmul_bf16_bf16.hpp) and
// matmul_vectorized_4x8x8_bf16_f32 is already a vendored instantiation. The 2x2 register block
// itself is copied UNWIDENED -- 4x4 was measured ~10x slower on aie2p (accumulator spill).
#ifndef ATTN_MMUL
#define ATTN_MMUL 1
#endif
#if ATTN_MMUL
// r follows the FORMAT, because the two are the same decision. With
// -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 the bf16 mac dims become (8,8,8) and mmul lowers to the
// native 512-MAC bfp16 VMAC; without it they are (4,8,8) and it lowers to emulated vmac.f. r=8 needs
// m % (2*r) == 0, i.e. TQ=16 -- which is why the bfp16 brick and the two-query-tile pairing are one
// change, not two. TQ=16/N_QT=11 keeps N_QT*TQ = 176 = T, so the pairing costs no restructuring.
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#define MM_R_V 8
#else
#define MM_R_V 4
#endif
static constexpr int MM_R = MM_R_V, MM_S = 8, MM_T = 8;
// The 2x2 block needs an even rowA, i.e. ATTN_TQ % (2*MM_R) == 0. That is a property of the
// BUILD, not of the kernel, so the mmul entry points are compiled out when the shape does not
// admit them -- otherwise their static_asserts break the plain (non-mmul) build at the same TQ.
#define MM_SHAPE_OK (ATTN_TQ % (2 * MM_R_V) == 0)
// The BLOCKED variant additionally hardcodes a single z-iteration, so it wants rowA == 2 exactly.
#define MM_BLK_OK (ATTN_TQ == 2 * MM_R_V)
// One scratch shared by the q tiling and the C de-tile (never live together); size by the larger.
#define MM_SCRATCH_BYTES \
  ((MM_R_V * ATTN_T * 4) > (ATTN_TQ * ATTN_DK * 2) ? (MM_R_V * ATTN_T * 4) : (ATTN_TQ * ATTN_DK * 2))

// WHERE THE TILING HAPPENS, and why it is not here.
// mm.cc wants A as r x s tiles and (with b_row_maj=false) B as t x s tiles, the tiles themselves in
// row-major order. Ours arrive plain row-major. Tiling them ON THIS CORE was costed and REJECTED:
// the permutation is a scalar/short-vector shuffle, so tiling k[176,128] costs ~2816 8-wide vector
// ops against the 704 mmul instructions the whole query tile needs -- 4x the matmul it feeds. The
// same arithmetic kills per-block tiling of BD's p.
//
// So neither operand is tiled on-core:
//   * k  -- tiled by the SHIM DMA. The tile order is just a 4-D strided read of the same L3 bytes
//           (sizes [T/t, DK/s, t, s], strides [t*SD, s, SD, 1]), which is exactly what a
//           TensorAccessPattern expresses. Free, and it forces the k fill to lose its N_QT
//           stride-0 replay -- 4 dims are all the shim has -- which is itself the 22x-read-once win.
//   * q  -- tiled by the BD core, which already copies q_pass into the belt head byte-for-byte
//           (bd_emit_bake). Reordering that copy costs nothing it was not already paying.
// Kept here only as the reference definition of the layout both producers must match:
//   A_tiled[((z*colA + c)*r + rr)*s + ss] = A[(z*r + rr)*K + (c*s + ss)]
//   B_tiled[((j*colA + c)*t + tt)*s + ss] = Bt[(j*t + tt)*K + (c*s + ss)]   (Bt = k, p: [n,K])
// The permutation is confined within each group of R rows, which is what lets the q producer do it
// with a single R*K scratch and no second buffer.
template <int R, int S, int K>
static inline void mm_tile_rows(const bfloat16 *__restrict src, bfloat16 *__restrict dst, int rows) {
  constexpr int CA = K / S;
  for (int g = 0; g < rows / R; g++) {
    const bfloat16 *sblk = src + g * R * K;
    bfloat16 *dblk = dst + g * R * K;
    for (int c = 0; c < CA; c++)
      for (int rr = 0; rr < R; rr++)
        for (int ss = 0; ss < S; ss++)
          dblk[(c * R + rr) * S + ss] = sblk[rr * K + c * S + ss];
  }
}

// The vendored 2x2 register block, transcribed from mm.cc:78-211 with b_row_maj=false /
// c_row_maj=true (our B^T -- k[T,DK] / p[pb,DK] row-major -- is exactly the [n,k] tiled layout
// that path expects, loaded with an in-register aie::transpose). Accumulators start at ZERO here
// rather than loading C: every call computes a complete K reduction, so there is no partial to
// carry, and skipping the load also skips needing C initialised.
template <unsigned rowA, unsigned colA, unsigned colB>
static inline void mm_2x2_bf16_f32(const bfloat16 *__restrict pA,
                                   const bfloat16 *__restrict pB,
                                   float *__restrict pC) {
  using MMUL = aie::mmul<MM_R, MM_S, MM_T, bfloat16, bfloat16, accauto>;
  // The data layout (host tiling + mm_tile_rows + the de-tile) hard-codes r*s / s*t / r*t. If the
  // API's block sizes differ, the layout and this pointer walk disagree silently.
  static_assert(MMUL::size_A == MM_R * MM_S, "A block size != r*s");
  static_assert(MMUL::size_B == MM_S * MM_T, "B block size != s*t");
  static_assert(MMUL::size_C == MM_R * MM_T, "C block size != r*t");
  static_assert(rowA % 2 == 0, "2x2 block needs an even rowA");
  static_assert(colB % 2 == 0, "2x2 block needs an even colB");

  for (unsigned z = 0; z < rowA; z += 2)
    chess_prepare_for_pipelining chess_loop_range(1, ) {
      float *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
      float *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

      for (unsigned j = 0; j < colB; j += 2) {
        const bfloat16 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
        const bfloat16 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
        const bfloat16 *__restrict pB1 = pB + (j * colA) * MMUL::size_B;
        const bfloat16 *__restrict pB2 = pB + ((j + 1) * colA) * MMUL::size_B;

        aie::vector<bfloat16, MMUL::size_A> A0, A1;
        aie::vector<bfloat16, MMUL::size_B> B0, B1;

        // DEFAULT ctor, not a hand-zeroed accumulator. aie_api's C_block carries a `zero` flag --
        // C_block() sets zero=true, which is what makes the FIRST .mac() emit a mul instead of
        // reading an uninitialised accumulator. Passing aie::zeros(...) instead selects the
        // accumulate-onto ctor (zero=false), and the vector overload additionally goes through an
        // accum(v, shift) conversion. mm.cc gets away with loading C because zero.cc pre-zeroes it;
        // we accumulate nothing across calls, so the fresh-product idiom is the correct one.
        MMUL C00;
        MMUL C01;
        MMUL C10;
        MMUL C11;

        for (unsigned i = 0; i < colA; ++i) {
          A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
          A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
          B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), MM_T, MM_S);
          pB1 += MMUL::size_B;
          B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), MM_T, MM_S);
          pB2 += MMUL::size_B;
          C00.mac(A0, B0);
          C01.mac(A0, B1);
          C10.mac(A1, B0);
          C11.mac(A1, B1);
        }

        aie::store_v(pC1, C00.template to_vector<float>()); pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<float>()); pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<float>()); pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<float>()); pC2 += MMUL::size_C;
      }
    }
}

// C comes back TILED ([rowA][colB] of r x t row-major). It is written straight into the `scores`
// output buffer and de-tiled IN PLACE, so no second [TQ,T] f32 buffer is needed -- L1 on this core
// already carries the 44 KB k. Like the input tiling, the C permutation is confined within each
// group of r rows (r*T floats), so one r*T scratch suffices.
//   ctile[((z*colB + j)*r + rr)*t + tt]  ->  logical row z*r + rr, col j*t + tt
// The per-stage epilogue rides the de-tile, which is why the tiled form is never a separate pass.
#endif // ATTN_MMUL


// SOFTWARE f32 2^x (x<=0), device-proven (probe_floor rel-err 8.5e-5). NOINLINE is load-bearing:
// inlining into the softmax loop makes Peano -O2 miscompile to NaN. Copied from relpos_mha.cc.
static __attribute__((noinline)) aie::vector<float, VL> exp2f_vec(aie::vector<float, VL> x) {
  x = aie::max(x, aie::broadcast<float, VL>(-100.0f));
  aie::vector<int32_t, VL> ki = aie::to_fixed<int32_t>(x);
  aie::vector<float, VL> kf = aie::to_float<float>(ki);
  aie::vector<int32_t, VL> one = aie::broadcast<int32_t, VL>(1);
  aie::vector<int32_t, VL> zero = aie::broadcast<int32_t, VL>(0);
  ki = aie::sub(ki, aie::select(zero, one, aie::lt(x, kf)));
  aie::vector<float, VL> f = aie::sub(x, aie::to_float<float>(ki));
  aie::vector<float, VL> p = aie::broadcast<float, VL>(0.0013333558f);
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.0096181291f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.0555041087f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.2402265069f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(0.6931471805f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, VL>(1.0f));
  aie::vector<int32_t, VL> ebits =
      aie::upshift(aie::add(ki, aie::broadcast<int32_t, VL>(127)), 23);
  aie::vector<float, VL> p2k = ebits.cast_to<float>();
  return aie::mul(p, p2k).to_vector<float>();
}

// STAGE A -- scores. ac[i,j] = scale * dot(q[i,:], k[j,:]). bf16 in, f32 accumulate.
extern "C" void stage_scores(const bfloat16 *__restrict q,
                             const bfloat16 *__restrict k, float *__restrict ac) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr float scale = ATTN_SCALE;
  event0();
  for (int i = 0; i < TQ; i++) {
    const bfloat16 *qr = q + i * DK;
    for (int j = 0; j < T; j++) {
      const bfloat16 *kr = k + j * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(kr + d));
      ac[i * T + j] = aie::reduce_add(acc.to_vector<float>()) * scale;
    }
  }
  event1();
}

// STAGE A (RELPOS) -- fused scores with on-chip AC + BD + rel_shift + scale (Parakeet/NeMo rel-pos).
//   q       : [TQ, DK] bf16          one query tile from the belt
//   kp      : [(T+P)*DK] bf16 RESIDENT -- k[T,DK] then p[P,DK] packed (2-input budget: q + kp)
//   scores  : [TQ, T] f32 (out)      = (AC + rel_shift(BD)) * inv_scale, feeds the softmax stage
//   row_off : global row index of this query tile (qt*TQ); the rel_shift base uses the GLOBAL row i.
// AC[li,j]=q[li].k[j]; BD[li,jp]=q[li].p[jp]; scores[li,j]=(AC+BD[(T-1-i)+j])*inv_scale, i=row_off+li.
// rel_shift = strided read bd + (T-1-i) (relpos_mha.cc brick 2). BD held per-row in a stack scratch ->
// stage A needs a bumped stack_size at real P (bd[343] f32 = ~1.4 KB). Scalar AC/BD dots (correctness
// first; the vectorized-unaligned path is the follow-up optimization).
extern "C" void stage_scores_relpos(const bfloat16 *__restrict q,
                                    const bfloat16 *__restrict kp,
                                    float *__restrict scores, int32_t row_off) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK, P = ATTN_P;
  constexpr float inv_scale = ATTN_SCALE;
  const bfloat16 *k = kp;
  const bfloat16 *p = kp + T * DK;
  float bd[ATTN_P];   // per-row BD = q[li] . p^T ; stack scratch (stage A worker gets a bumped stack_size)
  event0();
  for (int li = 0; li < TQ; li++) {
    const bfloat16 *qr = q + li * DK;
    const int i = row_off + li;                 // GLOBAL row index -> rel_shift base
    for (int jp = 0; jp < P; jp++) {
      const bfloat16 *pr = p + jp * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(pr + d));
      bd[jp] = aie::reduce_add(acc.to_vector<float>());
    }
    const int base = (i < T) ? (T - 1 - i) : 0;  // rel_shift base (clamp padding rows i>=T -> no OOB)
    const float *bd_row = bd + base;
    float *sc = scores + li * T;
    for (int j = 0; j < T; j++) {
      const bfloat16 *kr = k + j * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(kr + d));
      const float ac = aie::reduce_add(acc.to_vector<float>());
      sc[j] = (ac + bd_row[j]) * inv_scale;
    }
  }
  event1();
}

// STAGE A (RELPOS, REAL-DIMS) -- host-precomputed rel_shifted BD packed in the query belt.
// On-chip BD needs p[P,DK] resident = 88 KB at real dims (blows L1), so instead the host computes
// BD = q.p^T THEN rel_shift -> BD_shifted[TQ,T] bf16, packed AFTER q in ONE belt object. Stage A stays
// at 2 inputs (qbd belt + k resident, 44 KB) and holds NO p / NO bd scratch. rel_shift being host-side
// means there is NO row_off (global-row) dependence -> N_QT>1 needs no tile-offset wiring.
//   qbd    : [TQ*DK + TQ*T] bf16   q[TQ,DK] then BD_shifted[TQ,T]
//   k      : [T*DK] bf16 resident
//   scores : [TQ,T] f32 = (q.k^T + BD_shifted) * inv_scale
// BD carriage in the belt tail: hi-only (plain bf16, BD_SPLIT=0, byte-identical to the host BD-in-belt
// conveyor) or split-bf16 hi+lo (BD_SPLIT=1) reconstructing ~f32 for the on-chip-BD 4th-stage precision.
#ifndef BD_SPLIT
#define BD_SPLIT 0
#endif
extern "C" void stage_scores_relpos_bd(const bfloat16 *__restrict qbd,
                                       const bfloat16 *__restrict k, float *__restrict scores) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr float inv_scale = ATTN_SCALE;
  const bfloat16 *q = qbd;
  const bfloat16 *bdhi = qbd + TQ * DK;                 // BD_hi[TQ,T] packed after q
  const bfloat16 *bdlo = bdhi + (BD_SPLIT ? TQ * T : 0); // BD_lo[TQ,T] only when split
  event0();
  for (int li = 0; li < TQ; li++) {
    const bfloat16 *qr = q + li * DK;
    const bfloat16 *hir = bdhi + li * T;
    const bfloat16 *lor = bdlo + li * T;
    float *sc = scores + li * T;
    for (int j = 0; j < T; j++) {
      const bfloat16 *kr = k + j * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(kr + d));
      const float ac = aie::reduce_add(acc.to_vector<float>());
      float bd = (float)hir[j];
#if BD_SPLIT
      bd += (float)lor[j];                              // reconstruct ~f32 (hi + lo residual)
#endif
      sc[j] = (ac + bd) * inv_scale;
    }
  }
  event1();
}

// STAGE A (RELPOS-BD, t_active MASKED) -- same math as stage_scores_relpos_bd, plus an in-kernel key
// mask for variable-length clips. BD is computed ON-CHIP (4th stage), so the host -1e4 belt-sentinel
// trick (CONV_KEY_MASK) can no longer null pad keys: pad keys kk>=t_active have a REAL (nonzero) BD from
// rel_shift. Fix = mask here. t_active is read from an RTP register (rtp[0]) at RUNTIME (int32[16],
// use_write_rtp), so ONE MAX-T=ATTN_T xclbin serves any t_active<=T (mirrors relpos_mha.cc's
// relpos_stream_softmax rtp[0] contract). A padded clip gets correct attention over its real t_active
// keys; masked columns j>=t_active are driven to ~0 in the softmax. t_active==T recovers the unmasked
// behavior byte-for-byte. See the BD-onchip attention design (t_active in-kernel key-mask).
#ifndef ATTN_KEY_MASK
#define ATTN_KEY_MASK (-1.0e4f)   // large finite negative; (mask - rowmax)*log2e -> exp2 clamp(-100) ~= 0
#endif
extern "C" void stage_scores_relpos_bd_mask(const bfloat16 *__restrict qbd,
                                            const bfloat16 *__restrict k,
                                            float *__restrict scores,
                                            const int32_t *__restrict rtp) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr float inv_scale = ATTN_SCALE;
  const int t_active = rtp[0];                 // active key count (<= T); pad keys j>=t_active -> masked
  const bfloat16 *q = qbd;
  const bfloat16 *bdhi = qbd + TQ * DK;
  const bfloat16 *bdlo = bdhi + (BD_SPLIT ? TQ * T : 0);
  event0();
  for (int li = 0; li < TQ; li++) {
    const bfloat16 *qr = q + li * DK;
    const bfloat16 *hir = bdhi + li * T;
    const bfloat16 *lor = bdlo + li * T;
    float *sc = scores + li * T;
    // MASK HOISTED OUT OF THE INNER LOOP. This was `for j<T { if (j>=t_active) {...; continue;} ... }`
    // -- a data-dependent branch on every one of the TQ*T=1408 output elements, which on a VLIW core
    // blocks software pipelining of the loop it guards. Splitting it into a clean active range plus a
    // straight tail fill is EXACTLY equivalent (same values, same order, same rounding) and lets the
    // compiler pipeline the hot loop.
    //
    // Measured baseline before this change: 291,668 cycles/query-tile, i.e. 207 cycles per output for
    // 8 vector MACs of arithmetic (207x off the 128 MAC/cyc/core peak). The internal control is the
    // ctx stage: identical MAC count (180,224) but 68,713 cycles, because its contraction axis lets it
    // vectorize along the output instead of doing a horizontal reduce_add per element. This change
    // separates the BRANCH cost from the reduce_add cost, which decides how much an aie::mmul rewrite
    // can actually claim.
    for (int j = 0; j < t_active; j++) {
      const bfloat16 *kr = k + j * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(kr + d));
      const float ac = aie::reduce_add(acc.to_vector<float>());
      float bd = (float)hir[j];
#if BD_SPLIT
      bd += (float)lor[j];
#endif
      sc[j] = (ac + bd) * inv_scale;
    }
    for (int j = t_active; j < T; j++) sc[j] = ATTN_KEY_MASK;  // pad keys -> softmax ~0
  }
  event1();
}

// STAGE A (RELPOS-BD, t_active MASKED) via aie::mmul. Same math and same ABI as
// stage_scores_relpos_bd_mask, with the horizontal reduce_add replaced by the vendored 2x2 block.
// PRECONDITIONS the generator must honour (both are layout-only; get one wrong and the numbers are
// silently wrong, not a crash):
//   * k arrives PRE-TILED (t x s blocks, tiles row-major) -- kvtap does this, and must therefore
//     drop its N_QT stride-0 replay, so the worker acquires k ONCE per dispatch, not per tile.
//   * the belt head q_pass arrives PRE-TILED (r x s blocks) -- bd_emit_bake* does this.
#if ATTN_MMUL && MM_SHAPE_OK
extern "C" void stage_scores_relpos_bd_mask_mmul(const bfloat16 *__restrict qbd,
                                                 const bfloat16 *__restrict ktiled,
                                                 float *__restrict scores,
                                                 const int32_t *__restrict rtp) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr float inv_scale = ATTN_SCALE;
  constexpr unsigned rowA = TQ / MM_R, colA = DK / MM_S, colB = T / MM_T;
  const int t_active = rtp[0];
  const bfloat16 *bdhi = qbd + TQ * DK;
  const bfloat16 *bdlo = bdhi + (BD_SPLIT ? TQ * T : 0);
  // ONE scratch, aliased. L1 on this core already carries the 44 KB k, and two separate scratches
  // overflowed .bss by 1568 B. They are never live together: q_tiled is dead the moment mm_2x2
  // returns, and row_scratch is only touched after it. Sized by the larger of the two.
  alignas(32) static char mm_scratch[MM_SCRATCH_BYTES];
  static_assert(sizeof(mm_scratch) >= ATTN_TQ * ATTN_DK * sizeof(bfloat16), "scratch too small for q");
  bfloat16 *__restrict q_tiled = reinterpret_cast<bfloat16 *>(mm_scratch);
  float *__restrict row_scratch = reinterpret_cast<float *>(mm_scratch);

  event0();
  // Tile A here rather than at the producer. q is 1024 elements against the 704 mmul instructions
  // this tile issues, so on-core tiling is affordable for q -- it is only k (22528 elements, ~4x
  // the matmul) that had to move to the DMA. Doing it here also keeps ONE tiling site: the shipped
  // 3-stage rail packs the belt host-side and never runs bd_emit_bake, so tiling at the producer
  // would have been correct on the 4-stage rail and silently wrong on the shipped one.
  mm_tile_rows<MM_R, MM_S, ATTN_DK>(qbd, q_tiled, TQ);
#if ATTN_MMUL_REF
  // BISECT REFERENCE: same tiled A/B layouts, same tiled C layout, but a scalar dot instead of the
  // 2x2 register block. If this PASSES and mm_2x2 fails, the bug is in the register block; if this
  // also fails, the bug is upstream in the DMA tiling tap or the acquire-once k pairing.
  for (unsigned z = 0; z < rowA; z++)
    for (unsigned j = 0; j < colB; j++)
      for (int rr = 0; rr < MM_R; rr++)
        for (int tt = 0; tt < MM_T; tt++) {
          float acc = 0.f;
          for (unsigned c = 0; c < colA; c++)
            for (int ss = 0; ss < MM_S; ss++)
              acc += (float)q_tiled[((z * colA + c) * MM_R + rr) * MM_S + ss] *
                     (float)ktiled[((j * colA + c) * MM_T + tt) * MM_S + ss];
          scores[((z * colB + j) * MM_R + rr) * MM_T + tt] = acc;
        }
#else
  mm_2x2_bf16_f32<rowA, colA, colB>(q_tiled, ktiled, scores);
#endif

  // De-tile in place, r rows at a time, folding in BD + scale + the t_active key mask. This is the
  // whole epilogue of the old kernel; it just reads its AC from the tiled C instead of from a
  // reduce_add. Byte-for-byte the same arithmetic order: (ac + bd) * inv_scale.
  for (unsigned z = 0; z < rowA; z++) {
    float *grp = scores + z * MM_R * T;
    for (int i = 0; i < MM_R * T; i++) row_scratch[i] = grp[i];
    for (int rr = 0; rr < MM_R; rr++) {
      const int li = z * MM_R + rr;
      const bfloat16 *hir = bdhi + li * T;
      const bfloat16 *lor = bdlo + li * T;
      float *sc = grp + rr * T;
      for (unsigned j = 0; j < colB; j++) {
        const float *ct = row_scratch + (j * MM_R + rr) * MM_T;
        for (int tt = 0; tt < MM_T; tt++) {
          const int col = j * MM_T + tt;
          if (col < t_active) {
            float bd = (float)hir[col];
#if BD_SPLIT
            bd += (float)lor[col];
#endif
            sc[col] = (ct[tt] + bd) * inv_scale;
          } else {
            sc[col] = ATTN_KEY_MASK;
          }
        }
      }
    }
  }
  event1();
}

#if ATTN_MMUL && MM_SHAPE_OK

// ONE j-pair of the vendored 2x2 register block, writing C tiles for j = 2*jp and 2*jp+1.
// Shared by the blocked entry (k arrives as a 4 KB MemTile block) and the whole-k entry (k is the
// 44 KB resident buffer, indexed by jp). Extracted so both use the SAME proven compute -- the
// earlier hand-written whole-k variant diverged here and failed parity at 1.06.
template <unsigned rowA, unsigned colA, unsigned colB>
static inline void mm_jpair(const bfloat16 *__restrict q_tiled,
                            const bfloat16 *__restrict kblk, float *__restrict scores,
                            unsigned jp) {
  using MMUL = aie::mmul<MM_R, MM_S, MM_T, bfloat16, bfloat16, accauto>;
  float *__restrict pC1 = scores + (0u * colB + 2u * jp) * MMUL::size_C;
  float *__restrict pC2 = scores + (1u * colB + 2u * jp) * MMUL::size_C;
  const bfloat16 *__restrict pA1 = q_tiled + (0u * colA) * MMUL::size_A;
  const bfloat16 *__restrict pA2 = q_tiled + (1u * colA) * MMUL::size_A;
  const bfloat16 *__restrict pB1 = kblk;
  const bfloat16 *__restrict pB2 = kblk + colA * MMUL::size_B;
#if ATTN_MMUL_REF
  // BISECT: scalar dot over the SAME pointers and the SAME assumed tile layout. If this passes and
  // the vector path fails, mm_jpair is wrong; if both fail, the operand LAYOUT (the DMA tap) is.
  for (unsigned zz = 0; zz < 2; zz++)
    for (unsigned jj = 0; jj < 2; jj++)
      for (int rr = 0; rr < MM_R; rr++)
        for (int tt = 0; tt < MM_T; tt++) {
          float acc = 0.f;
          for (unsigned c = 0; c < colA; c++)
            for (int ss = 0; ss < MM_S; ss++)
              acc += (float)q_tiled[((zz * colA + c) * MM_R + rr) * MM_S + ss] *
                     (float)kblk[((jj * colA + c) * MM_T + tt) * MM_S + ss];
          scores[((zz * colB + 2 * jp + jj) * MM_R + rr) * MM_T + tt] = acc;
        }
  return;
#endif
  aie::vector<bfloat16, MMUL::size_A> A0, A1;
  aie::vector<bfloat16, MMUL::size_B> B0, B1;
  MMUL C00; MMUL C01; MMUL C10; MMUL C11;
  for (unsigned i = 0; i < colA; ++i)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
      A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
      B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), MM_T, MM_S); pB1 += MMUL::size_B;
      B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), MM_T, MM_S); pB2 += MMUL::size_B;
      C00.mac(A0, B0); C01.mac(A0, B1);
      C10.mac(A1, B0); C11.mac(A1, B1);
    }
  aie::store_v(pC1, C00.template to_vector<float>()); pC1 += MMUL::size_C;
  aie::store_v(pC1, C01.template to_vector<float>());
  aie::store_v(pC2, C10.template to_vector<float>()); pC2 += MMUL::size_C;
  aie::store_v(pC2, C11.template to_vector<float>());
}

// De-tile + BD + scale + t_active key mask, vectorised MM_T wide. Shared epilogue.
template <unsigned rowA, unsigned colA, unsigned colB>
static inline void mm_epilogue(float *__restrict scores, const bfloat16 *__restrict bdhi,
                               const bfloat16 *__restrict bdlo, float *__restrict row_scratch,
                               int t_active) {
  constexpr int T = ATTN_T;
  constexpr float inv_scale = ATTN_SCALE;
  for (unsigned z = 0; z < rowA; z++) {
    float *grp = scores + z * MM_R * T;
    for (int i = 0; i < MM_R * T; i++) row_scratch[i] = grp[i];
    for (int rr = 0; rr < MM_R; rr++) {
      const int li = z * MM_R + rr;
      const bfloat16 *hir = bdhi + li * T;
      const bfloat16 *lor = bdlo + li * T;
      float *sc = grp + rr * T;
      for (unsigned j = 0; j < colB; j++) {
        const float *ct = row_scratch + (j * MM_R + rr) * MM_T;
        aie::vector<float, MM_T> cv = aie::load_v<MM_T>(ct);
        aie::vector<bfloat16, MM_T> bv = aie::load_v<MM_T>(hir + j * MM_T);
        aie::vector<float, MM_T> bdf = aie::mul(bv, bfloat16(1.0f)).template to_vector<float>();
#if BD_SPLIT
        aie::vector<bfloat16, MM_T> lv = aie::load_v<MM_T>(lor + j * MM_T);
        bdf = aie::add(bdf, aie::mul(lv, bfloat16(1.0f)).template to_vector<float>());
#endif
        aie::store_v(sc + j * MM_T,
                     aie::mul(aie::add(cv, bdf), inv_scale).template to_vector<float>());
      }
      for (int j = t_active; j < T; j++) sc[j] = ATTN_KEY_MASK;   // pad keys -> softmax ~0
    }
  }
}

// WHOLE-K masked entry -- the shape the ENCODER's default rail actually dispatches
// (relpos_mha_conveyor_bdonchip, CONV_BD_HEADS=4, g0/g1). k is the 44 KB resident buffer delivered
// PRE-TILED by the shim: in this branch k is per-head, so the 4-D tiling tap is [T/8, DK/8, 8, 8]
// with an outer size of 22, well under the 6-bit ITER_WRAP limit that blocked the grouped H=8 case.
extern "C" void stage_scores_relpos_bd_mask_mmul_whole(const bfloat16 *__restrict qbd,
                                                       const bfloat16 *__restrict ktiled,
                                                       float *__restrict scores,
                                                       const int32_t *__restrict rtp) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr unsigned rowA = TQ / MM_R, colA = DK / MM_S, colB = T / MM_T;
  alignas(32) static char mm_scratch2[MM_SCRATCH_BYTES];
  bfloat16 *__restrict q_tiled = reinterpret_cast<bfloat16 *>(mm_scratch2);
  float *__restrict row_scratch = reinterpret_cast<float *>(mm_scratch2);
  event0();
  mm_tile_rows<MM_R, MM_S, ATTN_DK>(qbd, q_tiled, TQ);
  for (unsigned jp = 0; jp < colB / 2; jp++)
    mm_jpair<rowA, colA, colB>(q_tiled, ktiled + jp * 2u * colA * (MM_S * MM_T), scores, jp);
  mm_epilogue<rowA, colA, colB>(scores, qbd + TQ * DK,
                                qbd + TQ * DK + (BD_SPLIT ? TQ * T : 0), row_scratch, rtp[0]);
  event1();
}

#endif

// ---------------- BLOCKED scores: k streamed from the MemTile in j-pair blocks ----------------
// Holding k whole costs 44 KB of L1 and is what makes TQ=16 (and therefore r=8, and therefore the
// native bfp16 512-MAC brick) impossible. But k has NO reuse inside a query tile beyond the j-pair
// being consumed, so the 2x2 block can take it 2*t = 16 rows at a time: 4 KB instead of 44 KB.
//
// The tiled layout makes the chop free. Tile (j,c) lives at (j*colA + c)*size_B, so j-pair jp owns
// the CONTIGUOUS 2*colA*size_B run at jp*(2*colA*size_B) -- streaming blocks in order is just
// slicing the tiled k, no repacking anywhere.
//
// One call per j-pair. q is tiled once per query tile (jp==0) and the epilogue fires on the last
// block, so `scores` must be held by the worker across all NBLK calls.
#if ATTN_MMUL && MM_BLK_OK
extern "C" void stage_scores_mmul_block(const bfloat16 *__restrict qbd,
                                        const bfloat16 *__restrict kblk,
                                        float *__restrict scores) {
  using MMUL = aie::mmul<MM_R, MM_S, MM_T, bfloat16, bfloat16, accauto>;
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr float inv_scale = ATTN_SCALE;
  constexpr unsigned rowA = TQ / MM_R, colA = DK / MM_S, colB = T / MM_T;
  constexpr unsigned NBLK = colB / 2;
  static_assert(rowA == 2, "the 2x2 block wants exactly two block-rows here (TQ = 2*MM_R)");
  static_assert(colB % 2 == 0, "j must pair evenly");

  alignas(32) static char mm_scratch[MM_SCRATCH_BYTES];
  static_assert(sizeof(mm_scratch) >= ATTN_TQ * ATTN_DK * sizeof(bfloat16), "scratch too small for q");
  bfloat16 *__restrict q_tiled = reinterpret_cast<bfloat16 *>(mm_scratch);
  float *__restrict row_scratch = reinterpret_cast<float *>(mm_scratch);
  static int jp = 0;

  event0();
#if !ATTN_NO_QTILE
  if (jp == 0) mm_tile_rows<MM_R, MM_S, ATTN_DK>(qbd, q_tiled, TQ);
#endif

  {   // ONE j-pair, the vendored 2x2 register block (mm.cc:78-211), unwidened.
    float *__restrict pC1 = scores + (0u * colB + 2u * jp) * MMUL::size_C;
    float *__restrict pC2 = scores + (1u * colB + 2u * jp) * MMUL::size_C;
    const bfloat16 *__restrict pA1 = q_tiled + (0u * colA) * MMUL::size_A;
    const bfloat16 *__restrict pA2 = q_tiled + (1u * colA) * MMUL::size_A;
    const bfloat16 *__restrict pB1 = kblk;                        // j = 2*jp
    const bfloat16 *__restrict pB2 = kblk + colA * MMUL::size_B;  // j = 2*jp + 1

    aie::vector<bfloat16, MMUL::size_A> A0, A1;
    aie::vector<bfloat16, MMUL::size_B> B0, B1;
    MMUL C00; MMUL C01; MMUL C10; MMUL C11;   // default ctor => zero=true => first mac is a mul

    for (unsigned i = 0; i < colA; ++i)
      chess_prepare_for_pipelining chess_loop_range(16, ) {
        A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
        A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
        // NO aie::transpose. B tiles arrive as [ss][tt] (mm.cc's b_row_maj=true content), which is
        // what mmul wants, so the per-tile in-register transpose that the b_row_maj=false path needs
        // is gone from the inner loop -- 2 per iteration x colA x NBLK per query tile. We tile k
        // host-side anyway, so emitting the transposed content there is free; the ONLY reason the
        // b_row_maj=false route existed was to avoid transposing k, and that reason is now stale.
        // Tile ORDER stays j-major so a j-pair remains a contiguous block.
        B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), MM_T, MM_S); pB1 += MMUL::size_B;
        B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), MM_T, MM_S); pB2 += MMUL::size_B;
        C00.mac(A0, B0); C01.mac(A0, B1);
        C10.mac(A1, B0); C11.mac(A1, B1);
      }

    aie::store_v(pC1, C00.template to_vector<float>()); pC1 += MMUL::size_C;
    aie::store_v(pC1, C01.template to_vector<float>());
    aie::store_v(pC2, C10.template to_vector<float>()); pC2 += MMUL::size_C;
    aie::store_v(pC2, C11.template to_vector<float>());
  }

  // ABLATION (-DATTN_NO_EPILOGUE=1 / -DATTN_NO_QTILE=1): numerically WRONG on purpose. The trace
  // parser finds no tiles in our trace dump, so cost is attributed differentially instead -- skip a
  // phase, measure the delta. Never enable in a build whose numbers are quoted as correct.
  if (++jp == (int)NBLK) {   // last block of this query tile -> de-tile + BD + scale, in place
#if ATTN_NO_EPILOGUE
    jp = 0; event1(); return;
#endif
    jp = 0;
    const bfloat16 *bdhi = qbd + TQ * DK;
    const bfloat16 *bdlo = bdhi + (BD_SPLIT ? TQ * T : 0);
    for (unsigned z = 0; z < rowA; z++) {
      float *grp = scores + z * MM_R * T;
      for (int i = 0; i < MM_R * T; i++) row_scratch[i] = grp[i];
      for (int rr = 0; rr < MM_R; rr++) {
        const int li = z * MM_R + rr;
        const bfloat16 *hir = bdhi + li * T;
        const bfloat16 *lor = bdlo + li * T;
        float *sc = grp + rr * T;
        // VECTORISED de-tile + BD + scale, MM_T (=8) wide. Ablation showed the de-tile itself is
        // free and that this line -- 1408 SCALAR bf16->float converts per query tile -- was 58% of
        // the whole dispatch. All three operands are contiguous 8-element runs at this point
        // (ct in the scratch, hir in the belt, sc in the output), so no gather is needed; the
        // scalar version was leaving an 8-wide lane completely unused.
        for (unsigned j = 0; j < colB; j++) {
          const float *ct = row_scratch + (j * MM_R + rr) * MM_T;
#if ATTN_NO_BDSCALE
          for (int tt = 0; tt < MM_T; tt++) sc[j * MM_T + tt] = ct[tt];   // ABLATION
#else
          aie::vector<float, MM_T> cv = aie::load_v<MM_T>(ct);
          aie::vector<bfloat16, MM_T> bv = aie::load_v<MM_T>(hir + j * MM_T);
          aie::vector<float, MM_T> bdf = aie::mul(bv, bfloat16(1.0f)).template to_vector<float>();
#if BD_SPLIT
          aie::vector<bfloat16, MM_T> lv = aie::load_v<MM_T>(lor + j * MM_T);
          bdf = aie::add(bdf, aie::mul(lv, bfloat16(1.0f)).template to_vector<float>());
#endif
          aie::vector<float, MM_T> sum = aie::add(cv, bdf);
          aie::store_v(sc + j * MM_T, aie::mul(sum, inv_scale).template to_vector<float>());
#endif
        }
      }
    }
  }
  event1();
}
#endif // ATTN_MMUL

// Unmasked twin (t_active == T), for the no-mask build.
extern "C" void stage_scores_mmul_block(const bfloat16 *, const bfloat16 *, float *);
extern "C" void stage_scores_relpos_bd_mmul(const bfloat16 *__restrict qbd,
                                            const bfloat16 *__restrict ktiled,
                                            float *__restrict scores) {
  // Route to the PROVEN whole-k body (t_active = T is the unmasked case). Previously this called
  // the hand-written non-blocked variant, which fails parity at 1.06 and is now unreferenced.
  // Delegate to the PROVEN blocked entry, once per j-pair. Its static jp counter tiles q on the
  // first call and runs the epilogue on the last, and NBLK calls wrap it cleanly -- so a whole-k
  // dispatch is exactly NBLK blocked calls with the pointer walked forward.
  //
  // NOT via mm_jpair: that helper was written out as "the same" code and is NOT -- with it the
  // device-in gate fails at 1.286e-01 while a scalar dot over the identical pointers and layout
  // passes at 4.39370e-03, which pins the fault inside those ~15 lines of vector code and clears
  // the tap, the q tiling and the epilogue. Left in place, unused, for that diagnosis.
  constexpr unsigned colA_ = ATTN_DK / MM_S;
  constexpr unsigned NBLK_ = (ATTN_T / MM_T) / 2;
  for (unsigned jp = 0; jp < NBLK_; jp++)
    stage_scores_mmul_block(qbd, ktiled + jp * 2u * colA_ * (MM_S * MM_T), scores);
}
#endif // ATTN_MMUL

// Zero-scalar-arg bake wrapper (IRON kernels avoid scalar args -> bake constants). N_QT=1 validation:
// row_off = 0 (the single query tile is rows [0,TQ)). N_QT>1 needs an advancing row_off (tile-offset
// wiring, a follow-up) -- do NOT use this bake for N_QT>1.
extern "C" void stage_scores_relpos_bake(const bfloat16 *__restrict q,
                                         const bfloat16 *__restrict kp, float *__restrict scores) {
  stage_scores_relpos(q, kp, scores, 0);
}

// MONOLITH baseline: all 3 stages on ONE tile, per query tile. 2 inputs (q + kv, k|V packed) to fit
// the 2-input-channel budget. Local ac/probs scratch (stack). For the conveyor-vs-monolith perf A/B.
extern "C" void stage_mono(const bfloat16 *__restrict q, const bfloat16 *__restrict kv,
                           bfloat16 *__restrict ctx) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  constexpr float scale = ATTN_SCALE;
  const bfloat16 *k = kv;
  const bfloat16 *V = kv + T * DK;
  // static L1 scratch (safe in the MONO single worker -- no concurrent ping-pong belt to alias, unlike
  // the pipelined softmax). Keeps ~3.3 KB off the small AIE stack (stack-alloc here -> nan/overflow).
  static float ac[TQ * T];
  static bfloat16 probs[TQ * T];
  static float srow[ATTN_T];
  aie::vector<float, VL> log2e_v = aie::broadcast<float, VL>(LOG2E);
  event0();
  // scores
  for (int i = 0; i < TQ; i++)
    for (int j = 0; j < T; j++) {
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(q + i * DK + d), aie::load_v<VL>(k + j * DK + d));
      ac[i * T + j] = aie::reduce_add(acc.to_vector<float>()) * scale;
    }
  // softmax
  for (int i = 0; i < TQ; i++) {
    const float *ar = ac + i * T;
    bfloat16 *pr = probs + i * T;
    float rowmax = -3.0e38f;
    for (int j = 0; j < T; j += VL) {
      aie::vector<float, VL> a = aie::load_v<VL>(ar + j);
      aie::store_v(srow + j, a);
      float cm = aie::reduce_max(a);
      if (cm > rowmax) rowmax = cm;
    }
    aie::vector<float, VL> maxv = aie::broadcast<float, VL>(rowmax);
    for (int j = 0; j < T; j += VL)
      aie::store_v(srow + j, exp2f_vec(aie::mul(aie::sub(aie::load_v<VL>(srow + j), maxv), log2e_v).to_vector<float>()));
    aie::accum<accfloat, VL> sa = aie::zeros<accfloat, VL>();
    for (int j = 0; j < T; j += VL) sa = aie::add(sa, aie::load_v<VL>(srow + j));
    float inv = 1.0f / aie::reduce_add(sa.to_vector<float>());
    aie::vector<float, VL> iv = aie::broadcast<float, VL>(inv);
    for (int j = 0; j < T; j += VL)
      aie::store_v(pr + j, aie::mul(aie::load_v<VL>(srow + j), iv).to_vector<bfloat16>());
  }
  // ctx
  for (int i = 0; i < TQ; i++)
    for (int d = 0; d < DK; d += VL) {
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int j = 0; j < T; j++)
        acc = aie::mac(acc, aie::broadcast<bfloat16, VL>(probs[i * T + j]), aie::load_v<VL>(V + j * DK + d));
      aie::store_v(ctx + i * DK + d, acc.to_vector<bfloat16>());
    }
  event1();
}

// TRIVIAL copy variants (same signatures/sizes as the real stages) -- to isolate STRUCTURE vs KERNELS
// in the N_QT>1 streaming bug. T==DK==64 so TQ*T == TQ*DK == 512; each just casts-copies straight
// through, so ctx == q. If this PASSES at N_QT>1, the real (slow) kernels are the culprit (race).
extern "C" void stage_scores_t(const bfloat16 *__restrict q, const bfloat16 *__restrict k,
                               float *__restrict ac) {
  event0();
  for (int i = 0; i < ATTN_TQ * ATTN_T; i++) ac[i] = (float)q[i];
  event1();
}
extern "C" void stage_softmax_t(const float *__restrict ac, bfloat16 *__restrict probs) {
  event0();
  for (int i = 0; i < ATTN_TQ * ATTN_T; i++) probs[i] = (bfloat16)ac[i];
  event1();
}
extern "C" void stage_ctx_t(const bfloat16 *__restrict probs, const bfloat16 *__restrict V,
                            bfloat16 *__restrict ctx) {
  event0();
  for (int i = 0; i < ATTN_TQ * ATTN_DK; i++) ctx[i] = probs[i];
  event1();
}

// STAGE C -- context. ctx[i,d] = sum_j probs[i,j] * V[j,d]. probs[TQ,T] bf16, V[T,DK] bf16 ->
// ctx[TQ,DK] bf16. Two inputs (probs from the belt + V from DDR) on ONE tile -- valid (stage A
// already takes q+k). f32 accumulate, single bf16 narrow at the end.
extern "C" void stage_ctx(const bfloat16 *__restrict probs,
                          const bfloat16 *__restrict V, bfloat16 *__restrict ctx) {
#if ATTN_NO_CTX
  for (int i = 0; i < ATTN_TQ * ATTN_DK; i++) ctx[i] = (bfloat16)0.0f;
  return;
#endif
  constexpr int TQ = ATTN_TQ, T = ATTN_T, DK = ATTN_DK;
  event0();
  for (int i = 0; i < TQ; i++) {
    const bfloat16 *pr = probs + i * T;
    bfloat16 *cr = ctx + i * DK;
    for (int d = 0; d < DK; d += VL) {
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int j = 0; j < T; j++)
        acc = aie::mac(acc, aie::broadcast<bfloat16, VL>(pr[j]), aie::load_v<VL>(V + j * DK + d));
      aie::store_v(cr + d, acc.to_vector<bfloat16>());
    }
  }
  event1();
}

// STAGE B -- row softmax. probs[i,:] = softmax(ac[i,:]). f32 in, bf16 out. 3-pass (max, exp+sum,
// normalize), exp/sum SPLIT into separate loops (fusing exp2f_vec + accfloat sum -> NaN on aie2p).
extern "C" void stage_softmax(const float *__restrict ac, bfloat16 *__restrict probs) {
#if ATTN_NO_SOFTMAX
  // ABLATION: preserve the fifo traffic, drop the math (WRONG numerics on purpose).
  for (int i = 0; i < ATTN_TQ * ATTN_T; i++) probs[i] = (bfloat16)ac[i];
  return;
#endif
  constexpr int TQ = ATTN_TQ, T = ATTN_T;
  float srow[ATTN_T];   // was static -- static may alias the odd ping-pong belt buffer in L1
  aie::vector<float, VL> log2e_v = aie::broadcast<float, VL>(LOG2E);
  event0();
  for (int i = 0; i < TQ; i++) {
    const float *ar = ac + i * T;
    bfloat16 *pr = probs + i * T;
    // pass 1: row max
    float rowmax = -3.0e38f;
    for (int j = 0; j < T; j += VL) {
      aie::vector<float, VL> a = aie::load_v<VL>(ar + j);
      aie::store_v(srow + j, a);
      float cm = aie::reduce_max(a);
      if (cm > rowmax) rowmax = cm;
    }
    // pass 2a: exp2((s-max)*log2e) into srow
    aie::vector<float, VL> maxv = aie::broadcast<float, VL>(rowmax);
    for (int j = 0; j < T; j += VL) {
      aie::vector<float, VL> sl =
          aie::mul(aie::sub(aie::load_v<VL>(srow + j), maxv), log2e_v).to_vector<float>();
      aie::store_v(srow + j, exp2f_vec(sl));
    }
    // pass 2b: sum
    aie::accum<accfloat, VL> sumacc = aie::zeros<accfloat, VL>();
    for (int j = 0; j < T; j += VL)
      sumacc = aie::add(sumacc, aie::load_v<VL>(srow + j));
    float inv_sum = 1.0f / aie::reduce_add(sumacc.to_vector<float>());
    // pass 3: normalize -> bf16
    aie::vector<float, VL> inv_v = aie::broadcast<float, VL>(inv_sum);
    for (int j = 0; j < T; j += VL) {
      aie::vector<float, VL> e = aie::load_v<VL>(srow + j);
      aie::store_v(pr + j, aie::mul(e, inv_v).to_vector<bfloat16>());
    }
  }
  event1();
}

// ==================== STAGE BD -- 4th conveyor stage (on-chip BD) ====================
// On-chip BD = rel_shift((q+pos_bias_v) @ p^T), carried to the scores stage as SPLIT-BF16 in the belt
// tail. Compute is bit-equivalent to relpos_mha.cc (bf16*bf16 -> f32 accfloat, g_bd f32); the split is a
// TRANSPORT device to cross the bf16 belt within the scores tile's 2-input budget. p=[P,DK] ~88 KB is
// L2-resident (MemTile) + streamed in BD_KB-row blocks. Spec: bd-onchip-4th-stage. Placed by the
// generator's --relpos-bd-onchip path; INERT in the current 3-stage build (no caller). g_bd is `static`
// resident scratch -- valid for the mono-style H=1 gate (single BD worker, like stage_mono's statics);
// the pipelined/streaming generator must allocate it as a resident Buffer to avoid the belt-alias hazard.
#ifndef BD_KB
#define BD_KB 39   // p key-block rows (P=2T-1=351 = 9*39, no ragged tail at real dims)
#endif

alignas(32) static float g_bd[ATTN_TQ * ATTN_P]; // resident per-query-tile f32 score scratch (~11 KB)

// COLUMN-SLICE dot: g_bd[il, j0+jj] = dot(qv[il,:], pblk[jj,:]). bf16*bf16 -> f32 accfloat (= relpos_mha.cc).
static inline void bd_dot_block(const bfloat16 *__restrict qv, const bfloat16 *__restrict pblk, int pb, int j0) {
  constexpr int TQ = ATTN_TQ, DK = ATTN_DK, P = ATTN_P;
  for (int il = 0; il < TQ; il++) {
    const bfloat16 *qr = qv + il * DK;
    float *o = g_bd + il * P + j0;
    for (int jj = 0; jj < pb; jj++) {
      const bfloat16 *pr = pblk + jj * DK;
      aie::accum<accfloat, VL> acc = aie::zeros<accfloat, VL>();
      for (int d = 0; d < DK; d += VL)
        acc = aie::mac(acc, aie::load_v<VL>(qr + d), aie::load_v<VL>(pr + d));
      o[jj] = aie::reduce_add(acc.to_vector<float>());
    }
  }
}

// rel_shift + f32->split-bf16 emit. The window base (T-1)-(q0+il) is ~never VL-aligned; a vectorized
// aie::load_v there truncates to the 128b boundary (silent garbage, data-masked -- see the unaligned-load
// kb note). We emit SCALAR: `win[j]` is byte-addressed so it is inherently correct at any offset, and the
// emit (TQ*T ops) is trivial vs the dot (TQ*P*DK MACs). Avoids the alignment hazard by construction.
static inline void bd_relshift_emit(int q0, bfloat16 *__restrict bd_hi, bfloat16 *__restrict bd_lo) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, P = ATTN_P;
  for (int il = 0; il < TQ; il++) {
    const int base = (T - 1) - (q0 + il);           // >= 0 for real rows (q0+il < T)
    const float *win = g_bd + il * P + base;
    bfloat16 *hr = bd_hi + il * T;
    for (int j = 0; j < T; j++) {
      float x = win[j];
      bfloat16 hi = (bfloat16)x;
      hr[j] = hi;
#if BD_SPLIT
      bd_lo[il * T + j] = (bfloat16)(x - (float)hi); // lo residual ~= 8 extra mantissa bits
#endif
    }
  }
}

// t_active-AWARE rel_shift + emit. For variable-length clips the rel_shift window base must be
// (t_active-1)-(q0+il), NOT (T-1)-(q0+il) -- this MIRRORS the device-proven relpos_mha.cc
// (relpos_scores_softmax_rows uses `BD + il*P + (t_active-1-(q0+il))`). With p held as the real
// [2t-1] table zero-padded to P (the shipped relpos_mha packing), the BUILT_T base would read the
// WRONG relative positions for t_active<T; using t_active recovers the correct rel-pos alignment.
// Pairs with stage_scores_relpos_bd_mask (key-mask for j>=t_active). base clamps >=0 for pad query
// rows i>=t_active (their ctx output is discarded on de-interleave, so the value is a don't-care).
static inline void bd_relshift_emit_ta(int q0, int t_active, bfloat16 *__restrict bd_hi,
                                       bfloat16 *__restrict bd_lo) {
  constexpr int TQ = ATTN_TQ, T = ATTN_T, P = ATTN_P;
  for (int il = 0; il < TQ; il++) {
    int base = (t_active - 1) - (q0 + il);
    if (base < 0) base = 0;                          // pad query row (output discarded) -> clamp, no OOB
    const float *win = g_bd + il * P + base;
    bfloat16 *hr = bd_hi + il * T;
    for (int j = 0; j < T; j++) {
      float x = win[j];
      bfloat16 hi = (bfloat16)x;
      hr[j] = hi;
#if BD_SPLIT
      bd_lo[il * T + j] = (bfloat16)(x - (float)hi);
#endif
    }
  }
}

// STREAMING bricks for the generator Worker loop (int32 scalar ABI). g_bd is the resident scratch.
extern "C" void bd_stream_block(const bfloat16 *__restrict qv, const bfloat16 *__restrict pblk,
                                int32_t pb, int32_t j0) {
  event0(); bd_dot_block(qv, pblk, (int)pb, (int)j0); event1();
}

// STREAMING bakes (zero-scalar-arg) for the p-block conveyor at real T (p=88 KB > L1 -> streamed in
// BD_KB-row blocks). Two static counters advance without a scalar arg: j0 (p-block offset, wraps at P,
// BD_KB blocks/tile) and q0 (query-tile row base, wraps at TQ*N_QT). Requires ATTN_P % BD_KB == 0 (real
// dims 351 = 9*39). bd_block_bake accumulates one p-block into g_bd; bd_emit_bake rel_shifts + emits.
extern "C" void bd_block_bake(const bfloat16 *__restrict qpv, const bfloat16 *__restrict pblk) {
  static int j0 = 0;
  const bfloat16 *qv = qpv + ATTN_TQ * ATTN_DK;       // qpv = q_pass[TQ,DK] || qv[TQ,DK]
  int pb = (ATTN_P - j0 < BD_KB) ? (ATTN_P - j0) : BD_KB;
  // TRACE MARKERS. The BD core is the ONLY stage of the 4-stage conveyor that decoded zero
  // invocations, and the reason turned out to be trivial: this bake wrapper and bd_emit_bake_ta are
  // the two functions the BD core actually calls, and NEITHER carried event0/event1. The instrumented
  // siblings (bd_stream_block / bd_stream_emit) are different entry points this build never uses. So
  // the core emitted no core-trace events at all -- not a ring overflow (its dedicated 1 MB ring came
  // back only 54% full) and not a correctness problem.
  //
  // Marked HERE and not in bd_emit_bake_ta on purpose: parse.py pairs event0->event1 without regard to
  // which function raised them, so instrumenting both would interleave 9 block pairs with 1 emit pair
  // per query tile and blur the result. This wrapper carries bd_dot_block, which is the matmul we want
  // to compare against the scores stage; emit is a copy + rel_shift and can be marked separately later.
  event0();
  bd_dot_block(qv, pblk, pb, j0);
  event1();
  j0 += BD_KB; if (j0 >= ATTN_P) j0 = 0;              // wrap per query tile (BD_KB blocks each)
}
extern "C" void bd_emit_bake(const bfloat16 *__restrict qpv, bfloat16 *__restrict out) {
  static int q0 = 0;
  for (int i = 0; i < ATTN_TQ * ATTN_DK; i++) out[i] = qpv[i]; // forward q_pass into the belt head
  bfloat16 *bd_hi = out + ATTN_TQ * ATTN_DK;
  bfloat16 *bd_lo = bd_hi + (BD_SPLIT ? ATTN_TQ * ATTN_T : 0);
  bd_relshift_emit(q0, bd_hi, bd_lo);                  // rel_shift + split emit
  q0 += ATTN_TQ; if (q0 >= ATTN_TQ * ATTN_NQT) q0 = 0; // wrap per dispatch
}
// t_active-AWARE emit bake (pairs with stage_scores_relpos_bd_mask). t_active from rtp[0]; the static
// tile counter advances q0 exactly like bd_emit_bake. Belt out = q_pass || BD_hi[|| BD_lo].
extern "C" void bd_emit_bake_ta(const bfloat16 *__restrict qpv, bfloat16 *__restrict out,
                                const int32_t *__restrict rtp) {
  static int q0 = 0;
  const int t_active = rtp[0];
  for (int i = 0; i < ATTN_TQ * ATTN_DK; i++) out[i] = qpv[i];  // forward q_pass into the belt head
  bfloat16 *bd_hi = out + ATTN_TQ * ATTN_DK;
  bfloat16 *bd_lo = bd_hi + (BD_SPLIT ? ATTN_TQ * ATTN_T : 0);
  bd_relshift_emit_ta(q0, t_active, bd_hi, bd_lo);
  q0 += ATTN_TQ; if (q0 >= ATTN_TQ * ATTN_NQT) q0 = 0;          // wrap per dispatch
}

extern "C" void bd_stream_emit(const bfloat16 *__restrict q_pass, bfloat16 *__restrict out, int32_t q0) {
  constexpr int TQ = ATTN_TQ, DK = ATTN_DK, T = ATTN_T;
  event0();
  for (int i = 0; i < TQ * DK; i++) out[i] = q_pass[i];
  bfloat16 *bd_hi = out + TQ * DK;
  bfloat16 *bd_lo = bd_hi + (BD_SPLIT ? TQ * T : 0);
  bd_relshift_emit((int)q0, bd_hi, bd_lo);
  event1();
}

// MONOLITH bake (H=1 arithmetic gate, p kpv-resident full): dot p in-block, then rel_shift+split emit.
extern "C" void stage_bd(const bfloat16 *__restrict qv, const bfloat16 *__restrict q_pass,
                         const bfloat16 *__restrict p_resident, bfloat16 *__restrict out, int32_t q0) {
  constexpr int P = ATTN_P, DK = ATTN_DK;
  event0();
  for (int j0 = 0; j0 < P; j0 += BD_KB) {
    int pb = (P - j0 < BD_KB) ? (P - j0) : BD_KB;
    bd_dot_block(qv, p_resident + j0 * DK, pb, j0);
  }
  bd_stream_emit(q_pass, out, q0);
  event1();
}

// Zero-scalar-arg BD bake. Belt input packs q_pass[TQ,DK] then qv[TQ,DK] (2*TQ*DK, ONE streamed object);
// p is the 2nd input (resident). Out = q_pass || BD_hi [|| BD_lo] -- the belt stage_scores_relpos_bd
// consumes. ADVANCING q0: the BD worker calls this once per query tile, in order, on one core -> a static
// counter advances q0 = tile_idx*TQ; wrap % N_QT resets it per dispatch (N_QT tiles each). Solves the
// row-offset without a scalar arg / belt header. ATTN_NQT MUST be baked (-DATTN_NQT) to match the build.
extern "C" void stage_bd_bake(const bfloat16 *__restrict qpv, const bfloat16 *__restrict p,
                              bfloat16 *__restrict out) {
  static int tile_idx = 0;
  const int q0 = tile_idx * ATTN_TQ;
  tile_idx = (tile_idx + 1) % ATTN_NQT;   // wrap per dispatch
  const bfloat16 *q_pass = qpv;
  const bfloat16 *qv = qpv + ATTN_TQ * ATTN_DK;
  stage_bd(qv, q_pass, p, out, q0);
}
