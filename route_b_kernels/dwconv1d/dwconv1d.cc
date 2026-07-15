//===- dwconv1d.cc ----------------------------------------*- C++ -*-===//
//
// Depthwise conv1d, 'same' padding, stride 1, VECTORIZED with aie::sliding_mul
// (the COMPUTE FIR brick -- catalog #6). One channel is processed per kernel
// call (per-channel taps differ, so each channel is its own ObjectFifo tile).
//
//   out[t] = sum_{p=0..K-1} w[p] * in_pad[t + p]   (+ bias, k=9 path)
//
// where in_pad is `in` zero-padded by P on each side ('same' -> P = (K-1)/2).
// This is a cross-correlation (NO kernel flip), matching torch.nn.Conv1d and the
// host reference (scripts/parakeet_ref_encoder.py conv_module: pad=4, k=9).
//
// THE BRICK: aie::sliding_mul_ops<L, K, ...>::mul(coeff, 0, data, 0) computes L
// output lanes at once -- each lane `l` is sum_{p} coeff[p] * data[l + p], i.e.
// a length-K FIR over a sliding window, with the K-sum kept in the accfloat
// accumulator. One issue retires L (=32) time steps; the old scalar loop did one
// (t, tap) MAC at a time. This is the ~lanes x COMPUTE win the brick catalog
// flags (the depthwise conv was the last scalar-loop COMPUTE brick-miss).
//
// Precision: bf16 in -> accfloat (fp32) accumulate -> single bf16 round on store,
// so a numpy fp32-accumulate reference matches to bf16 ULP (rel-L2 << 0.08).
//
// Weight tile (KW=16 bf16): taps in slots [0..K-1]. For the k=9 Parakeet path the
// per-channel bias (BatchNorm folded into the depthwise bias) is carried in slot
// [K] = w[9] -- this keeps the (in, w, out) 3-buffer signature (no 4th DMA input)
// and folds the bias add into the on-chip epilogue. The k=5 GigaAM path is
// bias-free (no slot read past the taps).
//
// TRACKED COPY -- installed into mlir-aie/aie_kernels/aie2p/ by setup_route_b.sh
// (the mlir-aie tree is gitignored / re-cloned).
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

// STEP-4 fused-SiLU epilogue probe (opt-in; default OFF so normal builds are unchanged).
// Build with -DDWCONV_SILU=1 to fuse silu into dwconv1d_shift's tail (guarded block below).
// MUST be defined before dwconv1d_shift, which uses it. Reproduces the reverted step-4 attempt.
#ifndef DWCONV_SILU
#define DWCONV_SILU 0
#endif

// Emit L conv outputs starting at time `off`, storing to out[off .. off+L-1].
//   data window = in_pad[off .. off + L + K - 2]  (load VD >= L+K-1 contiguous)
template <int L, int K, int VD, bool BIAS>
static inline void dwconv1d_emit(const bfloat16 *__restrict buf, int off,
                                 const aie::vector<bfloat16, 16> &cv,
                                 const aie::vector<float, L> &bv,
                                 bfloat16 *__restrict out) {
  using MulOps = aie::sliding_mul_ops<L, K, 1, 1, 1, bfloat16, bfloat16>;
  aie::vector<bfloat16, VD> dv = aie::load_v<VD>(&buf[off]);
  // sliding FIR: acc[l] = sum_{p=0..K-1} cv[p] * dv[l + p]   (K-sum in accfloat)
  aie::accum<accfloat, L> acc = MulOps::mul(cv, 0, dv, 0);
  if constexpr (BIAS) {
    aie::vector<float, L> fv = acc.template to_vector<float>();
    aie::vector<float, L> sv = aie::add(fv, bv);
    aie::accum<accfloat, L> oacc;
    oacc.from_vector(sv);
    aie::store_v(&out[off], oacc.template to_vector<bfloat16>());
  } else {
    aie::store_v(&out[off], acc.template to_vector<bfloat16>());
  }
}

template <int T, int K, int P, int L, bool BIAS>
static inline void dwconv1d_same(const bfloat16 *restrict in,
                                 const bfloat16 *restrict w,
                                 bfloat16 *restrict out) {
  event0();
  constexpr int VD = 64; // data vector lanes (1024-bit = sliding_mul max_data_bits)
  static_assert(L + K - 1 <= VD, "FIR window must fit one data vector");
  static_assert(L % 32 == 0, "L must be a multiple of the bf16 vector width");
  // Padded working buffer, sized so any VD-wide load at the last chunk start
  // (off <= T - L) stays in-bounds; rounded to the 32-lane store width.
  constexpr int PADBUF = ((T + 2 * P + VD + 31) / 32) * 32;

  bfloat16 buf[PADBUF];
  const aie::vector<bfloat16, 32> z = aie::broadcast<bfloat16, 32>(bfloat16(0.0f));
  for (int i = 0; i < PADBUF; i += 32)
    aie::store_v(&buf[i], z);
  // copy the channel's time series into the body [P .. P+T-1]
  for (int t = 0; t < T; t++)
    buf[P + t] = in[t];

  aie::vector<bfloat16, 16> cv = aie::load_v<16>(w); // taps [0..K-1] (+ bias @ [K])
  aie::vector<float, L> bv;
  if constexpr (BIAS)
    bv = aie::broadcast<float, L>(static_cast<float>(w[K]));

  // Bulk: floor(T/L) full chunks, then a final chunk anchored at T-L so every
  // store is full-width and in-bounds (the tail overlaps + overwrites identical
  // values -- cheaper than a masked partial store).
  constexpr int NFULL = T / L;
  AIE_LOOP_UNROLL_FULL
  for (int c = 0; c < NFULL; c++)
    dwconv1d_emit<L, K, VD, BIAS>(buf, c * L, cv, bv, out);
  if constexpr (T % L != 0)
    dwconv1d_emit<L, K, VD, BIAS>(buf, T - L, cv, bv, out);

  event1();
}

// SCALAR FIR (correct on the current toolchain). Same 'same'-padded cross-correlation math as
// dwconv1d_same, but plain per-element MACs -- NO aie::sliding_mul. The vectorized sliding_mul path
// above is MISCOMPILED for bf16 (K=9, XDNA2) under toolchain.lock fb1f7095: it emits ~half-corrupted
// output (fails at L=16 and L=32), while this scalar path + the dataflow are both proven bit-correct
// (identity 1024/1024, random rel-L2 0.003). See the step-3 dwconv investigation. Depthwise conv is
// compute-cheap (C*T*K MACs), so the scalar cost is small; flip back to the sliding_mul brick by
// building with -DDWCONV_SCALAR=0 once the vectorized path is fixed/root-caused.
template <int T, int K, int P, bool BIAS>
static inline void dwconv1d_same_scalar(const bfloat16 *restrict in,
                                        const bfloat16 *restrict w,
                                        bfloat16 *restrict out) {
  event0();
  for (int t = 0; t < T; t++) {
    float acc = BIAS ? static_cast<float>(w[K]) : 0.0f; // BN-folded bias @ w[K] (=w[9] for k=9)
    for (int p = 0; p < K; p++) {
      int idx = t - P + p; // 'same' cross-correlation, zero outside [0,T)
      if (idx >= 0 && idx < T)
        acc += static_cast<float>(w[p]) * static_cast<float>(in[idx]);
    }
    out[t] = static_cast<bfloat16>(acc);
  }
  event1();
}

// VECTORIZED FIR via ALIGNED loads + aie::shuffle_down_fill (in-register sliding window). This is the
// CORRECT + fast path (identity 1024/1024, random rel-L2 0.003). It sidesteps the two toolchain traps
// the naive brick hits (see the step-3 dwconv investigation, both persistent to Peano nightly 071401):
//   (1) unaligned L1 vector loads SNAP to the aligned base -- so load_v(&buf[o+p]) at p!=0 (as the
//       old scalar-window / sliding_mul staging does) reads the wrong data; and
//   (2) aie::sliding_mul_ops<...,bf16,bf16> itself emits inf/nan even with aligned data (a genuine
//       aie_api fp-sliding-mul defect, reported upstream).
// Here every load is aligned (o steps by 16, buf is alignas(64)); tap-p window buf[o+p..o+p+15] is
// built purely in-register: shuffle_down_fill(a0,a1,p) = [a0[p..15] | a1[0..p-1]]. k=9 unrolled (the
// window o+8+15=o+23 stays within the a0||a1 32-lane concat).
template <int T, int K, int P, bool BIAS>
static inline void dwconv1d_shift(const bfloat16 *restrict in, const bfloat16 *restrict w, bfloat16 *restrict out) {
  event0();
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  static_assert(K == 9, "dwconv1d_shift is unrolled for k=9");
  static_assert(T % 16 == 0, "T must be a multiple of the 16-lane output chunk");
  constexpr int PADBUF = ((T + 2 * P + 32 + 31) / 32) * 32;
  alignas(64) bfloat16 buf[PADBUF];
  const ::aie::vector<bfloat16, 16> z = ::aie::broadcast<bfloat16, 16>(bfloat16(0.0f));
  for (int i = 0; i < PADBUF; i += 16) ::aie::store_v(&buf[i], z);
  for (int t = 0; t < T; t++) buf[P + t] = in[t];
  const float bias = BIAS ? static_cast<float>(w[K]) : 0.0f;
  const ::aie::vector<bfloat16, 16> c0 = ::aie::broadcast<bfloat16, 16>(w[0]), c1 = ::aie::broadcast<bfloat16, 16>(w[1]),
    c2 = ::aie::broadcast<bfloat16, 16>(w[2]), c3 = ::aie::broadcast<bfloat16, 16>(w[3]), c4 = ::aie::broadcast<bfloat16, 16>(w[4]),
    c5 = ::aie::broadcast<bfloat16, 16>(w[5]), c6 = ::aie::broadcast<bfloat16, 16>(w[6]), c7 = ::aie::broadcast<bfloat16, 16>(w[7]),
    c8 = ::aie::broadcast<bfloat16, 16>(w[8]);
  for (int o = 0; o < T; o += 16) {
    ::aie::vector<bfloat16, 16> a0 = ::aie::load_v<16>(&buf[o]), a1 = ::aie::load_v<16>(&buf[o + 16]);
    ::aie::accum<accfloat, 16> a;
    a.from_vector(::aie::broadcast<float, 16>(bias));
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 0), c0); a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 1), c1);
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 2), c2); a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 3), c3);
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 4), c4); a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 5), c5);
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 6), c6); a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 7), c7);
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 8), c8);
#if DWCONV_SILU
    // FUSED-SiLU REPRODUCER (opt-in, default OFF; -DDWCONV_SILU=1). Fuses SiLU as the
    // dwconv epilogue, hiprec recipe (matches glu.cc / mm_silu_epilogue_f32o_hiprec).
    // This MISCOMPILES on toolchain.lock fb1f7095: adding ANY multi-op epilogue to this
    // objectfifo-driven per-channel loop corrupts alternate (even) channel iterations
    // (ping-pong buffer 0), odd bit-exact; NOT the SFU, accum-feed, or objectfifo depth
    // (all refuted on device). => SiLU MUST be a SEPARATE brick, not fused here.
    // Full isolation: docs/log/2026-07/dwconv-fused-epilogue-alt-channel-miscompile.md
    //   silu(x) = x * sigmoid(x),  sigmoid(x) = 0.5*(1 + tanh(x/2))
    ::aie::vector<float, 16> xf = a.template to_vector<float>();
    ::aie::vector<float, 16> hx = ::aie::mul(xf, ::aie::broadcast<float, 16>(0.5f));
    ::aie::vector<bfloat16, 16> th = ::aie::tanh<bfloat16>(hx);
    ::aie::vector<bfloat16, 16> tp1 = ::aie::add(th, ::aie::broadcast<bfloat16, 16>(1.0f));
    ::aie::vector<bfloat16, 16> sig = ::aie::mul(tp1, ::aie::broadcast<bfloat16, 16>(0.5f));
    ::aie::accum<accfloat, 16> sacc;
    sacc.from_vector(sig);
    ::aie::vector<float, 16> sigf = sacc.template to_vector<float>();
    ::aie::vector<float, 16> ov = ::aie::mul(xf, sigf);
    ::aie::accum<accfloat, 16> oacc;
    oacc.from_vector(ov);
    ::aie::store_v(&out[o], oacc.template to_vector<bfloat16>());
#else
    ::aie::store_v(&out[o], a.template to_vector<bfloat16>());
#endif
  }
  event1();
}

// F32-OUTPUT variant of dwconv1d_shift (byte-for-byte the SAME validated aligned+shuffle FIR, storing
// the accumulator as f32 instead of a bf16 round). This is the producer stage of the FUSED
// dwconv->silu xclbin (dwconv_silu_iron.py): it hands the FIR result to an on-chip f32 ObjectFifo that
// the (unchanged) silu_row brick consumes device-to-device -- so the on-NPU SiLU costs NO extra
// hw-context switch and no host round-trip (the measured ~1 ms/block the SEPARATE silu xclbin added).
// NOT a fused epilogue: this stays a SIMPLE single-store FIR loop, so it is immune to the alt-channel
// per-tile-loop miscompile (that needs a HEAVY multi-op epilogue; a plain f32 store is one op). The
// silu itself remains its own core's simple loop. Keeping f32 here (no bf16 round) is >= as precise as
// the host path (bf16 dwconv out -> host f32 -> silu); WER-gated equal to the separate-brick silu.
template <int T, int K, int P, bool BIAS>
static inline void dwconv1d_shift_f32o(const bfloat16 *restrict in, const bfloat16 *restrict w, float *restrict out) {
  event0();
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  static_assert(K == 9, "dwconv1d_shift_f32o is unrolled for k=9");
  static_assert(T % 16 == 0, "T must be a multiple of the 16-lane output chunk");
  constexpr int PADBUF = ((T + 2 * P + 32 + 31) / 32) * 32;
  alignas(64) bfloat16 buf[PADBUF];
  const ::aie::vector<bfloat16, 16> z = ::aie::broadcast<bfloat16, 16>(bfloat16(0.0f));
  for (int i = 0; i < PADBUF; i += 16) ::aie::store_v(&buf[i], z);
  for (int t = 0; t < T; t++) buf[P + t] = in[t];
  const float bias = BIAS ? static_cast<float>(w[K]) : 0.0f;
  const ::aie::vector<bfloat16, 16> c0 = ::aie::broadcast<bfloat16, 16>(w[0]), c1 = ::aie::broadcast<bfloat16, 16>(w[1]),
    c2 = ::aie::broadcast<bfloat16, 16>(w[2]), c3 = ::aie::broadcast<bfloat16, 16>(w[3]), c4 = ::aie::broadcast<bfloat16, 16>(w[4]),
    c5 = ::aie::broadcast<bfloat16, 16>(w[5]), c6 = ::aie::broadcast<bfloat16, 16>(w[6]), c7 = ::aie::broadcast<bfloat16, 16>(w[7]),
    c8 = ::aie::broadcast<bfloat16, 16>(w[8]);
  for (int o = 0; o < T; o += 16) {
    ::aie::vector<bfloat16, 16> a0 = ::aie::load_v<16>(&buf[o]), a1 = ::aie::load_v<16>(&buf[o + 16]);
    ::aie::accum<accfloat, 16> a;
    a.from_vector(::aie::broadcast<float, 16>(bias));
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 0), c0); a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 1), c1);
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 2), c2); a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 3), c3);
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 4), c4); a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 5), c5);
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 6), c6); a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 7), c7);
    a = ::aie::mac(a, ::aie::shuffle_down_fill(a0, a1, 8), c8);
    ::aie::store_v(&out[o], a.template to_vector<float>());
  }
  event1();
}

// Kernel selection: default = the vectorized aligned+shuffle FIR (dwconv1d_shift). DWCONV_SCALAR=1
// forces the scalar fallback (dwconv1d_same_scalar). The aie::sliding_mul brick (dwconv1d_same) is
// RETAINED only as the broken-path reference for the upstream repro; never selected.
#ifndef DWCONV_SCALAR
#define DWCONV_SCALAR 0
#endif

extern "C" {

// Parakeet-TDT FastConformer depthwise conv: k=9, 'same' (pad=4), per-channel,
// with BatchNorm-folded bias carried in w[9]. in/out: T samples; w: 16-tile
// (taps [0..8], bias [9]). T = the encoder frame count baked at build time.
void dwconv1d_k9_bf16(bfloat16 *in, bfloat16 *w, bfloat16 *out) {
#if DWCONV_SCALAR
  dwconv1d_same_scalar<400, 9, 4, true>(in, w, out);
#else
  dwconv1d_shift<400, 9, 4, true>(in, w, out);
#endif
}

// GigaAM-v3 Conformer depthwise conv: k=5, 'same' (pad=2), bias-free. Scalar (correct); a vectorized
// k5 shift path is YAGNI until a GigaAM model is on the bench (dwconv1d_shift is unrolled for k9).
void dwconv1d_k5_bf16(bfloat16 *in, bfloat16 *w, bfloat16 *out) {
  dwconv1d_same_scalar<400, 5, 2, false>(in, w, out);
}

// Parakeet k=9 dwconv producing f32 (the fused dwconv->silu xclbin's producer stage). Same FIR as
// dwconv1d_k9_bf16, f32 output for the on-chip f32 ObjectFifo feeding the silu core.
void dwconv1d_k9_bf16_f32o(bfloat16 *in, bfloat16 *w, float *out) {
  dwconv1d_shift_f32o<400, 9, 4, true>(in, w, out);
}
}
