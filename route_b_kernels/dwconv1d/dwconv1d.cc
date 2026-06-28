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

extern "C" {

// Parakeet-TDT FastConformer depthwise conv: k=9, 'same' (pad=4), per-channel,
// with BatchNorm-folded bias carried in w[9]. in/out: T samples; w: 16-tile
// (taps [0..8], bias [9]). T = the encoder frame count baked at build time.
void dwconv1d_k9_bf16(bfloat16 *in, bfloat16 *w, bfloat16 *out) {
  dwconv1d_same<400, 9, 4, 32, true>(in, w, out);
}

// GigaAM-v3 Conformer depthwise conv: k=5, 'same' (pad=2), bias-free.
// in/out: 400 samples; w: 16-tile (first 5 = taps).
void dwconv1d_k5_bf16(bfloat16 *in, bfloat16 *w, bfloat16 *out) {
  dwconv1d_same<400, 5, 2, 32, false>(in, w, out);
}
}
