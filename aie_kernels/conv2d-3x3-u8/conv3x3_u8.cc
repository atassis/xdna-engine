// 3x3 same-padded convolution for one output row: int8 weights x 8- or 16-bit
// activations -> int32 accumulate, + int32 bias, rounding shift, saturate, then
// an optional pointwise epilogue. Entry points (in -> out):
//   conv3x3_u8        uint8 -> uint8; negative sums clamp to 0, so ReLU is implied
//   conv3x3_i8        int8 -> int8
//   conv3x3_i16i8     int16 -> int8
//   conv3x3_i8_lut    int8 -> int8, then q -> table[q + 128] (e.g. SiLU)
//   conv3x3_i16i8_lut int16 -> int8, then the same table
//   conv3x3_i8_lut16  int8 -> int16: q -> hi[q + 128] * 256 + lo[q + 128]
//   conv3x3_i8_gate   int8 -> int8, SPAN's attention block tail: with c = conv
//                     output, x = block input at the same pixel/channel (xrow,
//                     laid out like out) and att = table[c + 128]:
//                       sum = sat8((c*ga + x*gb) >> gs1)
//                       out = sat8((sat16(sum*att) * gc) >> gs2)
//                     xrow is consumed: the sum is parked in it.
// >> rounds half up, the conv's own requant rounding.
//
// Rows are channel-blocked and padded by one 8-pixel block of zeros on each
// side, so the kernel never branches on the row edge:
//   line0/1/2  input rows y-1, y, y+1   [CIN/8][width+16][8]   margins zero
//   out        output row y             [COUT/8][width+16][8]  margins written zero
//   params     weights [COUT/8][3 ky][CIN/8][3 kx][8 ic][8 oc] int8,
//              then bias [COUT/8][8 px][8 oc] int32 (repeated per pixel so it
//              loads straight into an accumulator tile)
// Tables are per-layer model constants, compiled in: CONV3X3_LUT_INC (and
// CONV3X3_LUT2_INC for the lo half of lut16) name files of 512 bf16 words
// (golden.lut_inc).
//
// check: 0 = top row (line0 absent), 1 = middle, 2 = bottom (line2 absent).
// Output pixels outside [valid_lo, valid_hi) are written as zero, so a strip's
// frame-edge columns stay zero-padded for the next layer.
// width % 16 == 0, CIN % 8 == 0, COUT % 16 == 0.

#include <aie_api/aie.hpp>
#include <stdint.h>

#define C3_PRAGMA(x) _Pragma(#x)
#define C3_LOOP_RANGE(lo, hi)                                                  \
  C3_PRAGMA(clang loop min_iteration_count(lo))                                \
  C3_PRAGMA(clang loop max_iteration_count(hi))

namespace {

enum Epi { NONE, LUT, GATE, LUT16 };

// v covers `px` pixels of 8 channels starting at pixel x0
template <typename T, unsigned N>
inline aie::vector<T, N> mask_block(aie::vector<T, N> v, int x0, int lo, int hi) {
  constexpr int px = N / 8;
  if (x0 >= lo && x0 + px <= hi)
    return v;
  uint64_t bits = 0;
  for (int i = 0; i < px; i++)
    if (x0 + i >= lo && x0 + i < hi)
      bits |= uint64_t(0xff) << (8 * i);
  if constexpr (N == 64)
    return aie::select(aie::zeros<T, N>(), v, aie::mask<N>::from_uint64(bits));
  else
    return aie::select(aie::zeros<T, N>(), v,
                       aie::mask<N>::from_uint32(uint32_t(bits)));
}

// The int8-key bf16 gather is the configuration verified on device (rope-lut);
// integers -128..255 are exact in bf16. Table placement is golden.pack_lut.
using Look = aie::parallel_lookup<int8, aie::lut<4, bfloat16>>;

#ifdef CONV3X3_LUT_INC
alignas(aie::vector_decl_align) static const uint16_t kLutAb[512] = {
#include CONV3X3_LUT_INC
};
alignas(aie::vector_decl_align) static const uint16_t kLutCd[512] = {
#include CONV3X3_LUT_INC
};
#endif
#ifdef CONV3X3_LUT2_INC
alignas(aie::vector_decl_align) static const uint16_t kLut2Ab[512] = {
#include CONV3X3_LUT2_INC
};
alignas(aie::vector_decl_align) static const uint16_t kLut2Cd[512] = {
#include CONV3X3_LUT2_INC
};
#endif

#ifdef CONV3X3_LUT_INC
// Applies the table in place to n int8 values at dst (n % 16 == 0). Keys are
// loaded as 16-lane vectors: extracting 16-lane groups from a 64-lane register
// vector fed fetch() wrong keys on device (probe_lut_gather_isolated.py). The
// lookup object is built per call: one built once at the top of the gate kernel
// and captured by reference gathered every key as 0
// (probe_conv3x3_gate_stages.py).
inline void apply_lut_inplace(int8_t *dst, int n) {
  const aie::lut<4, bfloat16> t(256, (const bfloat16 *)kLutAb,
                                (const bfloat16 *)kLutCd);
  Look look(t, 0, 128);
  for (int g = 0; g < n; g += 16) {
    aie::vector<int8, 16> k = aie::load_v<16>(dst + g);
    aie::store_v(dst + g, aie::to_fixed<int8>(look.fetch(k), 0));
  }
}
#endif

#if defined(CONV3X3_LUT_INC) && defined(CONV3X3_LUT2_INC)
// 64 int8 keys at keys -> 64 int16 values hi*256 + lo at dst. keys may be the
// second half of dst's 128 bytes: group g is read before bytes [32g, 32g+32)
// are written, and those only ever overlap keys already consumed.
inline void apply_lut16(const int8_t *keys, int16_t *dst) {
  const aie::lut<4, bfloat16> th(256, (const bfloat16 *)kLutAb,
                                 (const bfloat16 *)kLutCd);
  const aie::lut<4, bfloat16> tl(256, (const bfloat16 *)kLut2Ab,
                                 (const bfloat16 *)kLut2Cd);
  Look hi(th, 0, 128), lo(tl, 0, 128);
  for (int g = 0; g < 64; g += 16) {
    aie::vector<int8, 16> k = aie::load_v<16>(keys + g);
    aie::accum<acc32, 16> a;
    a.from_vector(aie::to_fixed<int16>(lo.fetch(k), 0));
    a = aie::mac(a, aie::to_fixed<int16>(hi.fetch(k), 0), (int16)256);
    aie::store_v(dst + g, a.template to_vector<int16>(0));
  }
}
#endif

// TA/PA: activation element type (aie_api / C); TO/PO: output. A tile is 64
// bytes of input: 8 pixels of int8 or 4 of int16, 8 channels each.
template <typename TA, typename PA, typename TO, typename PO, int CIN, int COUT,
          Epi EPI = NONE>
__attribute__((noinline)) void
conv3x3_core(const PA *__restrict line0, const PA *__restrict line1,
             const PA *__restrict line2, const int8_t *__restrict params,
             PO *__restrict out, int width, int check, int shift, int valid_lo,
             int valid_hi, const PO *__restrict xrow = nullptr, int ga = 0,
             int gb = 0, int gs1 = 0, int gc = 0, int gs2 = 0) {
  constexpr int PX = 8 / sizeof(PA); // pixels per tile
  constexpr int LANES = PX * 8;      // elements per input tile (64 bytes)
  using MMUL = aie::mmul<PX, 8, 8, TA, int8>;
  using VA = aie::vector<TA, LANES>;
  static_assert(CIN % 8 == 0 && COUT % 16 == 0, "channel blocking");
  static_assert(EPI != LUT16 || (sizeof(PA) == 1 && sizeof(PO) == 2),
                "lut16 maps int8 keys to int16");
  static_assert(EPI == LUT16 || EPI == NONE || sizeof(PO) == 1,
                "the table and gate epilogues write int8");
  constexpr int ICB = CIN / 8;
  constexpr int OCB = COUT / 8;
  constexpr int WBLK = 3 * ICB * 3 * 64; // weight bytes per output-channel block

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

#ifndef CONV3X3_LUT_INC
  static_assert(EPI == NONE, "table variants need -DCONV3X3_LUT_INC=<file>");
#endif
#ifndef CONV3X3_LUT2_INC
  static_assert(EPI != LUT16, "lut16 needs -DCONV3X3_LUT2_INC=<file>");
#endif
  const int32_t *bias = reinterpret_cast<const int32_t *>(params + OCB * WBLK);
  const PA *lines[3] = {line0, line1, line2};
  const int ky0 = (check == 0) ? 1 : 0;
  const int ky1 = (check == 2) ? 2 : 3;
  const int plane = (width + 16) * 8; // elements per 8-channel block of a padded row

  // one tile of PX pixels x 8 channels: requant, epilogue, column mask, store
  auto put = [&](PO *dst, MMUL &acc, int xs) {
    if constexpr (EPI == NONE) {
      aie::store_v(dst, mask_block(acc.template to_vector<TO>(shift), xs,
                                   valid_lo, valid_hi));
    } else if constexpr (EPI == LUT) {
#ifdef CONV3X3_LUT_INC
      aie::store_v(dst, acc.template to_vector<int8>(shift));
      apply_lut_inplace(reinterpret_cast<int8_t *>(dst), LANES);
      if (!(xs >= valid_lo && xs + PX <= valid_hi))
        aie::store_v(dst, mask_block(aie::load_v<LANES>(dst), xs, valid_lo,
                                     valid_hi));
#endif
    } else if constexpr (EPI == LUT16) {
#if defined(CONV3X3_LUT_INC) && defined(CONV3X3_LUT2_INC)
      int8_t *keys = reinterpret_cast<int8_t *>(dst) + LANES;
      aie::store_v(keys, acc.template to_vector<int8>(shift));
      apply_lut16(keys, reinterpret_cast<int16_t *>(dst));
      if (!(xs >= valid_lo && xs + PX <= valid_hi))
        aie::store_v(dst, mask_block(aie::load_v<LANES>(dst), xs, valid_lo,
                                     valid_hi));
#endif
    } else { // GATE
#ifdef CONV3X3_LUT_INC
      // Three passes through memory. A single pass (one load feeding both
      // fetch() and the sum) has not been re-tested since the lookup-object fix.
      aie::store_v(dst, acc.template to_vector<int8>(shift));
      PO *xp = const_cast<PO *>(xrow) + (dst - out);
      for (int g = 0; g < LANES; g += 16) {
        aie::accum<acc32, 16> a = aie::mul(aie::load_v<16>(dst + g), (int8)ga);
        a = aie::mac(a, aie::load_v<16>(xp + g), (int8)gb);
        aie::store_v(xp + g, a.template to_vector<int8>(gs1));
      }
      apply_lut_inplace(reinterpret_cast<int8_t *>(dst), LANES);
      for (int g = 0; g < LANES; g += 16) {
        aie::vector<int16, 16> p16 =
            aie::mul(aie::load_v<16>(xp + g), aie::load_v<16>(dst + g))
                .template to_vector<int16>(0);
        aie::store_v(dst + g,
                     aie::mul(p16, (int16)gc).template to_vector<int8>(gs2));
      }
      if (!(xs >= valid_lo && xs + PX <= valid_hi))
        aie::store_v(dst, mask_block(aie::load_v<LANES>(dst), xs, valid_lo,
                                     valid_hi));
#endif
    }
  };

  const aie::vector<TO, 64> zero = aie::zeros<TO, 64>();
  for (int ob = 0; ob < OCB; ob += 2) {
    PO *o0 = out + ob * plane;
    PO *o1 = o0 + plane;
    aie::store_v(o0, zero);
    aie::store_v(o1, zero);
    aie::store_v(o0 + plane - 64, zero);
    aie::store_v(o1 + plane - 64, zero);
    typename MMUL::accum_type bias0, bias1;
    bias0.from_vector(aie::load_v<LANES>(bias + ob * 64));
    bias1.from_vector(aie::load_v<LANES>(bias + (ob + 1) * 64));

    for (int x0 = 0; x0 < width; x0 += 2 * PX) {
      MMUL a00(bias0), a01(bias0), a10(bias1), a11(bias1);
      const int8_t *__restrict wk0 = params + ob * WBLK + ky0 * ICB * 192;
      const int8_t *__restrict wk1 = wk0 + WBLK;
      for (int ky = ky0; ky < ky1; ky++) {
        const PA *__restrict p = lines[ky] + (x0 + 8) * 8;
        C3_LOOP_RANGE(ICB, ICB)
        for (int icb = 0; icb < ICB; icb++) {
          VA prev = aie::load_v<LANES>(p - LANES);
          VA c0 = aie::load_v<LANES>(p);
          VA c1 = aie::load_v<LANES>(p + LANES);
          VA next = aie::load_v<LANES>(p + 2 * LANES);
          p += plane;
          // pixel blocks x0 and x0+PX, each shifted left/right by one pixel
          VA l0 = aie::shuffle_up_fill(c0, prev, 8);
          VA r0 = aie::shuffle_down_fill(c0, c1, 8);
          VA l1 = aie::shuffle_up_fill(c1, c0, 8);
          VA r1 = aie::shuffle_down_fill(c1, next, 8);
          aie::vector<int8, 64> b0 = aie::load_v<64>(wk0);
          aie::vector<int8, 64> b1 = aie::load_v<64>(wk1);
          a00.mac(l0, b0);
          a01.mac(l1, b0);
          a10.mac(l0, b1);
          a11.mac(l1, b1);
          b0 = aie::load_v<64>(wk0 + 64);
          b1 = aie::load_v<64>(wk1 + 64);
          a00.mac(c0, b0);
          a01.mac(c1, b0);
          a10.mac(c0, b1);
          a11.mac(c1, b1);
          b0 = aie::load_v<64>(wk0 + 128);
          b1 = aie::load_v<64>(wk1 + 128);
          a00.mac(r0, b0);
          a01.mac(r1, b0);
          a10.mac(r0, b1);
          a11.mac(r1, b1);
          wk0 += 192;
          wk1 += 192;
        }
      }
      const int c = (x0 + 8) * 8;
      put(o0 + c, a00, x0);
      put(o0 + c + LANES, a01, x0 + PX);
      put(o1 + c, a10, x0);
      put(o1 + c + LANES, a11, x0 + PX);
    }
  }
}

} // namespace

#ifndef CONV3X3_CIN
#define CONV3X3_CIN 64
#endif
#ifndef CONV3X3_COUT
#define CONV3X3_COUT 16
#endif

#define C3_ARGS                                                                \
  line0, line1, line2, params, out, width, check, shift, valid_lo, valid_hi
#define C3_SIG(PA, PO)                                                         \
  const PA *line0, const PA *line1, const PA *line2, const int8_t *params,     \
      PO *out, int32_t width, int32_t check, int32_t shift, int32_t valid_lo,  \
      int32_t valid_hi

extern "C" {

void conv3x3_u8(C3_SIG(uint8_t, uint8_t)) {
  conv3x3_core<uint8, uint8_t, uint8, uint8_t, CONV3X3_CIN, CONV3X3_COUT>(C3_ARGS);
}

void conv3x3_i8(C3_SIG(int8_t, int8_t)) {
  conv3x3_core<int8, int8_t, int8, int8_t, CONV3X3_CIN, CONV3X3_COUT>(C3_ARGS);
}

void conv3x3_i16i8(C3_SIG(int16_t, int8_t)) {
  conv3x3_core<int16, int16_t, int8, int8_t, CONV3X3_CIN, CONV3X3_COUT>(C3_ARGS);
}

#ifdef CONV3X3_LUT_INC // the table variants exist only with their table
void conv3x3_i8_lut(C3_SIG(int8_t, int8_t)) {
  conv3x3_core<int8, int8_t, int8, int8_t, CONV3X3_CIN, CONV3X3_COUT, LUT>(C3_ARGS);
}

void conv3x3_i16i8_lut(C3_SIG(int16_t, int8_t)) {
  conv3x3_core<int16, int16_t, int8, int8_t, CONV3X3_CIN, CONV3X3_COUT, LUT>(
      C3_ARGS);
}

void conv3x3_i8_gate(const int8_t *line0, const int8_t *line1,
                     const int8_t *line2, const int8_t *xrow, const int8_t *params,
                     int8_t *out, int32_t width, int32_t check, int32_t shift,
                     int32_t valid_lo, int32_t valid_hi, int32_t ga, int32_t gb,
                     int32_t gs1, int32_t gc, int32_t gs2) {
  conv3x3_core<int8, int8_t, int8, int8_t, CONV3X3_CIN, CONV3X3_COUT, GATE>(
      C3_ARGS, xrow, ga, gb, gs1, gc, gs2);
}
#endif

#if defined(CONV3X3_LUT_INC) && defined(CONV3X3_LUT2_INC)
void conv3x3_i8_lut16(C3_SIG(int8_t, int16_t)) {
  conv3x3_core<int8, int8_t, int16, int16_t, CONV3X3_CIN, CONV3X3_COUT, LUT16>(
      C3_ARGS);
}
#endif

} // extern "C"
