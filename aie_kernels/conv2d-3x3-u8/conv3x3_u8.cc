// 3x3 same-padded convolution for one output row: int8 weights x 8-bit
// activations -> int32 accumulate, + int32 bias, rounding shift, saturate to
// the 8-bit output type. conv3x3_u8: uint8 in/out, so negative sums clamp to 0
// and ReLU is implied. conv3x3_i8: int8 in/out, no activation.
// conv3x3_i8_lut: as conv3x3_i8, then each int8 output q becomes table[q + 128]
// (any pointwise activation, e.g. SiLU, is exact on an int8 input).
//
// Rows are channel-blocked and padded by one 8-pixel block of zeros on each
// side, so the kernel never branches on the row edge:
//   line0/1/2  input rows y-1, y, y+1   [CIN/8][width+16][8]   8-bit, margins zero
//   out        output row y             [COUT/8][width+16][8]  8-bit, margins written zero
//   params     weights [COUT/8][3 ky][CIN/8][3 kx][8 ic][8 oc] int8,
//              then bias [COUT/8][8 px][8 oc] int32 (repeated per pixel so it
//              loads straight into an 8x8 accumulator tile),
//
// LUT tables are per-layer model constants, compiled in: define CONV3X3_LUT_INC
// to a file holding the 512 bf16 words (golden.lut_inc).
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
#define C3_UNROLL_FULL _Pragma("clang loop unroll(full)")

namespace {

template <typename T> using V64 = aie::vector<T, 64>;

template <typename T>
inline V64<T> mask_block(V64<T> v, int x0, int lo, int hi) {
  if (x0 >= lo && x0 + 8 <= hi)
    return v;
  uint64_t bits = 0;
  for (int i = 0; i < 8; i++)
    if (x0 + i >= lo && x0 + i < hi)
      bits |= uint64_t(0xff) << (8 * i);
  return aie::select(aie::zeros<T, 64>(), v, aie::mask<64>::from_uint64(bits));
}

// T is uint8 or int8 (aie_api element types); the pointers carry the matching
// C type P (uint8_t / int8_t).
// Table values are bf16 (integers -128..127 are exact in bf16): the int8-key
// bf16 gather is the configuration verified on device (rope-lut); its table
// placement is golden.pack_lut. Fetched lanes come back in key order.
using Look = aie::parallel_lookup<int8, aie::lut<4, bfloat16>>;

#ifdef CONV3X3_LUT_INC
alignas(aie::vector_decl_align) static const uint16_t kLutAb[512] = {
#include CONV3X3_LUT_INC
};
alignas(aie::vector_decl_align) static const uint16_t kLutCd[512] = {
#include CONV3X3_LUT_INC
};
#endif

// Applies the table in place to the 64 bytes at dst. Keys are reloaded as
// 16-lane vectors: extracting 16-lane groups from the 64-lane register vector
// fed fetch() wrong keys on device (probe_lut_gather_isolated.py), while
// 16-lane loads gather exactly.
inline void apply_lut_inplace(int8_t *dst, Look &look) {
  for (int g = 0; g < 64; g += 16) {
    aie::vector<int8, 16> k = aie::load_v<16>(dst + g);
    aie::store_v(dst + g, aie::to_fixed<int8>(look.fetch(k), 0));
  }
}

template <typename T, typename P, int CIN, int COUT, bool LUT = false>
__attribute__((noinline)) void
conv3x3_core(const P *__restrict line0, const P *__restrict line1,
             const P *__restrict line2, const int8_t *__restrict params,
             P *__restrict out, int width, int check, int shift, int valid_lo,
             int valid_hi) {
  using MMUL = aie::mmul<8, 8, 8, T, int8>;
  static_assert(CIN % 8 == 0 && COUT % 16 == 0, "channel blocking");
  constexpr int ICB = CIN / 8;
  constexpr int OCB = COUT / 8;
  constexpr int WBLK = 3 * ICB * 3 * 64; // weight bytes per output-channel block

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  const int32_t *bias = reinterpret_cast<const int32_t *>(params + OCB * WBLK);
#ifdef CONV3X3_LUT_INC
  const aie::lut<4, bfloat16> lut_t(256, (const bfloat16 *)kLutAb,
                                    (const bfloat16 *)kLutCd);
  Look look(lut_t, 0, 128);
#endif
  const P *lines[3] = {line0, line1, line2};
  const int ky0 = (check == 0) ? 1 : 0;
  const int ky1 = (check == 2) ? 2 : 3;
  const int plane = (width + 16) * 8; // bytes per 8-channel block of a padded row
  const V64<T> zero = aie::zeros<T, 64>();

  // store one 8-pixel x 8-channel tile: optional table, then the column mask
  auto put = [&](P *dst, V64<T> v, int xs) {
    if constexpr (LUT) {
#ifdef CONV3X3_LUT_INC
      aie::store_v(dst, v);
      apply_lut_inplace(reinterpret_cast<int8_t *>(dst), look);
      if (!(xs >= valid_lo && xs + 8 <= valid_hi))
        aie::store_v(dst, mask_block(aie::load_v<64>(dst), xs, valid_lo, valid_hi));
#endif
    } else {
      aie::store_v(dst, mask_block(v, xs, valid_lo, valid_hi));
    }
  };

  for (int ob = 0; ob < OCB; ob += 2) {
    P *o0 = out + ob * plane;
    P *o1 = o0 + plane;
    aie::store_v(o0, zero);
    aie::store_v(o1, zero);
    aie::store_v(o0 + plane - 64, zero);
    aie::store_v(o1 + plane - 64, zero);
    aie::accum<acc32, 64> bias0, bias1;
    bias0.from_vector(aie::load_v<64>(bias + ob * 64));
    bias1.from_vector(aie::load_v<64>(bias + (ob + 1) * 64));

    for (int x0 = 0; x0 < width; x0 += 16) {
      MMUL a00(bias0), a01(bias0), a10(bias1), a11(bias1);
      const int8_t *__restrict wk0 = params + ob * WBLK + ky0 * ICB * 192;
      const int8_t *__restrict wk1 = wk0 + WBLK;
      for (int ky = ky0; ky < ky1; ky++) {
        const P *__restrict p = lines[ky] + (x0 + 8) * 8;
        C3_LOOP_RANGE(ICB, ICB)
        for (int icb = 0; icb < ICB; icb++) {
          V64<T> prev = aie::load_v<64>(p - 64);
          V64<T> c0 = aie::load_v<64>(p);
          V64<T> c1 = aie::load_v<64>(p + 64);
          V64<T> next = aie::load_v<64>(p + 128);
          p += plane;
          // pixel blocks x0 and x0+8, each shifted left/right by one pixel
          V64<T> l0 = aie::shuffle_up_fill(c0, prev, 8);
          V64<T> r0 = aie::shuffle_down_fill(c0, c1, 8);
          V64<T> l1 = aie::shuffle_up_fill(c1, c0, 8);
          V64<T> r1 = aie::shuffle_down_fill(c1, next, 8);
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
      put(o0 + c, a00.template to_vector<T>(shift), x0);
      put(o0 + c + 64, a01.template to_vector<T>(shift), x0 + 8);
      put(o1 + c, a10.template to_vector<T>(shift), x0);
      put(o1 + c + 64, a11.template to_vector<T>(shift), x0 + 8);
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

extern "C" {

void conv3x3_u8(const uint8_t *line0, const uint8_t *line1, const uint8_t *line2,
                const int8_t *params, uint8_t *out, int32_t width, int32_t check,
                int32_t shift, int32_t valid_lo, int32_t valid_hi) {
  conv3x3_core<uint8, uint8_t, CONV3X3_CIN, CONV3X3_COUT>(
      line0, line1, line2, params, out, width, check, shift, valid_lo, valid_hi);
}

void conv3x3_i8_lut(const int8_t *line0, const int8_t *line1,
                    const int8_t *line2, const int8_t *params, int8_t *out,
                    int32_t width, int32_t check, int32_t shift, int32_t valid_lo,
                    int32_t valid_hi) {
  conv3x3_core<int8, int8_t, CONV3X3_CIN, CONV3X3_COUT, true>(
      line0, line1, line2, params, out, width, check, shift, valid_lo, valid_hi);
}

void conv3x3_i8(const int8_t *line0, const int8_t *line1, const int8_t *line2,
                const int8_t *params, int8_t *out, int32_t width, int32_t check,
                int32_t shift, int32_t valid_lo, int32_t valid_hi) {
  conv3x3_core<int8, int8_t, CONV3X3_CIN, CONV3X3_COUT>(
      line0, line1, line2, params, out, width, check, shift, valid_lo, valid_hi);
}

} // extern "C"
