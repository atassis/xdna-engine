// 3x3 same-padded convolution for one output row: uint8 activations x int8
// weights -> int32 accumulate, + int32 bias, rounding shift, saturate to uint8
// (negative sums clamp to 0, so ReLU is implied).
//
// Layouts (all channel-blocked, 8 channels innermost):
//   line0/1/2  input rows y-1, y, y+1   [CIN/8][width][8]   uint8
//   out        output row y             [COUT/8][width][8]  uint8
//   params     weights [COUT/8][3 ky][CIN/8][3 kx][8 ic][8 oc] int8,
//              then bias [COUT/8][8 px][8 oc] int32 (per-channel bias repeated per pixel,
//              so it loads straight into an 8x8 accumulator tile)
//
// check: 0 = top row (line0 absent), 1 = middle, 2 = bottom (line2 absent).
// Output columns outside [valid_lo, valid_hi) are written as zero, so a strip's
// frame-edge columns stay zero-padded for the next layer.
// width % 8 == 0, CIN % 8 == 0, COUT % 16 == 0.

#include <aie_api/aie.hpp>
#include <stdint.h>

template <int CIN, int COUT>
static inline void conv3x3_u8_row(const uint8_t *line0, const uint8_t *line1,
                                  const uint8_t *line2, const int8_t *params,
                                  uint8_t *out, int width, int check, int shift,
                                  int valid_lo, int valid_hi) {
  static_assert(CIN % 8 == 0 && COUT % 16 == 0, "channel blocking");
  using MMUL = aie::mmul<8, 8, 8, uint8, int8>;
  constexpr int ICB = CIN / 8;
  constexpr int OCB = COUT / 8;
  constexpr int WBLK = 3 * ICB * 3 * 64; // weight bytes per output-channel block

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  const int8_t *wts = params;
  const int32_t *bias = reinterpret_cast<const int32_t *>(params + OCB * WBLK);
  const uint8_t *lines[3] = {line0, line1, line2};
  const int ky0 = (check == 0) ? 1 : 0;
  const int ky1 = (check == 2) ? 2 : 3;
  const int nxb = width / 8;
  const int plane = width * 8; // bytes per 8-channel block of a row
  const aie::vector<uint8, 64> zero = aie::zeros<uint8, 64>();

  for (int ob = 0; ob < OCB; ob += 2) {
    const int8_t *w0 = wts + ob * WBLK;
    const int8_t *w1 = w0 + WBLK;
    aie::accum<acc32, 64> bias0, bias1;
    bias0.from_vector(aie::load_v<64>(bias + ob * 64));
    bias1.from_vector(aie::load_v<64>(bias + (ob + 1) * 64));

    for (int xb = 0; xb < nxb; xb++) {
      MMUL acc0(bias0);
      MMUL acc1(bias1);
      for (int ky = ky0; ky < ky1; ky++) {
        const uint8_t *row = lines[ky] + xb * 64;
        const int8_t *wk0 = w0 + ky * ICB * 3 * 64;
        const int8_t *wk1 = w1 + ky * ICB * 3 * 64;
        for (int icb = 0; icb < ICB; icb++) {
          const uint8_t *p = row + icb * plane;
          aie::vector<uint8, 64> cur = aie::load_v<64>(p);
          aie::vector<uint8, 64> prev = xb > 0 ? aie::load_v<64>(p - 64) : zero;
          aie::vector<uint8, 64> next =
              xb < nxb - 1 ? aie::load_v<64>(p + 64) : zero;
          // pixels x-1..x+6 and x+1..x+8, each with its 8 channels
          aie::vector<uint8, 64> left = aie::shuffle_up_fill(cur, prev, 8);
          aie::vector<uint8, 64> right = aie::shuffle_down_fill(cur, next, 8);
          acc0.mac(left, aie::load_v<64>(wk0));
          acc1.mac(left, aie::load_v<64>(wk1));
          acc0.mac(cur, aie::load_v<64>(wk0 + 64));
          acc1.mac(cur, aie::load_v<64>(wk1 + 64));
          acc0.mac(right, aie::load_v<64>(wk0 + 128));
          acc1.mac(right, aie::load_v<64>(wk1 + 128));
          wk0 += 192;
          wk1 += 192;
        }
      }
      aie::store_v(out + ob * plane + xb * 64,
                   acc0.template to_vector<uint8>(shift));
      aie::store_v(out + (ob + 1) * plane + xb * 64,
                   acc1.template to_vector<uint8>(shift));
    }
  }

  for (int x = 0; x < width; x++) {
    if (x >= valid_lo && x < valid_hi)
      continue;
    for (int ob = 0; ob < OCB; ob++)
      for (int c = 0; c < 8; c++)
        out[ob * plane + x * 8 + c] = 0;
  }
}

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
  conv3x3_u8_row<CONV3X3_CIN, CONV3X3_COUT>(line0, line1, line2, params, out,
                                            width, check, shift, valid_lo,
                                            valid_hi);
}

} // extern "C"
