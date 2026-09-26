// 1x1 convolution over the channel concatenation of NSRC int8 rows, without
// materializing the concatenation: out = sat8((sum_s W_s x_s + bias) >> shift),
// >> rounding half up. Each source may carry its own quantization scale; fold
// it into that source's weight slice (golden.pack_params).
//
//   in0..in3  source rows, each [CSRC/8][width+16][8] int8, margins zero
//   out       [COUT/8][width+16][8] int8, margins written zero
//   params    weights [COUT/8][NSRC][CSRC/8][8 ic][8 oc] int8,
//             then bias [COUT/8][8 px][8 oc] int32
// Output pixels outside [valid_lo, valid_hi) are written as zero.
// width % 16 == 0, CSRC % 8 == 0, COUT % 16 == 0; unused sources may repeat a pointer
// when NSRC < 4.

#include <aie_api/aie.hpp>
#include <stdint.h>

#define C1_PRAGMA(x) _Pragma(#x)
#define C1_LOOP_RANGE(lo, hi)                                                  \
  C1_PRAGMA(clang loop min_iteration_count(lo))                                \
  C1_PRAGMA(clang loop max_iteration_count(hi))

#ifndef CONV1X1_NSRC
#define CONV1X1_NSRC 4
#endif
#ifndef CONV1X1_CSRC
#define CONV1X1_CSRC 48
#endif
#ifndef CONV1X1_COUT
#define CONV1X1_COUT 48
#endif

namespace {

inline aie::vector<int8, 64> mask_block(aie::vector<int8, 64> v, int x0, int lo,
                                        int hi) {
  if (x0 >= lo && x0 + 8 <= hi)
    return v;
  uint64_t bits = 0;
  for (int i = 0; i < 8; i++)
    if (x0 + i >= lo && x0 + i < hi)
      bits |= uint64_t(0xff) << (8 * i);
  return aie::select(aie::zeros<int8, 64>(), v, aie::mask<64>::from_uint64(bits));
}

template <int NSRC, int CSRC, int COUT>
__attribute__((noinline)) void
conv1x1_cat_core(const int8_t *__restrict in0, const int8_t *__restrict in1,
                 const int8_t *__restrict in2, const int8_t *__restrict in3,
                 const int8_t *__restrict params, int8_t *__restrict out,
                 int width, int shift, int valid_lo, int valid_hi) {
  using MMUL = aie::mmul<8, 8, 8, int8, int8>;
  static_assert(CSRC % 8 == 0 && COUT % 16 == 0 && NSRC >= 1 && NSRC <= 4);
  constexpr int SCB = CSRC / 8;
  constexpr int OCB = COUT / 8;
  constexpr int WBLK = NSRC * SCB * 64; // weight bytes per output-channel block

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::positive_inf);

  const int32_t *bias = reinterpret_cast<const int32_t *>(params + OCB * WBLK);
  const int8_t *src[4] = {in0, in1, in2, in3};
  const int plane = (width + 16) * 8;
  const aie::vector<int8, 64> zero = aie::zeros<int8, 64>();

  for (int ob = 0; ob < OCB; ob += 2) {
    int8_t *o0 = out + ob * plane;
    int8_t *o1 = o0 + plane;
    aie::store_v(o0, zero);
    aie::store_v(o1, zero);
    aie::store_v(o0 + plane - 64, zero);
    aie::store_v(o1 + plane - 64, zero);
    MMUL::accum_type bias0, bias1;
    bias0.from_vector(aie::load_v<64>(bias + ob * 64));
    bias1.from_vector(aie::load_v<64>(bias + (ob + 1) * 64));

    for (int x0 = 0; x0 < width; x0 += 16) {
      MMUL a00(bias0), a01(bias0), a10(bias1), a11(bias1);
      const int8_t *__restrict wk0 = params + ob * WBLK;
      const int8_t *__restrict wk1 = wk0 + WBLK;
      for (int s = 0; s < NSRC; s++) {
        const int8_t *__restrict p = src[s] + (x0 + 8) * 8;
        C1_LOOP_RANGE(SCB, SCB)
        for (int cb = 0; cb < SCB; cb++) {
          aie::vector<int8, 64> c0 = aie::load_v<64>(p);
          aie::vector<int8, 64> c1 = aie::load_v<64>(p + 64);
          p += plane;
          aie::vector<int8, 64> b0 = aie::load_v<64>(wk0);
          aie::vector<int8, 64> b1 = aie::load_v<64>(wk1);
          wk0 += 64;
          wk1 += 64;
          a00.mac(c0, b0);
          a01.mac(c1, b0);
          a10.mac(c0, b1);
          a11.mac(c1, b1);
        }
      }
      const int c = (x0 + 8) * 8;
      aie::store_v(o0 + c, mask_block(a00.to_vector<int8>(shift), x0, valid_lo, valid_hi));
      aie::store_v(o0 + c + 64,
                   mask_block(a01.to_vector<int8>(shift), x0 + 8, valid_lo, valid_hi));
      aie::store_v(o1 + c, mask_block(a10.to_vector<int8>(shift), x0, valid_lo, valid_hi));
      aie::store_v(o1 + c + 64,
                   mask_block(a11.to_vector<int8>(shift), x0 + 8, valid_lo, valid_hi));
    }
  }
}

} // namespace

extern "C" {

void conv1x1_cat_i8(const int8_t *in0, const int8_t *in1, const int8_t *in2,
                    const int8_t *in3, const int8_t *params, int8_t *out,
                    int32_t width, int32_t shift, int32_t valid_lo,
                    int32_t valid_hi) {
  conv1x1_cat_core<CONV1X1_NSRC, CONV1X1_CSRC, CONV1X1_COUT>(
      in0, in1, in2, in3, params, out, width, shift, valid_lo, valid_hi);
}

} // extern "C"
