// fused_attn.cc's x V kernels (FA_ROWS=16), fa_inv_l as rf_intmath's correctly rounded division.
#define fa_inv_l fa_inv_l_divsf3
#include "fused_attn.cc"
#undef fa_inv_l
#include "rf_intmath.h"
extern "C" void fa_inv_l(float *cl, float *inv) {
  ::aie::set_rounding(aie::rounding_mode::floor);
  for (int h = 0; h < fa::ROWS; h += 16)
    aie::store_v(reinterpret_cast<int32_t *>(inv + h),
                 rf::div_rn(rf::bc((int32_t)rf::bits(1.0f)), aie::load_v<16>(reinterpret_cast<const int32_t *>(cl + fa::ROWS + h))));
  ::aie::set_rounding(aie::rounding_mode::conv_even);
}
