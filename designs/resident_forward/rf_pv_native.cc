// x V on the native bf16 mmul (broadcast vmac.f, attn_ref.vmac_f) for the rows whose key block is
// partially visible (K055). Same operand layouts as mm.cc's 8x8x8 row-major path; loops kept
// rolled for program memory.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void rf_pv_native_slice(const bfloat16 *__restrict pA, const bfloat16 *__restrict pB,
                                   float *__restrict o, int32_t slice) {
  using MM = aie::mmul<8, 8, 8, bfloat16, bfloat16, accfloat>;
  constexpr unsigned RA = DIM_M / 8, CA = DIM_K / 8, CB = DIM_N / 8;
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *pC = o + slice * DIM_M * DIM_N;
  AIE_LOOP_NO_UNROLL
  for (unsigned z = 0; z < RA; ++z)
    AIE_LOOP_NO_UNROLL
    for (unsigned j = 0; j < CB; ++j) {
      MM c(aie::load_v<64>(pC + (z * CB + j) * 64));
      for (unsigned i = 0; i < CA; ++i)
        c.mac(aie::load_v<64>(pA + (z * CA + i) * 64), aie::load_v<64>(pB + (i * CB + j) * 64));
      aie::store_v(pC + (z * CB + j) * 64, c.to_vector<float>());
    }
}
