#include <aie_api/aie.hpp>
#include <stdint.h>
extern "C" void one_tofloat(const int32_t* in, float* out){
  auto v = aie::load_v<64>(in);
  aie::store_v(out, aie::to_float(v, 0));
}
