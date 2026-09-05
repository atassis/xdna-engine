// Cascade device test, kernel 2 (aie2p, Peano). Reads the 16-lane vector from
// the slave cascade (aie::cascade_in_i32), adds 100 to lane 0, and forwards it
// on the master cascade (aie::cascade_out). Exercises a cascade read+write hop.
#include <aie_api/aie.hpp>
extern "C" void extern_kernel2() {
  aie::vector<int32_t, 16> v = aie::cascade_in_i32();
  int32_t buf[16];
  aie::store_v(buf, v);
  buf[0] += 100;               // 14 -> 114
  aie::cascade_out(aie::load_v<16>(buf));
}
