// Cascade device test, kernel 3 (aie2p, Peano). Reads the 16-lane vector from
// the slave cascade (aie::cascade_in_i32) and writes buf[5] = lane0 + 100, so
// the host sees 214 iff the value propagated across both cascade hops.
#include <aie_api/aie.hpp>
extern "C" void extern_kernel3(int32_t *restrict buf, int N) {
  aie::vector<int32_t, 16> v = aie::cascade_in_i32();
  int32_t recv[16];
  aie::store_v(recv, v);
  for (int i = 0; i < N; i++)
    buf[i] = (i == 5) ? recv[0] + 100 : 0;
}
