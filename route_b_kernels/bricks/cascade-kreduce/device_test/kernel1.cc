// Cascade device test, kernel 1 (aie2p, Peano). Seeds a 16-lane int32 vector
// with lane 0 = 14 and writes it to the master cascade using aie::cascade_out
// (aie_api/cascade.hpp, the accessor under test). Mirrors the mlir-aie
// npu-xrt/cascade_flows kernel1, but through the aie_api typed accessor.
#include <aie_api/aie.hpp>
extern "C" void extern_kernel1() {
  int32_t seed[16] = {14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  aie::cascade_out(aie::load_v<16>(seed));
}
