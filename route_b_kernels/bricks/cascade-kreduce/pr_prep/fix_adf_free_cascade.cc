// FIX DIRECTION for the aie_api cascade/adf.h gap.
// Reach the hardware cascade WITHOUT <adf.h>, via the Peano intrinsics
// __builtin_aie2p_{mcd_write,scd_read}_* that already exist in this toolchain.
// A proposed adf-free aie::cascade_out()/cascade_in() wrapper would sit on exactly these.
#include <aie_api/aie.hpp>
using v16i32 = aie::vector<int32_t, 16>;
extern "C" v16i32 cascade_passthrough(v16i32 x) {
  __builtin_aie2p_mcd_write_vec(x, 0);      // PUT to master cascade datapath (adf-free)
  return __builtin_aie2p_scd_read_vec(0);   // GET from slave cascade datapath (adf-free)
}
