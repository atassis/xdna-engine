// Stand-in for the attention phase's L1 use (rf-one-image-overlay): fills a buffer sized like one
// FusedAttnX2 role (hd 256) so the overlaid gate/up storage is really clobbered between blocks.
#include <aie_api/aie.hpp>
#include <stdint.h>

static inline void fill(uint8_t *buf, int32_t n, int32_t seed) {
  aie::vector<int32_t, 16> v = aie::broadcast<int32_t, 16>(seed);
  for (int32_t i = 0; i < n; i += 64) {
    aie::store_v(reinterpret_cast<int32_t *>(buf + i), v);
    v = aie::add(v, aie::broadcast<int32_t, 16>(0x01010101));
  }
}

extern "C" {
void attn_standin_qk(uint8_t *buf, int32_t seed) { fill(buf, 45056, seed); }
void attn_standin_pv(uint8_t *buf, int32_t seed) { fill(buf, 41088, seed); }
void attn_standin_sm(uint8_t *buf, int32_t seed) { fill(buf, 26048, seed); }
}
