// RIFE backward warp (bilinear grid_sample), scalar, one AIE2P core.
//
// Semantics match model/warplayer.py::warp (Practical-RIFE): bilinear, padding_mode='border',
// align_corners=True, per-pixel flow in PIXEL units. Verified against torch grid_sample via
// numpy in warp_ref.py (rel_l2 9.0e-7) before this transliteration.
//
// No AIE2P unit does data-dependent 2-D addressing (no texture sampler, dma_bd is a fixed
// pattern, parallel_lookup is a <=32-lane 1-D table). So the design is: DMA brings one output
// TILE plus a fixed HALO into L1 (a static, non-data-dependent window); the core then does a
// per-pixel SCALAR gather -- ordinary pointer arithmetic on the already-resident L1 buffer,
// which is not restricted by DMA addressing rules. Pixels whose flow exceeds HALO clamp to the
// halo edge instead of the true image border (measured fallback rate: see rife-warp-halo-sizing).
#include <stdint.h>
#include <stddef.h>

#ifndef WARP_TILE_W
#define WARP_TILE_W 32
#endif
#ifndef WARP_TILE_H
#define WARP_TILE_H 8
#endif
#ifndef WARP_HALO
#define WARP_HALO 16
#endif
#ifndef WARP_CH
#define WARP_CH 3
#endif

#define WPAD_W (WARP_TILE_W + 2 * WARP_HALO)
#define WPAD_H (WARP_TILE_H + 2 * WARP_HALO)

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float floorf_(float v) { return (float)(int)(v >= 0.0f ? v : v - 0.999999f); }

#define WARP_TILE_PX (WARP_TILE_H * WARP_TILE_W)

// in_padded: [WPAD_H, WPAD_W, WARP_CH], tile origin at (WARP_HALO, WARP_HALO).
// flow: [2, WARP_TILE_H, WARP_TILE_W] (fx then fy), packed into one buffer -- a core tile
// has only 2 input DMA channels, so fx/fy share the second one alongside in_padded.
// out: [WARP_TILE_H, WARP_TILE_W, WARP_CH].
extern "C" void warp_kernel(const float *in_padded, const float *flow, float *out) {
  const float *fx = flow;
  const float *fy = flow + WARP_TILE_PX;
  for (int y = 0; y < WARP_TILE_H; y++) {
    for (int x = 0; x < WARP_TILE_W; x++) {
      float sx = (float)(x + WARP_HALO) + fx[y * WARP_TILE_W + x];
      float sy = (float)(y + WARP_HALO) + fy[y * WARP_TILE_W + x];

      float x0f = floorf_(sx), y0f = floorf_(sy);
      float wx = sx - x0f, wy = sy - y0f;
      int x0 = clampi((int)x0f, 0, WPAD_W - 1);
      int x1 = clampi((int)x0f + 1, 0, WPAD_W - 1);
      int y0 = clampi((int)y0f, 0, WPAD_H - 1);
      int y1 = clampi((int)y0f + 1, 0, WPAD_H - 1);

      const float *row0 = in_padded + (size_t)y0 * WPAD_W * WARP_CH;
      const float *row1 = in_padded + (size_t)y1 * WPAD_W * WARP_CH;
      const float *a = row0 + x0 * WARP_CH;
      const float *b = row0 + x1 * WARP_CH;
      const float *c = row1 + x0 * WARP_CH;
      const float *d = row1 + x1 * WARP_CH;

      float *dst = out + ((size_t)y * WARP_TILE_W + x) * WARP_CH;
      for (int ch = 0; ch < WARP_CH; ch++) {
        float top = a[ch] * (1.0f - wx) + b[ch] * wx;
        float bot = c[ch] * (1.0f - wx) + d[ch] * wx;
        dst[ch] = top * (1.0f - wy) + bot * wy;
      }
    }
  }
}
