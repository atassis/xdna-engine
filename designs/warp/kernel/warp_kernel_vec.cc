// RIFE backward warp, vectorized. Companion to warp_kernel.cc (kept as the scalar reference
// point). AIE2P has no scalar hardware FPU -- plain `float` arithmetic on the scalar core is
// a software-emulated library call (the reason warp_kernel.cc's scalar cyc/px is high), while
// the VECTOR unit's float/int32 ALU is native (proven by fsr1_kernel_vec.cc's 16x speedup over
// its own scalar kernel on the same hardware). So every op that ISN'T the gather itself moves
// to `aie::vector<T,VW>` here: position math, floor, clamp, and the bilinear lerp. Only the 4
// taps/channel/pixel stay scalar loads -- there is no shared address across lanes to vectorize
// them onto (RIFE's flow is independent per pixel; see warp_kernel.cc's header for why this
// differs from FSR1's fixed phase table).
//
// Format choices:
//  - Flow: fixed-point int16, Q8.7 (7 fractional bits, scale 128). Measured flow magnitude on
//    a real model run (see designs/warp/README.md) has p99.99=59.8px, max=66.3px; Q8.7 covers
//    +/-255.99px, >4x the measured max. This turns the per-pixel address computation into
//    integer add + arithmetic shift (floor-by-shift is EXACT for two's-complement), avoiding
//    any float floor/round-mode question entirely -- `aie::to_fixed<float>` on this target
//    rounds-to-nearest, not floor, so a float-based floor would need its own correction step.
//  - Pixel data: bf16. Keeps full float dynamic range (unlike int8, whose activation-path
//    accuracy is untested per rife-sizing.md) while halving L1 bytes vs f32.
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stddef.h>

#ifndef WARP_TILE_W
#define WARP_TILE_W 16
#endif
#ifndef WARP_TILE_H
#define WARP_TILE_H 8
#endif
#ifndef WARP_HALO
#define WARP_HALO 8
#endif
#ifndef WARP_CH
#define WARP_CH 3
#endif

#define VW WARP_TILE_W
#define WPAD_W (WARP_TILE_W + 2 * WARP_HALO)
#define WPAD_H (WARP_TILE_H + 2 * WARP_HALO)
#define FLOW_FRAC_BITS 7
#define FLOW_SCALE (1 << FLOW_FRAC_BITS)

using fvec = ::aie::vector<float, VW>;
using ivec = ::aie::vector<int32_t, VW>;

static inline fvec vmul(fvec a, fvec b) { return ::aie::mul(a, b).template to_vector<float>(); }

// bf16 <-> f32 widen/narrow, the idiom already proven in cast_quant_bf16_int8.cc.
static inline fvec widen_bf16(::aie::vector<bfloat16, VW> v) {
  ::aie::accum<accfloat, VW> a;
  a.from_vector(v);
  return a.template to_vector<float>();
}
// conv_even (round-to-nearest-even), matching a host f32->bf16 pack -- see cast_f32_bf16.cc.
// Without this the narrow uses AIE's default (truncation), which is what the ~4e-3 rel_l2
// against the numpy golden's astype(bfloat16) (round-to-nearest) traced back to.
static inline ::aie::vector<bfloat16, VW> narrow_bf16(fvec v) {
  ::aie::rounding_mode saved = ::aie::swap_rounding(::aie::rounding_mode::conv_even);
  ::aie::accum<accfloat, VW> a;
  a.from_vector(v);
  auto r = a.template to_vector<bfloat16>();
  ::aie::set_rounding(saved);
  return r;
}

// in_padded: [WPAD_H, WPAD_W, WARP_CH] bf16, tile origin at (WARP_HALO, WARP_HALO).
// flow_fixed: [2, WARP_TILE_H, WARP_TILE_W] int16, Q8.7 (fx then fy).
// out: [WARP_TILE_H, WARP_TILE_W, WARP_CH] bf16.
extern "C" void warp_kernel_vec(const bfloat16 *in_padded, const int16_t *flow_fixed,
                                bfloat16 *out) {
  static const int32_t LANE_X_INIT[VW] = {
#define L(i) ((i) + WARP_HALO) * FLOW_SCALE
      L(0), L(1), L(2), L(3), L(4), L(5), L(6), L(7),
#if VW > 8
      L(8), L(9), L(10), L(11), L(12), L(13), L(14), L(15),
#endif
#undef L
  };
  const ivec lane_x0 = ::aie::load_v<VW>(LANE_X_INIT);
  const ivec wpad_w1 = ::aie::broadcast<int32_t, VW>(WPAD_W - 1);
  const ivec wpad_h1 = ::aie::broadcast<int32_t, VW>(WPAD_H - 1);
  const ivec zero_i = ::aie::zeros<int32_t, VW>();
  const int32_t flow_row_px = WARP_TILE_H * WARP_TILE_W;

  for (int y = 0; y < WARP_TILE_H; y++) {
    // aie::unpack (NOT cast_to, which reinterprets bits at fixed total width and would
    // shuffle pairs of int16 into one int32) is the value-preserving sign-extending widen.
    ivec fx32 = ::aie::unpack(::aie::load_v<VW>(flow_fixed + y * VW));
    ivec fy32 = ::aie::unpack(::aie::load_v<VW>(flow_fixed + flow_row_px + y * VW));

    ivec sx = ::aie::add(lane_x0, fx32);
    ivec sy = ::aie::add(fy32, ::aie::broadcast<int32_t, VW>((y + WARP_HALO) * FLOW_SCALE));

    // Floor by arithmetic shift (exact for Q8.7 two's complement) -- see file header.
    ivec x0i = ::aie::downshift(sx, FLOW_FRAC_BITS);
    ivec y0i = ::aie::downshift(sy, FLOW_FRAC_BITS);
    ivec fracx_i = ::aie::sub(sx, ::aie::upshift(x0i, FLOW_FRAC_BITS));
    ivec fracy_i = ::aie::sub(sy, ::aie::upshift(y0i, FLOW_FRAC_BITS));
    // to_float's shift arg divides by 2^shift directly -- fracx_i/FLOW_SCALE in one op
    // (cast_to would reinterpret int32 bits as float, which is wrong here, not a widen).
    fvec wx = ::aie::to_float<float>(fracx_i, FLOW_FRAC_BITS);
    fvec wy = ::aie::to_float<float>(fracy_i, FLOW_FRAC_BITS);
    fvec one_m_wx = ::aie::sub(1.0f, wx);
    fvec one_m_wy = ::aie::sub(1.0f, wy);

    ivec x0c = ::aie::min(::aie::max(x0i, zero_i), wpad_w1);
    ivec x1c = ::aie::min(::aie::max(::aie::add(x0i, 1), zero_i), wpad_w1);
    ivec y0c = ::aie::min(::aie::max(y0i, zero_i), wpad_h1);
    ivec y1c = ::aie::min(::aie::max(::aie::add(y0i, 1), zero_i), wpad_h1);

    alignas(64) int32_t x0b[VW], x1b[VW], y0b[VW], y1b[VW];
    ::aie::store_v(x0b, x0c);
    ::aie::store_v(x1b, x1c);
    ::aie::store_v(y0b, y0c);
    ::aie::store_v(y1b, y1c);

    for (int ch = 0; ch < WARP_CH; ch++) {
      alignas(64) bfloat16 ta[VW], tb[VW], tc[VW], td[VW];
      for (int i = 0; i < VW; i++) {
        const bfloat16 *row0 = in_padded + (size_t)y0b[i] * WPAD_W * WARP_CH;
        const bfloat16 *row1 = in_padded + (size_t)y1b[i] * WPAD_W * WARP_CH;
        ta[i] = row0[x0b[i] * WARP_CH + ch];
        tb[i] = row0[x1b[i] * WARP_CH + ch];
        tc[i] = row1[x0b[i] * WARP_CH + ch];
        td[i] = row1[x1b[i] * WARP_CH + ch];
      }
      fvec a = widen_bf16(::aie::load_v<VW>(ta));
      fvec b = widen_bf16(::aie::load_v<VW>(tb));
      fvec c = widen_bf16(::aie::load_v<VW>(tc));
      fvec d = widen_bf16(::aie::load_v<VW>(td));

      fvec top = ::aie::add(vmul(a, one_m_wx), vmul(b, wx));
      fvec bot = ::aie::add(vmul(c, one_m_wx), vmul(d, wx));
      fvec res = ::aie::add(vmul(top, one_m_wy), vmul(bot, wy));

      alignas(64) bfloat16 outb[VW];
      ::aie::store_v(outb, narrow_bf16(res));
      for (int i = 0; i < VW; i++)
        out[(y * WARP_TILE_W + i) * WARP_CH + ch] = outb[i];
    }
  }
}
