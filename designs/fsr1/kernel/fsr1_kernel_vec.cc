// FSR1 EASU+RCAS, fixed x3 scale, VECTORIZED across source columns.
//
// Mapping (verified against cpu_ref.py first in designs/fsr1/kernel/phase_ref.py, host-only,
// float32-ULP match, max diff 3.8e-6): at x3 scale, output column ox = 3t+px maps to source
// column fpx(t) = t + OFFSET_X[px], with a per-phase CONSTANT fraction ppx = FRAC_X[px] (not a
// function of t). So for one phase px, every EASU tap is a t-shifted contiguous view of one
// source row -- no gather needed, just aie::vector loads at different offsets. Same for py/fpy
// on the row axis (kept as a scalar outer loop here).
//
// COMPUTE/FORMAT choice (aie2p-brick-catalog.md, per-op): fp32 vector add/sub/mul throughout
// (aie::vector<float,VW>, native aie2p elementwise float ops -- a different brick from the
// mmul unit's bf16-emulated fp32 path, which targets M>=8 GEMM, not this M=1-per-lane regime).
//
// Reciprocals: FIRST tried aie::inv/aie::invsqrt (hardware SFU) in place of FSR1's own
// APrxLoRcpF1/APrxLoRsqF1 bit-trick approximations, on the theory that aie2p's SFU is the
// specialized brick FSR1's bit tricks exist to work around (GPU shader ALUs of FSR1's era
// lacked a fast reciprocal unit). REVERTED after device testing found rel_l2 0.21 against
// cpu_ref.py -- traced (not assumed) to a real semantic difference, not a logic bug: on a flat
// run (dc=cb=0), APrxLoRcpF1(0) returns a large but FINITE float (0x7ef07ebb bit pattern), while
// an IEEE/SFU reciprocal of 0 is +inf. FSR1's approximation is not just "faster", it degrades
// gracefully on the degenerate input its own direction/length math routinely produces on flat
// image regions -- confirmed on host by swapping cpu_ref.py's phase-decomposed reformulation to
// exact 1/x, which produces the same NaN propagation on the same test image (see README). So the
// bit tricks are re-implemented here bit-for-bit, vectorized via int32 bit-cast + integer vector
// ops (aie::vector<T>::cast_to<int32_t>(), aie::sub, operator>>) -- same hardware brick choice as
// the scalar kernel, just SIMD. The exact case (FSR1's own plain ARcpF1, used once at the very
// end of EASU) still uses aie::inv, which approximates a true reciprocal directly.
#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#ifndef FSR1_IN_W
#define FSR1_IN_W 16
#endif
#ifndef FSR1_IN_H
#define FSR1_IN_H 6
#endif
#define VW FSR1_IN_W
#define FSR1_OUT_W (FSR1_IN_W * 3)
#define FSR1_OUT_H (FSR1_IN_H * 3)

using fvec = aie::vector<float, VW>;

static const int OFFSET_X[3] = {-1, 0, 0};
static const float FRAC_X[3] = {2.0f / 3.0f, 0.0f, 1.0f / 3.0f};

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline fvec vmul(fvec a, fvec b) { return aie::mul(a, b).template to_vector<float>(); }
static inline fvec vmuls(fvec a, float s) { return aie::mul(a, s).template to_vector<float>(); }

// FSR1's APrxLoRcpF1/APrxLoRsqF1 bit tricks, vectorized -- see the file header for why these
// are kept bit-for-bit rather than swapped for the hardware SFU aie::inv/aie::invsqrt.
static inline fvec aprx_lo_rcp_v(fvec a) {
  auto bits = a.template cast_to<int32_t>();
  auto r = aie::sub((int32_t)0x7EF07EBB, bits);
  return r.template cast_to<float>();
}
static inline fvec aprx_lo_rsq_v(fvec a) {
  auto bits = a.template cast_to<int32_t>();
  auto shifted = bits >> 1u;
  auto r = aie::sub((int32_t)0x5F347D74, shifted);
  return r.template cast_to<float>();
}

// VW consecutive elements of channel `ch`, row `y` (clamped), starting at source column `x0`
// (each lane independently clamped -- a scalar gather, not yet a SIMD one; see README).
static fvec load_tap(const float *planar, int ch, int y, int x0) {
  int yc = clampi(y, 0, FSR1_IN_H - 1);
  alignas(64) float buf[VW];
  const float *row = planar + (ch * FSR1_IN_H + yc) * FSR1_IN_W;
  for (int i = 0; i < VW; i++) buf[i] = row[clampi(x0 + i, 0, FSR1_IN_W - 1)];
  return aie::load_v<VW>(buf);
}

static inline fvec vluma(fvec r, fvec g, fvec b) {
  return aie::add(vmuls(b, 0.5f), aie::add(vmuls(r, 0.5f), g));
}

struct EasuAcc { fvec dirx, diry, len; };

static void easu_set_v(EasuAcc &a, float ppx, float ppy, int mask,
                       fvec lA, fvec lB, fvec lC, fvec lD, fvec lE) {
  float w;
  if (mask == 0) w = (1.0f - ppx) * (1.0f - ppy);
  else if (mask == 1) w = ppx * (1.0f - ppy);
  else if (mask == 2) w = (1.0f - ppx) * ppy;
  else w = ppx * ppy;

  fvec dc = aie::sub(lD, lC), cb = aie::sub(lC, lB);
  fvec lenX = aie::inv(aie::max(aie::abs(dc), aie::abs(cb)));
  fvec dirX = aie::sub(lD, lB);
  a.dirx = aie::add(a.dirx, vmuls(dirX, w));
  lenX = aie::min(aie::max(vmul(aie::abs(dirX), lenX), 0.0f), 1.0f);
  lenX = vmul(lenX, lenX);
  a.len = aie::add(a.len, vmuls(lenX, w));

  fvec ec = aie::sub(lE, lC), ca = aie::sub(lC, lA);
  fvec lenY = aie::inv(aie::max(aie::abs(ec), aie::abs(ca)));
  fvec dirY = aie::sub(lE, lA);
  a.diry = aie::add(a.diry, vmuls(dirY, w));
  lenY = aie::min(aie::max(vmul(aie::abs(dirY), lenY), 0.0f), 1.0f);
  lenY = vmul(lenY, lenY);
  a.len = aie::add(a.len, vmuls(lenY, w));
}

// optnone: at -O1/-O2/-Os Peano's register allocator fails ("ran out of registers") on this
// function's live-range shape (6 concurrent phase-state vectors x aggressive ILP scheduling
// extends their live ranges past where they're actually needed -- confirmed by -O0 compiling
// clean with the exact same code). A real toolchain-scheduling limit for this function's
// dependency shape, not a fundamental impossibility; forcing -O0 HERE only (optnone) is the
// workaround pending a from-scratch restructure. See README.
__attribute__((optnone)) static void easu_tap_v(fvec aC[3], fvec &aW, float tapx, float tapy, float ppx, float ppy,
                       fvec dirx, fvec diry, fvec len2x, fvec len2y, fvec lob, fvec clp,
                       const fvec c[3]) {
  float offx = tapx - ppx, offy = tapy - ppy;
  fvec vx = vmul(aie::add(vmuls(dirx, offx), vmuls(diry, offy)), len2x);
  fvec vy = vmul(aie::add(vmuls(dirx, -offy), vmuls(diry, offx)), len2y);
  fvec d2 = aie::min(aie::add(vmul(vx, vx), vmul(vy, vy)), clp);
  fvec wB = aie::sub(vmuls(d2, 2.0f / 5.0f), 1.0f);
  fvec wA = aie::sub(vmul(d2, lob), 1.0f);
  wB = vmul(wB, wB);
  wA = vmul(wA, wA);
  wB = aie::sub(vmuls(wB, 25.0f / 16.0f), (25.0f / 16.0f - 1.0f));
  fvec w = vmul(wB, wA);
  aC[0] = aie::add(aC[0], vmul(c[0], w));
  aC[1] = aie::add(aC[1], vmul(c[1], w));
  aC[2] = aie::add(aC[2], vmul(c[2], w));
  aW = aie::add(aW, w);
}

// Computes one phase-row: real output row = 3*s + py, columns {3t+px : t=0..VW-1}.
// `planar` is [channel][row][col], IN_H*IN_W each. Writes `out[c]` (VW lanes).
__attribute__((noinline)) static void easu_phase_row(const float *planar, int s, int px, int py, fvec out[3]) {
  float ppx = FRAC_X[px], ppy = FRAC_X[py];
  int fpy = s + OFFSET_X[py];
  int fpx0 = OFFSET_X[px];

  fvec bR = load_tap(planar, 0, fpy - 1, fpx0), bG = load_tap(planar, 1, fpy - 1, fpx0),
       bB = load_tap(planar, 2, fpy - 1, fpx0);
  fvec cR = load_tap(planar, 0, fpy - 1, fpx0 + 1), cG = load_tap(planar, 1, fpy - 1, fpx0 + 1),
       cB = load_tap(planar, 2, fpy - 1, fpx0 + 1);
  fvec eR = load_tap(planar, 0, fpy, fpx0 - 1), eG = load_tap(planar, 1, fpy, fpx0 - 1),
       eB = load_tap(planar, 2, fpy, fpx0 - 1);
  fvec fR = load_tap(planar, 0, fpy, fpx0), fG = load_tap(planar, 1, fpy, fpx0),
       fB = load_tap(planar, 2, fpy, fpx0);
  fvec gR = load_tap(planar, 0, fpy, fpx0 + 1), gG = load_tap(planar, 1, fpy, fpx0 + 1),
       gB = load_tap(planar, 2, fpy, fpx0 + 1);
  fvec hR = load_tap(planar, 0, fpy, fpx0 + 2), hG = load_tap(planar, 1, fpy, fpx0 + 2),
       hB = load_tap(planar, 2, fpy, fpx0 + 2);
  fvec iR = load_tap(planar, 0, fpy + 1, fpx0 - 1), iG = load_tap(planar, 1, fpy + 1, fpx0 - 1),
       iB = load_tap(planar, 2, fpy + 1, fpx0 - 1);
  fvec jR = load_tap(planar, 0, fpy + 1, fpx0), jG = load_tap(planar, 1, fpy + 1, fpx0),
       jB = load_tap(planar, 2, fpy + 1, fpx0);
  fvec kR = load_tap(planar, 0, fpy + 1, fpx0 + 1), kG = load_tap(planar, 1, fpy + 1, fpx0 + 1),
       kB = load_tap(planar, 2, fpy + 1, fpx0 + 1);
  fvec lR = load_tap(planar, 0, fpy + 1, fpx0 + 2), lG = load_tap(planar, 1, fpy + 1, fpx0 + 2),
       lB_ = load_tap(planar, 2, fpy + 1, fpx0 + 2);
  fvec nR = load_tap(planar, 0, fpy + 2, fpx0), nG = load_tap(planar, 1, fpy + 2, fpx0),
       nB = load_tap(planar, 2, fpy + 2, fpx0);
  fvec oR = load_tap(planar, 0, fpy + 2, fpx0 + 1), oG = load_tap(planar, 1, fpy + 2, fpx0 + 1),
       oB = load_tap(planar, 2, fpy + 2, fpx0 + 1);

  fvec bL = vluma(bR, bG, bB), cL = vluma(cR, cG, cB);
  fvec eL = vluma(eR, eG, eB), fL = vluma(fR, fG, fB), gL = vluma(gR, gG, gB), hL = vluma(hR, hG, hB);
  fvec iL = vluma(iR, iG, iB), jL = vluma(jR, jG, jB), kL = vluma(kR, kG, kB), lL = vluma(lR, lG, lB_);
  fvec nL = vluma(nR, nG, nB), oL = vluma(oR, oG, oB);

  EasuAcc acc{aie::zeros<float, VW>(), aie::zeros<float, VW>(), aie::zeros<float, VW>()};
  easu_set_v(acc, ppx, ppy, 0, bL, eL, fL, gL, jL);
  easu_set_v(acc, ppx, ppy, 1, cL, fL, gL, hL, kL);
  easu_set_v(acc, ppx, ppy, 2, fL, iL, jL, kL, nL);
  easu_set_v(acc, ppx, ppy, 3, gL, jL, kL, lL, oL);

  fvec dirR = aie::add(vmul(acc.dirx, acc.dirx), vmul(acc.diry, acc.diry));
  auto zro = aie::lt(dirR, 1.0f / 32768.0f);
  fvec rdirR = aie::invsqrt(dirR);
  fvec ones = aie::broadcast<float, VW>(1.0f);
  rdirR = aie::select(rdirR, ones, zro);
  fvec dirx = vmul(aie::select(acc.dirx, ones, zro), rdirR);
  fvec diry = vmul(acc.diry, rdirR);

  fvec length = vmuls(acc.len, 0.5f);
  length = vmul(length, length);
  fvec mx = aie::max(aie::abs(dirx), aie::abs(diry));
  fvec stretch = vmul(aie::add(vmul(dirx, dirx), vmul(diry, diry)), aie::inv(mx));
  fvec len2x = aie::add(vmul(aie::sub(stretch, 1.0f), length), 1.0f);
  fvec len2y = aie::add(vmuls(length, -0.5f), 1.0f);
  fvec lob = aie::add(vmuls(length, (1.0f / 4.0f - 0.04f - 0.5f)), 0.5f);
  fvec clp = aie::inv(lob);

  fvec min4[3], max4[3];
  fvec jc[3] = {jR, jG, jB}, gc[3] = {gR, gG, gB}, ic[3] = {iR, iG, iB}, kc[3] = {kR, kG, kB};
  for (int c = 0; c < 3; c++) {
    min4[c] = aie::min(aie::min(aie::min(jc[c], gc[c]), ic[c]), kc[c]);
    max4[c] = aie::max(aie::max(aie::max(jc[c], gc[c]), ic[c]), kc[c]);
  }

  fvec aC[3] = {aie::zeros<float, VW>(), aie::zeros<float, VW>(), aie::zeros<float, VW>()};
  fvec aW = aie::zeros<float, VW>();
  // Real loop (not 12 unrolled call sites) -- shrinks the call site at the cost of one
  // small indexed table lookup per iteration; easu_tap_v itself is a single shared function
  // either way (not duplicated), this only affects code size HERE.
  static const float TAP_XY[12][2] = {
      {0.0f, -1.0f}, {1.0f, -1.0f}, {-1.0f, 1.0f}, {0.0f, 1.0f}, {0.0f, 0.0f}, {-1.0f, 0.0f},
      {1.0f, 1.0f}, {2.0f, 1.0f}, {2.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 2.0f}, {0.0f, 2.0f},
  };
  fvec tapR[12] = {bR, cR, iR, jR, fR, eR, kR, lR, hR, gR, oR, nR};
  fvec tapG[12] = {bG, cG, iG, jG, fG, eG, kG, lG, hG, gG, oG, nG};
  fvec tapB[12] = {bB, cB, iB, jB, fB, eB, kB, lB_, hB, gB, oB, nB};
  for (int t = 0; t < 12; t++) {
    fvec tap[3] = {tapR[t], tapG[t], tapB[t]};
    easu_tap_v(aC, aW, TAP_XY[t][0], TAP_XY[t][1], ppx, ppy, dirx, diry, len2x, len2y, lob, clp,
              tap);
  }

  fvec invW = aie::inv(aW);
  for (int c = 0; c < 3; c++)
    out[c] = aie::min(max4[c], aie::max(min4[c], vmul(aC[c], invW)));
}

// EASU only, vectorized (RCAS is a second pass, kept scalar in a SEPARATE kernel/dispatch --
// gamescope itself runs EASU and RCAS as two separate GPU dispatches too, so this mirrors the
// reference pipeline rather than deviating from it; it is also what let the combined kernel fit
// program memory -- see README's .text accounting).
extern "C" {
void fsr1_easu_vec(const float *in_rgb, float *out_rgb) {
  // Deinterleave RGB -> planar (scalar; small crop, not the bottleneck -- see README).
  static float planar[3 * FSR1_IN_H * FSR1_IN_W];
  for (int y = 0; y < FSR1_IN_H; y++)
    for (int x = 0; x < FSR1_IN_W; x++)
      for (int c = 0; c < 3; c++)
        planar[(c * FSR1_IN_H + y) * FSR1_IN_W + x] = in_rgb[(y * FSR1_IN_W + x) * 3 + c];

  for (int py = 0; py < 3; py++) {
    for (int s = 0; s < FSR1_IN_H; s++) {
      int oy = 3 * s + py;
      for (int px = 0; px < 3; px++) {
        fvec out[3];
        easu_phase_row(planar, s, px, py, out);
        alignas(64) float lane[3][VW];
        aie::store_v(lane[0], out[0]);
        aie::store_v(lane[1], out[1]);
        aie::store_v(lane[2], out[2]);
        for (int t = 0; t < VW; t++) {
          int ox = 3 * t + px;
          float *dst = &out_rgb[(oy * FSR1_OUT_W + ox) * 3];
          dst[0] = lane[0][t]; dst[1] = lane[1][t]; dst[2] = lane[2][t];
        }
      }
    }
  }
}
}
