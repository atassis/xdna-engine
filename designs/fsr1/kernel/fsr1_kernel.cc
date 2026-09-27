// FSR1 EASU+RCAS, fixed x3 scale, one small RGB crop resident in L1.
// Scalar fp32, no aie_api vector ops -- a correctness-first port; see designs/fsr1/README.md
// for the brick-first vectorization levers this leaves on the table.
//
// Mirrors designs/fsr1/cpu_ref.py line for line (which mirrors ffx_fsr1.h/ffx_a.h). Shape is
// fixed at compile time via FSR1_IN_W/FSR1_IN_H (output is always 3x).
#include <stdint.h>
#include <string.h>

#ifndef FSR1_IN_W
#define FSR1_IN_W 8
#endif
#ifndef FSR1_IN_H
#define FSR1_IN_H 8
#endif
#define FSR1_OUT_W (FSR1_IN_W * 3)
#define FSR1_OUT_H (FSR1_IN_H * 3)

static inline uint32_t f2u(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
static inline float u2f(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// APrxLoRcpF1 / APrxMedRcpF1 / APrxLoRsqF1, ffx_a.h bit-trick approximations.
static inline float aprx_lo_rcp(float a) { return u2f(0x7EF07EBBu - f2u(a)); }
static inline float aprx_med_rcp(float a) {
  float b = u2f(0x7EF19FFFu - f2u(a));
  return b * (-b * a + 2.0f);
}
static inline float aprx_lo_rsq(float a) { return u2f(0x5F347D74u - (f2u(a) >> 1)); }
static inline float satf(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }
static inline float lumaf(float r, float g, float b) { return b * 0.5f + (r * 0.5f + g); }

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// textureGather-equivalent 2x2 tap at (gx,gy) in pixel space, clamp-to-edge. tap[k][c],
// k in {bl,br,tr,tl} order (matches ffx_fsr1.h's gather component convention, see cpu_ref.py).
static void gather4(const float *img, int w, int h, float gx, float gy, float tap[4][3]) {
  // floorf-equivalent without <math.h> (no libm dependency in the kernel).
  float fx = gx - 0.5f, fy = gy - 0.5f;
  int x0 = (fx >= 0.0f) ? (int)fx : (int)fx - (fx == (int)fx ? 0 : 1);
  int y0 = (fy >= 0.0f) ? (int)fy : (int)fy - (fy == (int)fy ? 0 : 1);
  int x0c = clampi(x0, 0, w - 1), x1c = clampi(x0 + 1, 0, w - 1);
  int y0c = clampi(y0, 0, h - 1), y1c = clampi(y0 + 1, 0, h - 1);
  for (int c = 0; c < 3; c++) {
    tap[0][c] = img[(y1c * w + x0c) * 3 + c]; // bl
    tap[1][c] = img[(y1c * w + x1c) * 3 + c]; // br
    tap[2][c] = img[(y0c * w + x1c) * 3 + c]; // tr
    tap[3][c] = img[(y0c * w + x0c) * 3 + c]; // tl
  }
}

static void easu_set(float *dirx, float *diry, float *len, float ppx, float ppy, int mask,
                     float lA, float lB, float lC, float lD, float lE) {
  float w;
  if (mask == 0) w = (1.0f - ppx) * (1.0f - ppy);
  else if (mask == 1) w = ppx * (1.0f - ppy);
  else if (mask == 2) w = (1.0f - ppx) * ppy;
  else w = ppx * ppy;

  float dc = lD - lC, cb = lC - lB;
  float lenX = dc < 0 ? -dc : dc;
  float cbA = cb < 0 ? -cb : cb;
  lenX = lenX > cbA ? lenX : cbA;
  lenX = aprx_lo_rcp(lenX);
  float dirX = lD - lB;
  *dirx += dirX * w;
  float adx = dirX < 0 ? -dirX : dirX;
  lenX = satf(adx * lenX);
  lenX *= lenX;
  *len += lenX * w;

  float ec = lE - lC, ca = lC - lA;
  float lenY = ec < 0 ? -ec : ec;
  float caA = ca < 0 ? -ca : ca;
  lenY = lenY > caA ? lenY : caA;
  lenY = aprx_lo_rcp(lenY);
  float dirY = lE - lA;
  *diry += dirY * w;
  float ady = dirY < 0 ? -dirY : dirY;
  lenY = satf(ady * lenY);
  lenY *= lenY;
  *len += lenY * w;
}

static void easu_tap(float aC[3], float *aW, float offx, float offy, float dirx, float diry,
                     float len2x, float len2y, float lob, float clp, const float c[3]) {
  float vx = offx * dirx + offy * diry;
  float vy = offx * (-diry) + offy * dirx;
  vx *= len2x;
  vy *= len2y;
  float d2 = vx * vx + vy * vy;
  d2 = d2 < clp ? d2 : clp;
  float wB = (2.0f / 5.0f) * d2 - 1.0f;
  float wA = lob * d2 - 1.0f;
  wB *= wB;
  wA *= wA;
  wB = (25.0f / 16.0f) * wB - (25.0f / 16.0f - 1.0f);
  float w = wB * wA;
  aC[0] += c[0] * w;
  aC[1] += c[1] * w;
  aC[2] += c[2] * w;
  *aW += w;
}

// One output pixel of EASU. img is the resident IN_W x IN_H x 3 crop.
static void easu_pixel(const float *img, int ox, int oy, float out[3]) {
  const float iw = (float)FSR1_IN_W, ih = (float)FSR1_IN_H;
  const float ow = (float)FSR1_OUT_W, oh = (float)FSR1_OUT_H;
  float ppx = (float)ox * (iw / ow) + (0.5f * iw / ow - 0.5f);
  float ppy = (float)oy * (ih / oh) + (0.5f * ih / oh - 0.5f);
  float fpx = (ppx >= 0.0f) ? (float)(int)ppx : (float)((int)ppx - (ppx == (int)ppx ? 0 : 1));
  float fpy = (ppy >= 0.0f) ? (float)(int)ppy : (float)((int)ppy - (ppy == (int)ppy ? 0 : 1));
  ppx -= fpx;
  ppy -= fpy;

  // p0..p3 gather centers in pixel space, derived from FsrEasuCon's con1/con2/con3 offsets
  // (fpx,fpy is the 'f' tap's own position -- see cpu_ref.py / ffx_fsr1.h comment).
  float p0x = fpx + 1.0f, p0y = fpy - 1.0f;
  float p1x = p0x - 1.0f, p1y = p0y + 2.0f;
  float p2x = p0x + 1.0f, p2y = p0y + 2.0f;
  float p3x = p0x, p3y = p0y + 4.0f;

  float bczz[4][3], ijfe[4][3], klhg[4][3], zzon[4][3];
  gather4(img, FSR1_IN_W, FSR1_IN_H, p0x, p0y, bczz);
  gather4(img, FSR1_IN_W, FSR1_IN_H, p1x, p1y, ijfe);
  gather4(img, FSR1_IN_W, FSR1_IN_H, p2x, p2y, klhg);
  gather4(img, FSR1_IN_W, FSR1_IN_H, p3x, p3y, zzon);

  const float *b_ = bczz[0], *c_ = bczz[1];
  const float *i_ = ijfe[0], *j_ = ijfe[1], *f_ = ijfe[2], *e_ = ijfe[3];
  const float *k_ = klhg[0], *l_ = klhg[1], *h_ = klhg[2], *g_ = klhg[3];
  const float *o_ = zzon[2], *n_ = zzon[3];

  float bL = lumaf(b_[0], b_[1], b_[2]), cL = lumaf(c_[0], c_[1], c_[2]);
  float iL = lumaf(i_[0], i_[1], i_[2]), jL = lumaf(j_[0], j_[1], j_[2]);
  float fL = lumaf(f_[0], f_[1], f_[2]), eL = lumaf(e_[0], e_[1], e_[2]);
  float kL = lumaf(k_[0], k_[1], k_[2]), lL = lumaf(l_[0], l_[1], l_[2]);
  float hL = lumaf(h_[0], h_[1], h_[2]), gL = lumaf(g_[0], g_[1], g_[2]);
  float oL = lumaf(o_[0], o_[1], o_[2]), nL = lumaf(n_[0], n_[1], n_[2]);

  float dirx = 0.0f, diry = 0.0f, len = 0.0f;
  easu_set(&dirx, &diry, &len, ppx, ppy, 0, bL, eL, fL, gL, jL);
  easu_set(&dirx, &diry, &len, ppx, ppy, 1, cL, fL, gL, hL, kL);
  easu_set(&dirx, &diry, &len, ppx, ppy, 2, fL, iL, jL, kL, nL);
  easu_set(&dirx, &diry, &len, ppx, ppy, 3, gL, jL, kL, lL, oL);

  float dirR = dirx * dirx + diry * diry;
  int zro = dirR < (1.0f / 32768.0f);
  float rdirR = aprx_lo_rsq(dirR);
  rdirR = zro ? 1.0f : rdirR;
  dirx = zro ? 1.0f : dirx;
  dirx *= rdirR;
  diry *= rdirR;

  len *= 0.5f;
  len *= len;
  float adx = dirx < 0 ? -dirx : dirx, ady = diry < 0 ? -diry : diry;
  float mx = adx > ady ? adx : ady;
  float stretch = (dirx * dirx + diry * diry) * aprx_lo_rcp(mx);
  float len2x = 1.0f + (stretch - 1.0f) * len;
  float len2y = 1.0f + (-0.5f) * len;
  float lob = 0.5f + ((1.0f / 4.0f - 0.04f) - 0.5f) * len;
  float clp = aprx_lo_rcp(lob);

  float min4[3], max4[3];
  for (int c = 0; c < 3; c++) {
    float v0 = j_[c], v1 = g_[c], v2 = i_[c], v3 = k_[c];
    float mn = v0 < v1 ? v0 : v1;
    mn = mn < v2 ? mn : v2;
    mn = mn < v3 ? mn : v3;
    float mx2 = v0 > v1 ? v0 : v1;
    mx2 = mx2 > v2 ? mx2 : v2;
    mx2 = mx2 > v3 ? mx2 : v3;
    min4[c] = mn;
    max4[c] = mx2;
  }

  float aC[3] = {0.0f, 0.0f, 0.0f}, aW = 0.0f;
  struct { float ox, oy; const float *c; } taps[12] = {
      {0.0f, -1.0f, b_}, {1.0f, -1.0f, c_}, {-1.0f, 1.0f, i_}, {0.0f, 1.0f, j_},
      {0.0f, 0.0f, f_}, {-1.0f, 0.0f, e_}, {1.0f, 1.0f, k_}, {2.0f, 1.0f, l_},
      {2.0f, 0.0f, h_}, {1.0f, 0.0f, g_}, {1.0f, 2.0f, o_}, {0.0f, 2.0f, n_},
  };
  for (int t = 0; t < 12; t++) {
    easu_tap(aC, &aW, taps[t].ox - ppx, taps[t].oy - ppy, dirx, diry, len2x, len2y, lob, clp,
             taps[t].c);
  }
  float invW = 1.0f / aW; // FSR1 uses the exact rcp here, not an approximation.
  for (int c = 0; c < 3; c++) {
    float v = aC[c] * invW;
    v = v < min4[c] ? min4[c] : v;
    v = v > max4[c] ? max4[c] : v;
    out[c] = v;
  }
}

static const float FSR_RCAS_LIMIT = 0.25f - 1.0f / 16.0f;

static void rcas_pixel(const float *easu, int w, int h, int ox, int oy, float con,
                       float out[3]) {
  auto load = [&](int dy, int dx, float v[3]) {
    int yy = clampi(oy + dy, 0, h - 1), xx = clampi(ox + dx, 0, w - 1);
    for (int c = 0; c < 3; c++) v[c] = easu[(yy * w + xx) * 3 + c];
  };
  float b[3], d[3], e[3], f[3], hh[3];
  load(-1, 0, b);
  load(0, -1, d);
  load(0, 0, e);
  load(0, 1, f);
  load(1, 0, hh);

  float mn4[3], mx4[3];
  for (int c = 0; c < 3; c++) {
    float mn = b[c] < d[c] ? b[c] : d[c];
    mn = mn < f[c] ? mn : f[c];
    mn = mn < hh[c] ? mn : hh[c];
    float mx = b[c] > d[c] ? b[c] : d[c];
    mx = mx > f[c] ? mx : f[c];
    mx = mx > hh[c] ? mx : hh[c];
    mn4[c] = mn;
    mx4[c] = mx;
  }
  float lobe_c[3];
  for (int c = 0; c < 3; c++) {
    float hitMin = (mn4[c] < e[c] ? mn4[c] : e[c]) / (4.0f * mx4[c]);
    float hitMax = (1.0f - (mx4[c] > e[c] ? mx4[c] : e[c])) / (4.0f * mn4[c] - 4.0f);
    lobe_c[c] = (-hitMin) > hitMax ? -hitMin : hitMax;
  }
  float lobe = lobe_c[0] > lobe_c[1] ? lobe_c[0] : lobe_c[1];
  lobe = lobe > lobe_c[2] ? lobe : lobe_c[2];
  lobe = lobe < 0.0f ? lobe : 0.0f;
  lobe = lobe > (-FSR_RCAS_LIMIT) ? lobe : (-FSR_RCAS_LIMIT);
  lobe *= con;
  float rcpL = aprx_med_rcp(4.0f * lobe + 1.0f);
  for (int c = 0; c < 3; c++)
    out[c] = (lobe * b[c] + lobe * d[c] + lobe * hh[c] + lobe * f[c] + e[c]) * rcpL;
}

extern "C" {
void fsr1_strip(const float *in_rgb, float *out_rgb) {
  static float easu_buf[FSR1_OUT_H * FSR1_OUT_W * 3];
  const float sharpness = 0.2f;
  // FsrRcasCon: con = exp2(-sharpness). No exp2 intrinsic assumed available here;
  // sharpness is fixed at compile time so this is a compile-time-foldable constant.
  const float con = 0.87055056329f; // 2^-0.2, matches cpu_ref.py's fsr_rcas_con(0.2)

  for (int oy = 0; oy < FSR1_OUT_H; oy++)
    for (int ox = 0; ox < FSR1_OUT_W; ox++)
      easu_pixel(in_rgb, ox, oy, &easu_buf[(oy * FSR1_OUT_W + ox) * 3]);

  for (int oy = 0; oy < FSR1_OUT_H; oy++)
    for (int ox = 0; ox < FSR1_OUT_W; ox++)
      rcas_pixel(easu_buf, FSR1_OUT_W, FSR1_OUT_H, ox, oy, con,
                &out_rgb[(oy * FSR1_OUT_W + ox) * 3]);
}
}

extern "C" {
void fsr1_easu_only(const float *in_rgb, float *out_rgb) {
  for (int oy = 0; oy < FSR1_OUT_H; oy++)
    for (int ox = 0; ox < FSR1_OUT_W; ox++)
      easu_pixel(in_rgb, ox, oy, &out_rgb[(oy * FSR1_OUT_W + ox) * 3]);
}
}
