// Key-range split (flash decoding) entry points beside rf_attn_glue_h.cc's: each column attends to
// its own slice of the keys and exports its normalised O with the rows' running max and sum, which a
// merge combines across columns. Layouts as rf_attn_glue_h.cc (16 rows a head, H heads a pass).
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void rf_sm_block(float *s, uint8_t *pc, float *st, uint8_t *w, int32_t tb, int32_t H);
}

namespace {
constexpr unsigned R = 16, T = 8, KEYS = 64, SW = 64;
constexpr unsigned P_H = R * KEYS * 2, CL_H = 2 * R * 4, ST_H = 48, SLICE_E = SW * R;

// A P record's tail after the partial-row masks: each head's running max, R floats.
inline unsigned m_off(unsigned H) { return H * (P_H + CL_H) + 64; }

inline aie::vector<float, 64> row_factors(const float *v) {
  return aie::concat(aie::broadcast<float, 8>(v[0]), aie::broadcast<float, 8>(v[1]), aie::broadcast<float, 8>(v[2]),
                     aie::broadcast<float, 8>(v[3]), aie::broadcast<float, 8>(v[4]), aie::broadcast<float, 8>(v[5]),
                     aie::broadcast<float, 8>(v[6]), aie::broadcast<float, 8>(v[7]));
}
}  // namespace

extern "C" {

#ifdef RF_GLUE_QK
// rf_sm_block, then the heads' running max into the record (the x V core exports it with O).
void rf_sm_block_m(float *__restrict s, uint8_t *__restrict pc, float *__restrict st, uint8_t *w, int32_t tb,
                   int32_t H) {
  rf_sm_block(s, pc, st, w, tb, H);
  for (int32_t h = 0; h < H; ++h)
    aie::store_v(reinterpret_cast<float *>(pc + m_off(H)) + h * R, aie::load_v<16>(st + h * ST_H));
}
#endif

#if defined(RF_GLUE_PV) || defined(RF_GLUE_MERGE)

namespace {
constexpr unsigned UNIT = 1728, OROWS_B = R * 256 * 2, TAIL_B = 192;
constexpr float LOWEST_F = -3.3895313892515355e38f;

// 2^x for x >= -126: floor split, a degree-6 polynomial on the fraction, the integer part into
// the exponent (aie::exp2 is the SFU's linear 2^x, several percent off).
static aie::vector<float, 16> exp2v(aie::vector<float, 16> x) {
  x = aie::max(x, aie::broadcast<float, 16>(-126.0f));
  ::aie::set_rounding(aie::rounding_mode::floor);
  aie::vector<int32_t, 16> n = aie::to_fixed<int32_t>(x, 0);
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  aie::vector<float, 16> f = aie::sub(x, aie::to_float<float>(n, 0));
  constexpr float c[7] = {1.0f, 0.69314718f, 0.24022651f, 0.05550411f, 0.00961813f, 0.00133336f, 0.00015404f};
  aie::vector<float, 16> p = aie::broadcast<float, 16>(c[6]);
  for (int k = 5; k >= 0; --k)
    p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, 16>(c[k]));
  aie::vector<int32_t, 16> e = aie::add(n, aie::broadcast<int32_t, 16>(127));
  aie::vector<int32_t, 16> eb = aie::mul(e, 8388608).to_vector<int32_t>(0);
  return aie::mul(p, eb.cast_to<float>()).to_vector<float>();
}
}  // namespace

#endif

#ifdef RF_GLUE_PV
// The finished O share as bf16 rows [H x R][hb x 64] (rf_pv_finish_a's rounding, no blocking); the
// rows' sum and max leave from the record's tail (l, the masks, m). geo = H | hb << 4.
void rf_pv_finish_rows(const float *__restrict o, const float *__restrict inv, bfloat16 *__restrict ob,
                       int32_t geo) {
  unsigned H = geo & 15, hb = (geo >> 4) & 15, W = hb * SW;
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  for (unsigned h = 0; h < H; ++h)
    for (unsigned sl = 0; sl < hb; ++sl)
      for (unsigned rt = 0; rt < R / T; ++rt) {
        const aie::vector<float, 64> f = row_factors(inv + h * R + rt * T);
        for (unsigned nt = 0; nt < 8; ++nt) {
          aie::vector<bfloat16, 64> b =
              aie::mul(aie::load_v<64>(o + (h * hb + sl) * SLICE_E + (rt * 8 + nt) * 64), f).to_vector<bfloat16>();
          for (unsigned r = 0; r < T; ++r)
            aie::store_v(ob + (h * R + rt * T + r) * W + sl * SW + nt * 8, b.extract<8>(r));
        }
      }
}
// ---- the 8-way merge, on the x V core of one column, one worker's 256 dims. A column's record
// arrives in 1728-B units: its O rows (bf16 [16][256], 8192 B), the tail [l][masks][m] (192 B),
// padding. st = [M 16][L 16]; acc = f32 [16][256] row-major.
void fa_inv_l(float *cl, float *inv);
void rf_merge_init(float *acc, float *st) {
  for (unsigned i = 0; i < R * 256; i += 16)
    aie::store_v(acc + i, aie::zeros<float, 16>());
  aie::store_v(st, aie::broadcast<float, 16>(LOWEST_F));
  aie::store_v(st + R, aie::zeros<float, 16>());
}

void rf_merge_unit(const uint8_t *__restrict unit, bfloat16 *__restrict ob, uint8_t *__restrict tail, int32_t u) {
  uint8_t *o8 = reinterpret_cast<uint8_t *>(ob);
  for (unsigned p = 0; p < UNIT / 64; ++p) {
    unsigned o = u * UNIT + p * 64;
    if (o < OROWS_B)
      aie::store_v(o8 + o, aie::load_v<64>(unit + p * 64));
    else if (o < OROWS_B + TAIL_B)
      aie::store_v(tail + (o - OROWS_B), aie::load_v<64>(unit + p * 64));
  }
}

void rf_merge_col(float *__restrict acc, float *__restrict st, const bfloat16 *__restrict ob,
                  const uint8_t *__restrict tail) {
  const aie::vector<float, 16> lc = aie::load_v<16>(reinterpret_cast<const float *>(tail));
  const aie::vector<float, 16> mc = aie::load_v<16>(reinterpret_cast<const float *>(tail + 128));
  const aie::vector<float, 16> M = aie::load_v<16>(st), L = aie::load_v<16>(st + R);
  // a column with no visible key for a row (l = 0, m = -inf) is selected out, never multiplied
  const auto live = aie::gt(aie::load_v<16>(reinterpret_cast<const int32_t *>(tail)), aie::zeros<int32_t, 16>());
  const aie::vector<float, 16> ms = aie::select(M, mc, live);
  const aie::vector<float, 16> Mn = aie::max(M, ms);
  const aie::vector<float, 16> a = exp2v(aie::sub(M, Mn));
  const aie::vector<float, 16> b = aie::select(aie::zeros<float, 16>(), aie::mul(lc, exp2v(aie::sub(ms, Mn))).to_vector<float>(), live);
  aie::store_v(st, Mn);
  aie::store_v(st + R, aie::add(aie::mul(L, a).to_vector<float>(), b));
  alignas(64) float av[R], bv[R];
  aie::store_v(av, a);
  aie::store_v(bv, b);
  for (unsigned r = 0; r < R; ++r)
    for (unsigned d = 0; d < 256; d += 16) {
      aie::accum<accfloat, 16> x;
      x.from_vector(aie::load_v<16>(ob + r * 256 + d));
      aie::vector<float, 16> t = aie::mul(aie::load_v<16>(acc + r * 256 + d), av[r]).to_vector<float>();
      aie::store_v(acc + r * 256 + d, aie::add(t, aie::mul(x.to_vector<float>(), bv[r]).to_vector<float>()));
    }
}

void rf_merge_finish(const float *__restrict acc, float *__restrict st, bfloat16 *__restrict out) {
  fa_inv_l(st, st);                            // st[r] = 1 / L[r]
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  alignas(64) float iv[R];
  aie::store_v(iv, aie::load_v<16>(st));
  for (unsigned r = 0; r < R; ++r)
    for (unsigned d = 0; d < 256; d += 16)
      aie::store_v(out + r * 256 + d, aie::mul(aie::load_v<16>(acc + r * 256 + d), iv[r]).to_vector<bfloat16>());
}
#endif

#ifdef RF_GLUE_MERGE
// ---- the layer image's merge, on raw O. The 8 columns' tail units come first ([l 16 f32][masks][m
// 16 f32]), then each column's O (the x V worker's f32 accumulator, 10 units, its tile layout: element
// e is in row ((e % 1024) / 512) * 8 + (e % 64) / 8). ws (pc) = [l 8 x 16][m 8 x 16][b 8 x 16][1/L 16].
namespace {
inline unsigned row_of(unsigned e) { return ((e % 1024) / 512) * 8 + (e % 64) / 8; }
}  // namespace

// column j's tail; after the 8th: M = max over live columns, b_j = 2^(m_j - M) (0 where a column sees no
// key), 1/L with L = sum l_j b_j, and acc zeroed
void rf_merge_tails(float *__restrict acc, float *__restrict ws, const uint8_t *__restrict unit, int32_t j) {
  aie::store_v(ws + j * R, aie::load_v<16>(reinterpret_cast<const float *>(unit)));
  aie::store_v(ws + 128 + j * R, aie::load_v<16>(reinterpret_cast<const float *>(unit + 128)));
  if (j != 7)
    return;
  aie::vector<float, 16> M = aie::broadcast<float, 16>(LOWEST_F), L = aie::zeros<float, 16>();
  for (unsigned c = 0; c < 8; ++c) {
    const auto live = aie::gt(aie::load_v<16>(reinterpret_cast<const int32_t *>(ws + c * R)), aie::zeros<int32_t, 16>());
    M = aie::max(M, aie::select(M, aie::load_v<16>(ws + 128 + c * R), live));
  }
  for (unsigned c = 0; c < 8; ++c) {
    const aie::vector<float, 16> l = aie::load_v<16>(ws + c * R);
    const auto live = aie::gt(aie::load_v<16>(reinterpret_cast<const int32_t *>(ws + c * R)), aie::zeros<int32_t, 16>());
    const aie::vector<float, 16> bc = aie::select(aie::zeros<float, 16>(), exp2v(aie::sub(aie::load_v<16>(ws + 128 + c * R), M)), live);
    aie::store_v(ws + 256 + c * R, bc);
    L = aie::add(L, aie::mul(l, bc).to_vector<float>());
  }
  // 1 / L: the SFU reciprocal and one Newton step (the merge cores carry no fa_inv_l)
  const auto nz = aie::gt(L, aie::zeros<float, 16>());
  aie::vector<float, 16> r = aie::inv(aie::select(aie::broadcast<float, 16>(1.0f), L, nz));
  r = aie::mul(r, aie::sub(aie::broadcast<float, 16>(2.0f), aie::mul(L, r).to_vector<float>())).to_vector<float>();
  aie::store_v(ws + 384, aie::select(aie::zeros<float, 16>(), r, nz));
  for (unsigned e = 0; e < R * 256; e += 16)
    aie::store_v(acc + e, aie::zeros<float, 16>());
}

// column j's O unit u (0..9): acc += b_j[row] * O over its floats
void rf_merge_ounit(float *__restrict acc, const float *__restrict ws, const uint8_t *__restrict unit, int32_t j,
                    int32_t u) {
  const float *o = reinterpret_cast<const float *>(unit), *b = ws + 256 + j * R;
  const unsigned e0 = u * (UNIT / 4), e1 = e0 + UNIT / 4 < R * 256 ? e0 + UNIT / 4 : R * 256;
  for (unsigned e = e0; e < e1; e += 8) {
    aie::vector<float, 8> t = aie::mul(aie::load_v<8>(o + (e - e0)), b[row_of(e)]).to_vector<float>();
    aie::store_v(acc + e, aie::add(aie::load_v<8>(acc + e), t));
  }
}

void rf_merge_norm(float *__restrict acc, const float *__restrict ws) {
  for (unsigned e = 0; e < R * 256; e += 8)
    aie::store_v(acc + e, aie::mul(aie::load_v<8>(acc + e), ws[384 + row_of(e)]).to_vector<float>());
}

#endif

#ifdef RF_GLUE_ROW0
// A merged row (one head, this worker's 256 dims, f32 in dim order) as the only row of the worker's O
// tile layout, factor 1, for rf_pv_finish_a.
void rf_o_row0(const uint8_t *__restrict unit, float *__restrict o, float *__restrict inv) {
  const float *v = reinterpret_cast<const float *>(unit);
  for (unsigned e = 0; e < R * 256; e += 16)
    aie::store_v(o + e, aie::zeros<float, 16>());
  for (unsigned k = 0; k < 32; ++k)            // dims 8k .. 8k+7: slice k / 8, column tile k % 8, row 0
    aie::store_v(o + (k / 8) * 1024 + (k % 8) * 64, aie::load_v<8>(v + 8 * k));
  aie::store_v(inv, aie::broadcast<float, 16>(1.0f));
}

#endif
}
