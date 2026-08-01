#!/usr/bin/env python3
"""Bisect from the working read-modify-write up to the real rope kernel.

probe_rope_rmw established a KNOWN-GOOD baseline with the harness's exact three-buffer signature:
copy, then a per-row 16-lane load/store identity over `qk_out + m*D`, 2048/2048 exact, with or without
the parallel_lookup objects constructed. The real kernel, same shape, zeroes every row after the first.

So the fault is in what the real kernel adds. Add it back one layer at a time; the first arm that
breaks names the layer.

  A  rmw            load/store identity                                  (known good)
  B  rot0           FULL rotation arithmetic with keys forced to ZERO.
                    keys=0 -> sin=0, cos=1 -> out1 = x1*1 - x2*0 = x1 and out2 = x2, so this is
                    still the identity, but it exercises the fetch and every accum/mul/sub.
  C  rot_pos0       adds the pos/theta/wrap/quantize chain with pos=0, which must also yield keys=0.

A passes and B fails  -> the fetch or the rotation arithmetic.
A,B pass and C fails  -> the angle/quantize chain, i.e. the per-row scalar `pos[m]` read.
All pass             -> the fault needs something only the real kernel still has.
"""
import pathlib

import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot

M, D = 16, 128
N = M * D
KVEC = 16
HALF = D // 2
CBUF_N = M + HALF
TAB = pathlib.Path("route_b_kernels/bricks/rope-lut/rope_lut_tables.inc").resolve()

PRE = f'''#include <aie_api/aie.hpp>
#include <stdint.h>
#include "{TAB}"
constexpr unsigned kVec = {KVEC};
constexpr unsigned kRotHalf = {HALF};
constexpr float kPi = 3.14159265358979323846f;
'''

ROT_BODY = '''
      ::aie::accum<accfloat, kVec> x1a; x1a.from_vector(::aie::load_v<kVec>(row + i), 0);
      ::aie::accum<accfloat, kVec> x2a; x2a.from_vector(::aie::load_v<kVec>(row + kRotHalf + i), 0);
      ::aie::accum<accfloat, kVec> sina; sina.from_vector(sv, 0);
      ::aie::accum<accfloat, kVec> cosa; cosa.from_vector(cv, 0);
      ::aie::vector<float, kVec> x1 = x1a.to_vector<float>();
      ::aie::vector<float, kVec> x2 = x2a.to_vector<float>();
      ::aie::vector<float, kVec> s = sina.to_vector<float>();
      ::aie::vector<float, kVec> c = cosa.to_vector<float>();
      ::aie::vector<float, kVec> out1 = ::aie::sub(::aie::mul(x1, c).template to_vector<float>(),
                                                   ::aie::mul(x2, s).template to_vector<float>());
      ::aie::vector<float, kVec> out2 = ::aie::add(::aie::mul(x1, s).template to_vector<float>(),
                                                   ::aie::mul(x2, c).template to_vector<float>());
      ::aie::accum<accfloat, kVec> o1; o1.from_vector(out1, 0);
      ::aie::accum<accfloat, kVec> o2; o2.from_vector(out2, 0);
      ::aie::store_v(row + i, o1.template to_vector<bfloat16>());
      ::aie::store_v(row + kRotHalf + i, o2.template to_vector<bfloat16>());
'''

KEYS_ZERO = "      ::aie::vector<int8, kVec> keys = ::aie::zeros<int8, kVec>();\n"

KEYS_POS = '''      ::aie::vector<float, kVec> invf = ::aie::load_v<kVec>(inv_freq + i);
      ::aie::vector<float, kVec> theta = ::aie::mul(posf, invf);
      ::aie::vector<float, kVec> kwf = ::aie::mul(theta, 1.0f / (2.0f * kPi));
      ::aie::vector<int32, kVec> k = ::aie::to_fixed<int32>(kwf);
      ::aie::vector<float, kVec> kf = ::aie::to_float(k);
      ::aie::vector<float, kVec> ktwopi = ::aie::mul(kf, 2.0f * kPi);
      ::aie::vector<float, kVec> wrapped = ::aie::sub(theta, ktwopi);
      ::aie::vector<float, kVec> q = ::aie::mul(wrapped, 128.0f / kPi);
      ::aie::vector<int8, kVec> keys = ::aie::to_fixed<int8>(q);
'''


def body_loop(arm):
    """The per-row work, identical text for the inline and the separate-function arms."""
    s = f'  for (unsigned m = 0; m < {M}u; ++m) {{\n    bfloat16 *row = qk + (size_t)m * {D}u;\n'
    s += '''    const int32_t p = pos[m];
    ::aie::vector<float, kVec> posf =
        ::aie::mul(::aie::to_float(::aie::broadcast<int32, kVec>(p)), 1.0f);
'''
    s += "    for (unsigned i = 0; i < kRotHalf; i += kVec) {\n"
    s += KEYS_POS
    s += '''      ::aie::vector<bfloat16, kVec> sv = sin_look.fetch(keys);
      ::aie::vector<bfloat16, kVec> cv = cos_look.fetch(keys);
'''
    return s + ROT_BODY + "    }\n  }\n"


def shim_fn(name):
    """Arm D: arm C's body moved into a SEPARATE extern "C" function, called from the entry
    point -- exactly rope_lut.cc's structure. Everything else is byte-identical to arm C."""
    return PRE + f'''extern "C" void {name}_body(bfloat16 *restrict qk, const int32_t *restrict pos,
                                const float *restrict inv_freq) {{
  const ::aie::lut<4, bfloat16> sin_lut(256, kSinLutAb, kSinLutCd);
  const ::aie::lut<4, bfloat16> cos_lut(256, kCosLutAb, kCosLutCd);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> sin_look(sin_lut, 0, 128);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> cos_look(cos_lut, 0, 128);
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
{body_loop("rot_pos0")}}}

extern "C" void {name}(bfloat16 *restrict qk_in, int32_t *restrict cbuf,
                     bfloat16 *restrict qk_out) {{
  for (unsigned i = 0; i < {N}u; ++i) qk_out[i] = qk_in[i];
  const int32_t *pos = cbuf;
  const float *inv_freq = (const float *)(cbuf + {M});
  {name}_body(qk_out, pos, inv_freq);
}}
'''


def shim(name, arm):
    if arm == "fn":
        return shim_fn(name)
    s = PRE + f'''extern "C" void {name}(bfloat16 *restrict qk_in, int32_t *restrict cbuf,
                     bfloat16 *restrict qk_out) {{
  for (unsigned i = 0; i < {N}u; ++i) qk_out[i] = qk_in[i];
  const ::aie::lut<4, bfloat16> sin_lut(256, kSinLutAb, kSinLutCd);
  const ::aie::lut<4, bfloat16> cos_lut(256, kCosLutAb, kCosLutCd);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> sin_look(sin_lut, 0, 128);
  ::aie::parallel_lookup<int8, ::aie::lut<4, bfloat16>> cos_look(cos_lut, 0, 128);
  ::aie::set_rounding(::aie::rounding_mode::conv_even);
  const int32_t *pos = cbuf;
  const float *inv_freq = (const float *)(cbuf + {M});
  (void)pos; (void)inv_freq;
  for (unsigned m = 0; m < {M}u; ++m) {{
    bfloat16 *row = qk_out + (size_t)m * {D}u;
'''
    if arm == "rmw":
        s += f'''    for (unsigned i = 0; i < {D}u; i += kVec)
      ::aie::store_v(row + i, ::aie::load_v<kVec>(row + i));
'''
    else:
        if arm == "rot_pos0":
            s += '''    const int32_t p = pos[m];
    ::aie::vector<float, kVec> posf =
        ::aie::mul(::aie::to_float(::aie::broadcast<int32, kVec>(p)), 1.0f);
'''
        s += "    for (unsigned i = 0; i < kRotHalf; i += kVec) {\n"
        s += KEYS_ZERO if arm == "rot0" else KEYS_POS
        s += '''      ::aie::vector<bfloat16, kVec> sv = sin_look.fetch(keys);
      ::aie::vector<bfloat16, kVec> cv = cos_look.fetch(keys);
'''
        s += ROT_BODY + "    }\n"
    return s + "  }\n}\n"


def run(label, name, arm, src, want, cbuf):
    p = GEN / f"{name}_shim.cc"
    p.write_text(shim(name, arm))
    d = _build_oneshot(name, p, [N, CBUF_N], N,
                       [ml_dtypes.bfloat16, np.int32], ml_dtypes.bfloat16, [])
    it = iron.tensor(np.ascontiguousarray(src), dtype=ml_dtypes.bfloat16, device="npu")
    cb = iron.tensor(np.ascontiguousarray(cbuf), dtype=np.int32, device="npu")
    ot = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
    d(it, cb, ot)
    got = ot.numpy().astype(np.float32).reshape(M, D)
    bad = got != want
    rows = sorted(set(np.nonzero(bad)[0].tolist()))
    print(f"{label}: exact {N - int(bad.sum())}/{N}   damaged rows {rows if rows else 'NONE'}")
    if rows:
        print(f"    row {rows[0]} got[:6] {got[rows[0]][:6].tolist()}")


def main():
    rng = np.random.default_rng(0)
    src = rng.standard_normal(N).astype(np.float32).astype(ml_dtypes.bfloat16)
    want = src.astype(np.float32).reshape(M, D)
    inv_freq = (1.0 / (10000.0 ** (np.arange(0, HALF) * 2.0 / D))).astype(np.float32)
    cbuf = np.concatenate([np.zeros(M, np.int32), inv_freq.view(np.int32)]).astype(np.int32)
    print("every arm must be the identity: output == input\n")
    run("A rmw     ", "bis_rmw", "rmw", src, want, cbuf)
    run("B rot0    ", "bis_rot0", "rot0", src, want, cbuf)
    run("C rot_pos0", "bis_rotpos", "rot_pos0", src, want, cbuf)
    run("D separate", "bis_fn", "fn", src, want, cbuf)


if __name__ == "__main__":
    main()
