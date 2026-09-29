#!/usr/bin/env python3
"""Isolated int8-key bf16 gather: known keys in, raw bf16 bits and the int8 conversion out.
Separates a gather failure from a bf16->int8 conversion failure. Prints, exits 0."""
import importlib.util
from pathlib import Path
import numpy as np
import bricklib

B = Path(__file__).parent.parent / "conv2d-3x3-u8"
_s = importlib.util.spec_from_file_location("g", B / "golden.py"); g = importlib.util.module_from_spec(_s); _s.loader.exec_module(g)
inc = g.lut_inc(np.arange(256) - 128, bricklib.GEN / "lutiso_place.inc")   # identity, rope placement
keys = np.array(list(range(-32, 32)), np.int8)
body = f'''
#include <aie_api/aie.hpp>
using Look = aie::parallel_lookup<int8, aie::lut<4, bfloat16>>;
alignas(1024) static const uint16_t kAb[512] = {{
#include "{inc}"
}};
alignas(1024) static const uint16_t kCd[512] = {{
#include "{inc}"
}};
extern "C" void lutiso(int8_t *k, int16_t *o) {{
  const aie::lut<4, bfloat16> t(256, (const bfloat16 *)kAb, (const bfloat16 *)kCd);
  Look look(t, 0, 128);
  aie::vector<int8, 64> kv = aie::load_v<64>(k);
  for (int gi = 0; gi < 4; gi++) {{
    // A: 16 keys extracted from a 64-lane vector
    aie::vector<int8, 16> ka = kv.extract<16>(gi);
    aie::store_v(o + gi * 16, aie::vector<int16, 16>(aie::unpack(aie::to_fixed<int8>(look.fetch(ka), 0))));
    // B: 16 keys loaded directly
    aie::vector<int8, 16> kb = aie::load_v<16>(k + gi * 16);
    aie::store_v(o + 64 + gi * 16, aie::vector<int16, 16>(aie::unpack(aie::to_fixed<int8>(look.fetch(kb), 0))));
    // C: 16 keys rebuilt through float, as rope-lut builds them
    aie::vector<float, 16> kf = aie::to_float(aie::vector<int32, 16>(aie::unpack(aie::unpack(kb))), 0);
    aie::vector<int8, 16> kc = aie::to_fixed<int8>(kf, 0);
    aie::store_v(o + 128 + gi * 16, aie::vector<int16, 16>(aie::unpack(aie::to_fixed<int8>(look.fetch(kc), 0))));
  }}
}}
'''
shim = bricklib.GEN / "lutiso_shim.cc"
shim.write_text(body)
design = bricklib._build_oneshot("lutiso", shim, [64], 192, [np.int8], np.int16, [], stack_size=4096)
import aie.iron as iron
kt = iron.tensor(keys, dtype=np.int8, device="npu")
ot = iron.zeros((192,), dtype=np.int16, device="npu")
design(kt, ot)
o = ot.numpy().astype(np.int64)
bits = (o[:64] & 0xffff).astype(np.uint32) << 16
print("keys          :", keys.tolist())
print("A extract     :", o[:64].tolist())
print("B load16      :", o[64:128].tolist())
print("C via float   :", o[128:192].tolist())
