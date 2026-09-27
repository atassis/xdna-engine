#!/usr/bin/env python3
"""Isolated int8-key int8-VALUE gather: aie::lut<4, int8> (ValueWords=1, same load_lut_2x_int8
family as bfloat16's ValueWords=2 path) -- does it return the value directly, skipping the
to_fixed<int8> conversion every SiLU/gate epilogue does today after a bf16 fetch? Table = identity
(v(k)=k) so any gather error is visible directly. Prints, exits 0."""
from pathlib import Path
import numpy as np
import bricklib

table8 = (np.arange(256) - 128).astype(np.int8)


def pack_lut8(t):
    """Same physical placement as golden.pack_lut, int8 values."""
    t = np.asarray(t, np.int8)
    phys = np.zeros(512, np.int8)
    for j, v in enumerate(t):
        s0 = 16 * (j // 16) + (j % 16) // 2
        for sl in (s0, s0 + 8):
            phys[2 * sl + (j % 2)] = v
    return phys


w8 = pack_lut8(table8)
inc_path = bricklib.GEN / "lutiso8_place.inc"
inc_path.write_text(",\n".join(", ".join(str(int(x)) for x in w8[i:i + 8])
                               for i in range(0, 512, 8)) + "\n")

keys = np.array(list(range(-32, 32)), np.int8)
body = f'''
#include <aie_api/aie.hpp>
using Look = aie::parallel_lookup<int8, aie::lut<4, int8>>;
alignas(1024) static const int8_t kAb[512] = {{
#include "{inc_path}"
}};
alignas(1024) static const int8_t kCd[512] = {{
#include "{inc_path}"
}};
extern "C" void lutiso8(int8_t *k, int8_t *o) {{
  const aie::lut<4, int8> t(256, (const int8 *)kAb, (const int8 *)kCd);
  Look look(t, 0, 128);
  for (int gi = 0; gi < 4; gi++) {{
    aie::vector<int8, 16> kb = aie::load_v<16>(k + gi * 16);
    aie::store_v(o + gi * 16, look.fetch(kb));
  }}
}}
'''
shim = bricklib.GEN / "lutiso8_shim.cc"
shim.write_text(body)
design = bricklib._build_oneshot("lutiso8", shim, [64], 64, [np.int8], np.int8, [], stack_size=4096)
import aie.iron as iron
kt = iron.tensor(keys, dtype=np.int8, device="npu")
ot = iron.zeros((64,), dtype=np.int8, device="npu")
design(kt, ot)
o = ot.numpy().astype(np.int64)
expect = keys.astype(np.int64)
match = (o == expect)
print("keys  :", keys.tolist())
print("got   :", o.tolist())
print("expect:", expect.tolist())
print(f"match: {match.sum()}/{match.size}")
if not match.all():
    bad = np.where(~match)[0]
    print(f"FIRST MISMATCH at index {bad[0]}: key={keys[bad[0]]} got={o[bad[0]]} expect={expect[bad[0]]}")
