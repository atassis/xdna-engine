#!/usr/bin/env python3
"""Isolated int8-key int32-VALUE gather: does aie::lut<4, int32> (a value type parallel_lookup's
fetch() supports generically, per detail/aie2/parallel_lookup.hpp's ValueWords=4 branch --
load_lut_2x_int32) actually gather correct int32 values on device, or is bf16 the only value
type that works (as every kernel here assumes)? Keys loaded from memory (the known-good path
from probe_lut_gather_isolated.py; register-extract of keys is NOT tested here, that failure is
orthogonal to the value type). Table = identity*1000 so a wrong gather is obviously wrong, not
off-by-a-rounding-ulp. Prints, exits 0."""
import importlib.util
from pathlib import Path
import numpy as np
import bricklib

B = Path(__file__).parent.parent / "conv2d-3x3-u8"
_s = importlib.util.spec_from_file_location("g", B / "golden.py"); g = importlib.util.module_from_spec(_s); _s.loader.exec_module(g)

# int32 table: value(k) = (k) * 1000, k in [-128, 127] -- large enough that bf16 could NOT hold it
# exactly (the thing being tested), small enough int32 arithmetic never saturates.
table32 = ((np.arange(256) - 128) * 1000).astype(np.int32)


def pack_lut32(t):
    """Same physical placement as golden.pack_lut, but int32 values, no bf16 cast."""
    t = np.asarray(t, np.int32)
    phys = np.zeros(512, np.int32)
    for j, v in enumerate(t):
        s0 = 16 * (j // 16) + (j % 16) // 2
        for sl in (s0, s0 + 8):
            phys[2 * sl + (j % 2)] = v
    return phys


w32 = pack_lut32(table32)
inc_path = bricklib.GEN / "lutiso32_place.inc"
inc_path.write_text(",\n".join(", ".join(f"0x{int(x) & 0xffffffff:08x}" for x in w32[i:i + 8])
                               for i in range(0, 512, 8)) + "\n")

keys = np.array(list(range(-32, 32)), np.int8)
body = f'''
#include <aie_api/aie.hpp>
using Look = aie::parallel_lookup<int8, aie::lut<4, int32>>;
alignas(1024) static const uint32_t kAb[512] = {{
#include "{inc_path}"
}};
alignas(1024) static const uint32_t kCd[512] = {{
#include "{inc_path}"
}};
extern "C" void lutiso32(int8_t *k, int32_t *o) {{
  const aie::lut<4, int32> t(256, (const int32 *)kAb, (const int32 *)kCd);
  Look look(t, 0, 128);
  for (int gi = 0; gi < 4; gi++) {{
    // keys loaded from memory (the known-good pattern; register extract is untested here)
    aie::vector<int8, 16> kb = aie::load_v<16>(k + gi * 16);
    aie::store_v(o + gi * 16, look.fetch(kb));
  }}
}}
'''
shim = bricklib.GEN / "lutiso32_shim.cc"
shim.write_text(body)
design = bricklib._build_oneshot("lutiso32", shim, [64], 64, [np.int8], np.int32, [], stack_size=4096)
import aie.iron as iron
kt = iron.tensor(keys, dtype=np.int8, device="npu")
ot = iron.zeros((64,), dtype=np.int32, device="npu")
design(kt, ot)
o = ot.numpy().astype(np.int64)
expect = (keys.astype(np.int64)) * 1000
match = (o == expect)
print("keys  :", keys.tolist())
print("got   :", o.tolist())
print("expect:", expect.tolist())
print(f"match: {match.sum()}/{match.size}")
if not match.all():
    bad = np.where(~match)[0]
    print(f"FIRST MISMATCH at index {bad[0]}: key={keys[bad[0]]} got={o[bad[0]]} expect={expect[bad[0]]}")
