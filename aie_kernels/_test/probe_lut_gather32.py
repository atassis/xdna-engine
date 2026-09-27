#!/usr/bin/env python3
"""Phase 1i: is aie_api's OWN Vec::size()==32 fetch() path (parallel_lookup.hpp's aie2 backend,
which internally does two extract<16>() from a 32-lane accumulator and combines) usable directly
on a 32-lane LOAD, in place of conv3x3_u8.cc's current 2x(load_v<16>+fetch+store) loop? If so,
apply_lut_inplace/apply_lut16/the gate epilogue can halve their fetch-call count. This is a
DIFFERENT code path from probe_lut_gather_isolated.py's method A (extract<16> from a 64-lane
REGISTER, confirmed broken on device) -- fetch()'s internal extract is from a 32-lane accumulator
built via acc.from_vector(), the API's own tested/supported size, not an ad hoc register slice.
Verify bit-exact against the known-good 16-lane-loop path before touching conv3x3_u8.cc.
Prints, exits 0."""
import importlib.util
from pathlib import Path
import numpy as np
import bricklib

B = Path(__file__).parent.parent / "conv2d-3x3-u8"
_s = importlib.util.spec_from_file_location("g", B / "golden.py"); g = importlib.util.module_from_spec(_s); _s.loader.exec_module(g)
inc = g.lut_inc(np.arange(256) - 128, bricklib.GEN / "lutgather32_place.inc")   # identity table
rng = np.random.default_rng(3)
keys = rng.integers(-128, 128, size=64).astype(np.int8)
body = f'''
#include <aie_api/aie.hpp>
using Look = aie::parallel_lookup<int8, aie::lut<4, bfloat16>>;
alignas(1024) static const uint16_t kAb[512] = {{
#include "{inc}"
}};
alignas(1024) static const uint16_t kCd[512] = {{
#include "{inc}"
}};
extern "C" void lutgather32(int8_t *k, int16_t *o) {{
  const aie::lut<4, bfloat16> t(256, (const bfloat16 *)kAb, (const bfloat16 *)kCd);
  Look look(t, 0, 128);
  // Baseline (known-good): 4x (16-lane load_v + fetch + store)
  for (int g = 0; g < 64; g += 16) {{
    aie::vector<int8, 16> kv = aie::load_v<16>(k + g);
    aie::store_v(o + g, aie::vector<int16, 16>(aie::unpack(aie::to_fixed<int8>(look.fetch(kv), 0))));
  }}
  // Candidate: 2x (32-lane load_v + fetch + store), one fetch() call covering 32 keys
  for (int g = 0; g < 64; g += 32) {{
    aie::vector<int8, 32> kv = aie::load_v<32>(k + g);
    aie::vector<int8, 32> r = aie::to_fixed<int8>(look.fetch(kv), 0);
    for (int gi = 0; gi < 32; gi += 16)
      aie::store_v(o + 64 + g + gi, aie::vector<int16, 16>(aie::unpack(r.extract<16>(gi / 16))));
  }}
}}
'''
shim = bricklib.GEN / "lutgather32_shim.cc"
shim.write_text(body)
design = bricklib._build_oneshot("lutgather32", shim, [64], 192, [np.int8], np.int16, [], stack_size=4096)
import aie.iron as iron
kt = iron.tensor(keys, dtype=np.int8, device="npu")
ot = iron.zeros((192,), dtype=np.int16, device="npu")
design(kt, ot)
o = ot.numpy().astype(np.int64)
base, cand = o[:64], o[64:128]
match = int((base == cand).sum())
print("keys      :", keys.tolist())
print("baseline  :", base.tolist())
print("candidate :", cand.tolist())
print(f"match {match} / 64")
if match != 64:
    m = base != cand
    print("  mismatch idx:", np.where(m)[0].tolist())
    print("  baseline[m] :", base[m].tolist())
    print("  candidate[m]:", cand[m].tolist())
