#!/usr/bin/env python3
"""Phase 1j step 1: reproduce 3f11dcb's "hoisted Look, captured by reference, gathered every
key as 0" failure in isolation, and bisect WHICH of {by-reference capture, NOINLINE outlining,
repeated call across a loop} is the actual cause.

Five arms, one kernel call, same 64 keys, 4 groups of 16 (matching the row-of-groups shape
apply_lut_inplace/put() actually have):
  A base      -- Look built FRESH inside each group's call (today's conv3x3_u8.cc shape). Expect correct.
  B ref+noinl -- Look built ONCE before the loop; a NOINLINE helper takes `const Look &`,
                 called once per group. Matches 3f11dcb's shape (hoisted + by-ref + outlined).
  C ref+inline-- same as B but the helper is forced ALWAYS_INLINE (no real function boundary).
  D val+noinl -- Look built ONCE; NOINLINE helper takes it BY VALUE (a copy per call).
  E ref+lambda-- Look built ONCE; a `[&]` LAMBDA (not a free function) does the fetch, called
                 from a NOINLINE trampoline that forces the lambda to be outlined -- closest
                 structural match to conv3x3_core's `put` lambda pattern.
Prints match counts for all five; exits 0.
"""
import importlib.util
from pathlib import Path
import numpy as np
import bricklib

B = Path(__file__).parent.parent / "conv2d-3x3-u8"
_s = importlib.util.spec_from_file_location("g", B / "golden.py"); g = importlib.util.module_from_spec(_s); _s.loader.exec_module(g)
inc = g.lut_inc(np.arange(256) - 128, bricklib.GEN / "luthoist_place.inc")   # identity table
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

__attribute__((noinline)) static void fetch_ref(Look &look, const int8_t *k, int16_t *o) {{
  aie::vector<int8, 16> kv = aie::load_v<16>(k);
  aie::store_v(o, aie::vector<int16, 16>(aie::unpack(aie::to_fixed<int8>(look.fetch(kv), 0))));
}}

__attribute__((always_inline)) static inline void fetch_ref_inline(Look &look, const int8_t *k, int16_t *o) {{
  aie::vector<int8, 16> kv = aie::load_v<16>(k);
  aie::store_v(o, aie::vector<int16, 16>(aie::unpack(aie::to_fixed<int8>(look.fetch(kv), 0))));
}}

__attribute__((noinline)) static void fetch_val(Look look, const int8_t *k, int16_t *o) {{
  aie::vector<int8, 16> kv = aie::load_v<16>(k);
  aie::store_v(o, aie::vector<int16, 16>(aie::unpack(aie::to_fixed<int8>(look.fetch(kv), 0))));
}}

extern "C" void luthoist(int8_t *k, int16_t *o) {{
  const aie::lut<4, bfloat16> t(256, (const bfloat16 *)kAb, (const bfloat16 *)kCd);

  // A: base -- fresh Look per group.
  for (int g = 0; g < 64; g += 16) {{
    Look look(t, 0, 128);
    aie::vector<int8, 16> kv = aie::load_v<16>(k + g);
    aie::store_v(o + g, aie::vector<int16, 16>(aie::unpack(aie::to_fixed<int8>(look.fetch(kv), 0))));
  }}

  // B: hoisted, by-reference, NOINLINE free function (3f11dcb's shape).
  {{
    Look look(t, 0, 128);
    for (int g = 0; g < 64; g += 16)
      fetch_ref(look, k + g, o + 64 + g);
  }}

  // C: hoisted, by-reference, ALWAYS_INLINE free function.
  {{
    Look look(t, 0, 128);
    for (int g = 0; g < 64; g += 16)
      fetch_ref_inline(look, k + g, o + 128 + g);
  }}

  // D: hoisted, BY VALUE, NOINLINE free function.
  {{
    Look look(t, 0, 128);
    for (int g = 0; g < 64; g += 16)
      fetch_val(look, k + g, o + 192 + g);
  }}

  // E: hoisted, by-reference LAMBDA, called through a NOINLINE trampoline that forces the
  // lambda's body to be outlined (closest structural match to conv3x3_core's `put`).
  {{
    Look look(t, 0, 128);
    auto do_fetch = [&](const int8_t *kk, int16_t *oo) {{
      aie::vector<int8, 16> kv = aie::load_v<16>(kk);
      aie::store_v(oo, aie::vector<int16, 16>(aie::unpack(aie::to_fixed<int8>(look.fetch(kv), 0))));
    }};
    for (int g = 0; g < 64; g += 16)
      do_fetch(k + g, o + 256 + g);
  }}
}}
'''
shim = bricklib.GEN / "luthoist_shim.cc"
shim.write_text(body)
design = bricklib._build_oneshot("luthoist", shim, [64], 320, [np.int8], np.int16, [], stack_size=4096)
import aie.iron as iron
kt = iron.tensor(keys, dtype=np.int8, device="npu")
ot = iron.zeros((320,), dtype=np.int16, device="npu")
design(kt, ot)
o = ot.numpy().astype(np.int64)
arms = {
    "A base       ": o[0:64],
    "B ref+noinl  ": o[64:128],
    "C ref+inline ": o[128:192],
    "D val+noinl  ": o[192:256],
    "E ref+lambda ": o[256:320],
}
base = arms["A base       "]
print("keys:", keys.tolist())
for name, v in arms.items():
    match = int((v == base).sum())
    zeros = int((v == 0).sum())
    print(f"{name}: match {match}/64 vs base, zeros {zeros}/64  first8={v[:8].tolist()}")
