#!/usr/bin/env python3
"""Which table does each output lane of parallel_lookup actually read?

probe_gather_ramp.py concluded the fetch merely PERMUTES its inputs (perm [0,8,1,9,...])
and rope_lut.cc compensates with concat(filter_even, filter_odd). That probe was
degenerate in two ways at once and could not have seen anything else:

  1. its LUT was a ramp, value == key + 128, so "lane 1 holds the value for input lane 8"
     and "lane 1 read table index key+8" produce the identical reading;
  2. it passed the SAME array as both `ab` and `cd` (`lut(256, kRampAb, kRampAb)`), so any
     mix-up between the two halves was invisible.

The rope kernel violates both assumptions: it passes two DIFFERENT arrays, and its values
are not its keys. probe_rope_sincos then showed that with ALL keys equal to 0 the fetch
still returns two DIFFERENT values alternating -- which a permutation cannot do.

This probe breaks the degeneracy: ab[i] = i, cd[i] = i + 1000, and all keys = 0. Then
every output lane reports, unambiguously, both WHICH array it came from (< or >= 1000)
and WHICH index it used (the value itself).
"""
import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot

N = 32


def write_tables():
    """ab[i] = i, cd[i] = i + 1000, 256 entries each, as bf16-exact integers."""
    ab = np.arange(256, dtype=np.float32)
    cd = (np.arange(256) + 1000).astype(np.float32)
    # bf16 represents these integers exactly (all < 2^8 and 1000..1255 need 11 bits of
    # mantissa... 1255 needs 11 -> NOT exact in bf16's 8-bit mantissa). Use a coarser
    # offset that stays bf16-exact: multiples of 8 above 1024 are exact.
    cd = (np.arange(256) * 8 + 2048).astype(np.float32)
    lines = ["// AUTO-GENERATED discriminating LUT: ab[i]=i, cd[i]=i*8+2048",
             "static const bfloat16 kAbT[256] = {" + ",".join(f"{v:.1f}" for v in ab) + "};",
             "static const bfloat16 kCdT[256] = {" + ",".join(f"{v:.1f}" for v in cd) + "};"]
    p = GEN / "abcd_tables.inc"
    p.write_text("\n".join(lines) + "\n")
    return p, ab, cd


def main():
    tab, ab, cd = write_tables()
    shim = (
        '#include <aie_api/aie.hpp>\n#include <stdint.h>\n'
        f'#include "{tab}"\n'
        'extern "C" void abcd_probe(int8_t *restrict keys, bfloat16 *restrict out) {\n'
        '  const ::aie::lut<4,bfloat16> lut(256, kAbT, kCdT);\n'
        '  ::aie::parallel_lookup<int8,::aie::lut<4,bfloat16>> sl(lut, 0, 128);\n'
        f'  ::aie::vector<int8,16> k = ::aie::load_v<16>(keys);\n'
        '  ::aie::store_v(out, sl.fetch(k));\n'
        '  ::aie::vector<int8,16> k2 = ::aie::load_v<16>(keys + 16);\n'
        '  ::aie::store_v(out + 16, sl.fetch(k2));\n'
        '}\n')
    p = GEN / "abcd_shim.cc"
    p.write_text(shim)
    design = _build_oneshot("abcd_probe", p, [N], N, [np.int8], ml_dtypes.bfloat16, [])

    for label, keys in (("all keys = 0", np.zeros(N, dtype=np.int8)),
                        ("all keys = 3", np.full(N, 3, dtype=np.int8)),
                        ("ramp keys 0..31", np.arange(N, dtype=np.int8))):
        kt = iron.tensor(np.ascontiguousarray(keys), dtype=np.int8, device="npu")
        ot = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
        design(kt, ot)
        v = ot.numpy().astype(np.float32)
        src = ["cd" if x >= 2048 else "ab" for x in v]
        idx = [int((x - 2048) // 8) if x >= 2048 else int(x) for x in v]
        print(f"\n### {label}")
        print(f"  raw    {v[:16].tolist()}")
        print(f"  source {src[:16]}")
        print(f"  index  {idx[:16]}")


if __name__ == "__main__":
    main()
