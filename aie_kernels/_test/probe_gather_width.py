#!/usr/bin/env python3
"""Is parallel_lookup's bf16 fetch usable at width 16, or only at its native 32?

rope_lut.cc sets kFetchW=16 while its own comment records that a bf16-valued LUT has
output_lanes == 512/8/2 == 32. probe_gather_ramp.py (which works) fetches at 32. If the
hardware always produces 32 lanes, a 16-lane fetch reads a subset of a wider result and
the values land in the wrong places -- the same sub-native class of bug as the int8 key
store, one layer up.

Feed an identity ramp LUT (value[k] == k+128) so the returned value IS its own key, and
read it back at both widths. Any deviation from the permuted 128..159 that width 32 gives
tells us width 16 is not a legal fetch granularity.
"""
import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot

TAB = str(GEN) + "/ramp_tables.inc"


def probe(width, n):
    shim = (
        '#include <aie_api/aie.hpp>\n#include <stdint.h>\n'
        f'#include "{TAB}"\n'
        'extern "C" void gather_w(int8_t *restrict keys, bfloat16 *restrict sout) {\n'
        '  const ::aie::lut<4,bfloat16> lut(256, kRampAb, kRampAb);\n'
        '  ::aie::parallel_lookup<int8, ::aie::lut<4,bfloat16>> sl(lut, 0, 128);\n'
        f'  for (unsigned i = 0; i < {n}; i += {width}) {{\n'
        f'    ::aie::vector<int8,{width}> k = ::aie::load_v<{width}>(keys + i);\n'
        '    ::aie::store_v(sout + i, sl.fetch(k));\n'
        '  }\n}\n')
    p = GEN / f"gather_w{width}_shim.cc"
    p.write_text(shim)
    design = _build_oneshot(f"gather_w", p, [n], n, [np.int8], ml_dtypes.bfloat16, [])
    keys = np.arange(n, dtype=np.int8)
    kt = iron.tensor(np.ascontiguousarray(keys), dtype=np.int8, device="npu")
    ot = iron.zeros((n,), dtype=ml_dtypes.bfloat16, device="npu")
    design(kt, ot)
    return ot.numpy().astype(np.float32)


def main():
    n = 64
    exp = np.arange(n) + 128  # ramp LUT: value == key + 128
    for width in (32, 16):
        got = probe(width, n)
        # undo the documented within-group de-interleave, at THIS width, and see if the
        # ramp comes back in order. If it does, the width is usable; if not, it is not.
        g = got.reshape(-1, width)
        deint = np.concatenate([np.concatenate([grp[0::2], grp[1::2]]) for grp in g])
        ok_raw = np.array_equal(got, exp)
        ok_deint = np.array_equal(deint, exp)
        print(f"\n### fetch width {width}")
        print(f"  raw[:16]        {got[:16].astype(int).tolist()}")
        print(f"  de-interleaved  {deint[:16].astype(int).tolist()}   (expect 128..143)")
        print(f"  in-order raw={ok_raw}  in-order after de-interleave={ok_deint}")
        missing = sorted(set(exp.tolist()) - set(got.astype(int).tolist()))
        dup = len(got) - len(set(got.astype(int).tolist()))
        print(f"  values missing from the read: {missing if missing else 'none'}   duplicates: {dup}")


if __name__ == "__main__":
    main()
