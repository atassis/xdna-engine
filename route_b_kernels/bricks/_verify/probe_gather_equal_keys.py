#!/usr/bin/env python3
"""Does a width-16 fetch return EQUAL values for EQUAL keys?

probe_gather_width exonerated the gather using keys 0..n-1 -- all DISTINCT. The rope-lut failure is
at pos=0, where every key is 0. "Distinct keys map correctly" does not imply "equal keys map
correctly", and the whole redirect in rope-lut-gather-exonerated-hunt-moves-to-the-keys rests on
the second claim. This tests it.

Identity-ramp LUT (L[idx] = idx), so key k reads k+128. Feed a constant key vector and read the RAW
fetch, before any de-interleave -- with all keys equal, the de-interleave is the identity, so raw is
the whole story. Expected: every lane returns 128+key.

If lanes 8-15 come back at 128+key+8, the +8 offset lives in the GATHER and the exoneration was
wrong -- narrowed to the equal-key case, which the ramp probe structurally could not see.
"""
import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot

TAB = str(GEN) + "/ramp_tables.inc"
N = 64
WIDTH = 16


def probe(key_value):
    shim = (
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f'#include "{TAB}"\n'
        "extern \"C\" void gather_eq(int8_t *restrict keys, bfloat16 *restrict sout) {\n"
        "  const ::aie::lut<4,bfloat16> lut(256, kRampAb, kRampAb);\n"
        "  ::aie::parallel_lookup<int8, ::aie::lut<4,bfloat16>> sl(lut, 0, 128);\n"
        f"  for (unsigned i = 0; i < {N}; i += {WIDTH}) {{\n"
        f"    ::aie::vector<int8,{WIDTH}> k = ::aie::load_v<{WIDTH}>(keys + i);\n"
        "    ::aie::store_v(sout + i, sl.fetch(k));\n"
        "  }\n}\n"
    )
    p = GEN / "gather_eq_shim.cc"
    p.write_text(shim)
    design = _build_oneshot("gather_eq", p, [N], N, [np.int8], ml_dtypes.bfloat16, [])
    keys = np.full(N, key_value, dtype=np.int8)
    kt = iron.tensor(np.ascontiguousarray(keys), dtype=np.int8, device="npu")
    ot = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
    design(kt, ot)
    return ot.numpy().astype(np.float32)


def main():
    for key_value in (0, 5, -3):
        got = probe(key_value)
        want = 128 + key_value
        uniq = sorted(set(got.astype(int).tolist()))
        print(f"\n### all keys = {key_value}   (every lane must read {want})")
        print(f"  raw[:16]  {got[:16].astype(int).tolist()}")
        print(f"  distinct values across all {N} lanes: {uniq}")
        ok = np.all(got == want)
        print(f"  all lanes correct: {ok}")
        if not ok:
            bad = np.nonzero(got != want)[0]
            print(f"  wrong lanes (first 16): {bad[:16].tolist()}")
            print(f"  lane index mod {WIDTH}: {sorted(set((bad % WIDTH).tolist()))}")
            print(f"  offset seen: {sorted(set((got[bad] - want).astype(int).tolist()))}")


if __name__ == "__main__":
    main()
