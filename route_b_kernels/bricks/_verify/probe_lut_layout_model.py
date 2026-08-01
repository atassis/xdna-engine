#!/usr/bin/env python3
"""Test the measured lut<4,bfloat16> layout model end to end.

The 256-key address sweep (probe_gather_nondegenerate) measured, for logical index j = key + 128:

    slotA = 16*(j // 16) + (j % 16) // 2      slotB = slotA + 8

read from BOTH ab and cd -- four slots per logical value. slotA is NOT injective (128 distinct over
256 keys) because (j % 16) // 2 collapses j and j+1, so the remaining bit must be the two uint16
halves of the 4-byte slot: even j takes the low half, odd j the high half.

That gives a complete placement rule. This builds a table under it with L[j] = j, so a returned
value names its own logical index, and requires a CONSTANT key vector to come back as ONE value on
every lane. That single check confirms the slot formula, the ab/cd duplication and the half
selection together; any of them wrong and lanes disagree.
"""
import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot

N = 64
WIDTH = 16
TAB = GEN / "layout_model_tables.inc"


def build_table(values):
    """Place `values[j]` at every slot the hardware reads for logical index j."""
    ab = np.zeros(512, dtype=np.float32)
    cd = np.zeros(512, dtype=np.float32)
    for j, v in enumerate(values):
        s0 = 16 * (j // 16) + (j % 16) // 2
        s1 = s0 + 8
        par = j % 2
        for s in (s0, s1):
            ab[2 * s + par] = v
            cd[2 * s + par] = v
    return ab, cd


def write_tables():
    vals = np.arange(256, dtype=np.float32)  # L[j] = j: a value names its own index
    ab, cd = build_table(vals)
    for v in (ab, cd):
        assert np.array_equal(v.astype(ml_dtypes.bfloat16).astype(np.float32), v)

    def emit(name, w16):
        w = w16.astype(ml_dtypes.bfloat16).view(np.uint16)
        rows = [
            "  " + ", ".join(f"0x{x:04x}" for x in w[i : i + 8]) + ","
            for i in range(0, len(w), 8)
        ]
        return (
            f"alignas(::aie::vector_decl_align) static const uint16_t {name}[512] = {{\n"
            + "\n".join(rows)
            + "\n};\n"
        )

    TAB.parent.mkdir(parents=True, exist_ok=True)
    TAB.write_text(
        "// LUT built under the MEASURED placement rule, L[j] = j.\n"
        "//   s0 = 16*(j//16) + (j%16)//2,  s1 = s0+8,  half = j%2,  written to both ab and cd.\n"
        + emit("kLmAb", ab)
        + emit("kLmCd", cd)
    )


def probe(key_value):
    shim = (
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f'#include "{TAB}"\n'
        'extern "C" void gather_lm(int8_t *restrict keys, bfloat16 *restrict sout) {\n'
        "  const ::aie::lut<4,bfloat16> lut(256, kLmAb, kLmCd);\n"
        "  ::aie::parallel_lookup<int8, ::aie::lut<4,bfloat16>> sl(lut, 0, 128);\n"
        f"  for (unsigned i = 0; i < {N}; i += {WIDTH}) {{\n"
        f"    ::aie::vector<int8,{WIDTH}> k = ::aie::load_v<{WIDTH}>(keys + i);\n"
        "    ::aie::store_v(sout + i, sl.fetch(k));\n"
        "  }\n}\n"
    )
    p = GEN / "gather_lm_shim.cc"
    p.write_text(shim)
    design = _build_oneshot("gather_lm", p, [N], N, [np.int8], ml_dtypes.bfloat16, [])
    keys = np.full(N, key_value, dtype=np.int8)
    kt = iron.tensor(np.ascontiguousarray(keys), dtype=np.int8, device="npu")
    ot = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
    design(kt, ot)
    return ot.numpy().astype(np.float32).astype(int)


def main():
    write_tables()
    print("placement: s0 = 16*(j//16) + (j%16)//2, s1 = s0+8, half = j%2, both ab and cd\n")
    bad = []
    for key in range(-128, 128):
        got = probe(key)
        want = key + 128
        uniq = sorted(set(got.tolist()))
        ok = uniq == [want]
        if not ok:
            bad.append((key, want, uniq))
        if key in (-128, -3, 0, 1, 5, 64, 127) or not ok and len(bad) <= 5:
            print(f"  key {key:>4} want {want:>3}: distinct {uniq}  {'OK' if ok else 'MISMATCH'}")
    print(f"\nkeys returning exactly one correct value: {256 - len(bad)}/256")
    if bad:
        print(f"first failures: {bad[:5]}")


if __name__ == "__main__":
    main()
