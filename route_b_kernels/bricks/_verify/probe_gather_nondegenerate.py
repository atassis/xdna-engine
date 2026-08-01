#!/usr/bin/env python3
"""Name the rope-lut gather's addressing fault, with both prior degeneracies removed.

Every gather probe so far has been degenerate in the two ways 2026-07-25 warned about:
  (1) an identity-ramp LUT (value == index), so a value cannot distinguish "lane permuted" from
      "index offset"; and
  (2) ab == cd (the same array passed twice), so reading the wrong one of the pair is invisible.

Tables here remove both:
    ab[j] = 2*(j % 128)       -> EVEN  values 0..254
    cd[j] = 2*(j % 128) + 1   -> ODD   values 1..255

so a returned value decodes to two independent facts:
    parity      -> which table the lane actually read (even = ab, odd = cd)
    value >> 1  -> the index it actually read, mod 128

Neither equals the index, so nothing can be mistaken for its own address. All values are integers
<= 255 and therefore exact in bf16 (8-bit significand), so the oracle cannot round.

Keys are CONSTANT per run -- that is the input that makes a permutation provably the identity, so
any spread across lanes is an addressing fault rather than a reordering. Keys are swept across the
signed int8 range because the failure at key=0 and key=5 and key=-3 did not share a single offset.
"""
import numpy as np
import ml_dtypes

from bricklib import GEN, iron, _build_oneshot

N = 64
WIDTH = 16
TAB = GEN / "nondegen_tables.inc"


def write_tables():
    ab = np.array([2 * (j % 128) for j in range(256)], dtype=np.float32)
    cd = np.array([2 * (j % 128) + 1 for j in range(256)], dtype=np.float32)
    assert np.array_equal(ab.astype(ml_dtypes.bfloat16).astype(np.float32), ab)
    assert np.array_equal(cd.astype(ml_dtypes.bfloat16).astype(np.float32), cd)

    def emit(name, vals):
        w = np.repeat(vals.astype(ml_dtypes.bfloat16).view(np.uint16), 2)
        rows = [
            "  " + ", ".join(f"0x{x:04x}" for x in w[i : i + 8]) + ","
            for i in range(0, len(w), 8)
        ]
        return (
            f"alignas(::aie::vector_decl_align) static const uint16_t {name}[{len(w)}] = {{\n"
            + "\n".join(rows)
            + "\n};\n"
        )

    TAB.parent.mkdir(parents=True, exist_ok=True)
    TAB.write_text(
        "// Non-degenerate gather-LUT pair: ab holds EVEN values, cd holds ODD, neither equal to its\n"
        "// own index. parity of a fetched value names the table; value>>1 names the index mod 128.\n"
        + emit("kNdAb", ab)
        + emit("kNdCd", cd)
    )


def probe(key_value):
    shim = (
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f'#include "{TAB}"\n'
        'extern "C" void gather_nd(int8_t *restrict keys, bfloat16 *restrict sout) {\n'
        "  const ::aie::lut<4,bfloat16> lut(256, kNdAb, kNdCd);\n"
        "  ::aie::parallel_lookup<int8, ::aie::lut<4,bfloat16>> sl(lut, 0, 128);\n"
        f"  for (unsigned i = 0; i < {N}; i += {WIDTH}) {{\n"
        f"    ::aie::vector<int8,{WIDTH}> k = ::aie::load_v<{WIDTH}>(keys + i);\n"
        "    ::aie::store_v(sout + i, sl.fetch(k));\n"
        "  }\n}\n"
    )
    p = GEN / "gather_nd_shim.cc"
    p.write_text(shim)
    design = _build_oneshot("gather_nd", p, [N], N, [np.int8], ml_dtypes.bfloat16, [])
    keys = np.full(N, key_value, dtype=np.int8)
    kt = iron.tensor(np.ascontiguousarray(keys), dtype=np.int8, device="npu")
    ot = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
    design(kt, ot)
    return ot.numpy().astype(np.float32).astype(int)


def main():
    write_tables()
    print("tables: ab[j]=2*(j%128) even, cd[j]=2*(j%128)+1 odd; parity=table, value>>1=index%128\n")
    for key in (0, 1, 5, -3, 64, -64, 127, -128):
        got = probe(key)
        want_idx = (key + 128) % 128
        table = np.where(got % 2 == 0, "ab", "cd")
        idx = got >> 1
        print(f"### all keys = {key}   want index%128 = {want_idx} from ab on every lane")
        print(f"  raw[:16]    {got[:16].tolist()}")
        print(f"  table[:16]  {table[:16].tolist()}")
        print(f"  index[:16]  {idx[:16].tolist()}")
        bad_tab = int((table != "ab").sum())
        bad_idx = int((idx != want_idx).sum())
        print(f"  lanes reading cd: {bad_tab}/{N}    lanes with wrong index: {bad_idx}/{N}")
        if bad_idx:
            print(f"  distinct indices seen: {sorted(set(idx.tolist()))}  (offsets {sorted(set((idx - want_idx).tolist()))})")
        print()


if __name__ == "__main__":
    main()
