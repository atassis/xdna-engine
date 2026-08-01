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
    # bf16 holds integers exactly only up to 256, and naming (table, index) needs 2*256 = 512 states.
    # One run therefore cannot resolve both. LO gives table + index mod 128; HI gives table + the
    # index's high bit. index = 128*hi + lo.
    ab = np.array([2 * (j % 128) for j in range(256)], dtype=np.float32)
    cd = np.array([2 * (j % 128) + 1 for j in range(256)], dtype=np.float32)
    ab_hi = np.array([2 * (j >> 7) for j in range(256)], dtype=np.float32)
    cd_hi = np.array([2 * (j >> 7) + 1 for j in range(256)], dtype=np.float32)
    for v in (ab, cd, ab_hi, cd_hi):
        assert np.array_equal(v.astype(ml_dtypes.bfloat16).astype(np.float32), v)

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
        + emit("kNdAbHi", ab_hi)
        + emit("kNdCdHi", cd_hi)
    )


def probe(key_value, hi=False):
    a, c = ("kNdAbHi", "kNdCdHi") if hi else ("kNdAb", "kNdCd")
    fn = f"gather_nd_{'hi' if hi else 'lo'}"  # must match the design name passed to _build_oneshot
    shim = (
        "#include <aie_api/aie.hpp>\n#include <stdint.h>\n"
        f'#include "{TAB}"\n'
        f'extern "C" void {fn}(int8_t *restrict keys, bfloat16 *restrict sout) {{\n'
        f"  const ::aie::lut<4,bfloat16> lut(256, {a}, {c});\n"
        "  ::aie::parallel_lookup<int8, ::aie::lut<4,bfloat16>> sl(lut, 0, 128);\n"
        f"  for (unsigned i = 0; i < {N}; i += {WIDTH}) {{\n"
        f"    ::aie::vector<int8,{WIDTH}> k = ::aie::load_v<{WIDTH}>(keys + i);\n"
        "    ::aie::store_v(sout + i, sl.fetch(k));\n"
        "  }\n}\n"
    )
    p = GEN / f"gather_nd_{'hi' if hi else 'lo'}_shim.cc"
    p.write_text(shim)
    design = _build_oneshot(
        f"gather_nd_{'hi' if hi else 'lo'}", p, [N], N, [np.int8], ml_dtypes.bfloat16, []
    )
    keys = np.full(N, key_value, dtype=np.int8)
    kt = iron.tensor(np.ascontiguousarray(keys), dtype=np.int8, device="npu")
    ot = iron.zeros((N,), dtype=ml_dtypes.bfloat16, device="npu")
    design(kt, ot)
    return ot.numpy().astype(np.float32).astype(int)


def main():
    write_tables()
    print("index = 128*hi + lo, table from parity. Sweeping the full signed int8 key range.\n")
    rows = []
    for key in range(-128, 128):
        lo = probe(key)
        hi = probe(key, hi=True)
        # lane 0 of each 4-lane group is the (ab, i) slot; lane 1 is (ab, i+8).
        i0 = 128 * (hi[0] >> 1) + (lo[0] >> 1)
        i1 = 128 * (hi[1] >> 1) + (lo[1] >> 1)
        t0, t1 = lo[0] % 2, lo[1] % 2
        rows.append((key, key + 128, i0, i1, t0, t1))

    print(f"{'key':>5} {'want j':>7} {'slotA':>6} {'slotB':>6} {'B-A':>4}  parity(A,B)")
    for key, want, i0, i1, t0, t1 in rows:
        if key % 16 == 0 or key in (-128, 127, 1, 5, -3):
            print(f"{key:>5} {want:>7} {i0:>6} {i1:>6} {i1 - i0:>4}  ({t0},{t1})")

    arr = np.array([(w, a, b) for _, w, a, b, _, _ in rows])
    print(f"\nslotB - slotA: distinct values {sorted(set((arr[:, 2] - arr[:, 1]).tolist()))}")
    print(f"slotA - want:  distinct values {sorted(set((arr[:, 1] - arr[:, 0]).tolist()))}")
    inj = len(set(arr[:, 1].tolist()))
    print(f"slotA distinct over 256 keys: {inj}  -> {'injective' if inj == 256 else 'NOT injective'}")


if __name__ == "__main__":
    main()
