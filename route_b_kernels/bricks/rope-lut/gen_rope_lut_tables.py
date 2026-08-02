#!/usr/bin/env python3
"""Generate rope_lut_tables.inc under the MEASURED aie::lut<4,bfloat16> placement rule.

The hardware reads four slots per logical entry j (measured over all 256 keys by
probe_lut_layout_model.py, which asserts this rule against the device):

    s0 = 16*(j // 16) + (j % 16) // 2      s1 = s0 + 8      half = j % 2

from BOTH ab and cd. All four must hold the same value. `s0` alone is not injective -- it maps j and
j+1 together -- so the 4-byte slot's two uint16 halves carry the remaining bit.

Verified end to end by probe_lut_layout_model.py: a table built this way returns exactly one correct
value on all 64 lanes for every one of the 256 keys.

The previous layout wrote arr[2j] = arr[2j+1] = L[j] linearly with ab == cd, so the four read slots
held L[s0] and L[s0+8] -- two values where one is required.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from golden import sincos_lut_tables  # noqa: E402

OUT = pathlib.Path(__file__).parent / "rope_lut_tables.inc"


def place(values: np.ndarray) -> np.ndarray:
    """Scatter a 256-entry logical table into the 512-uint16 physical layout."""
    out = np.zeros(512, dtype=np.float32)
    for j, v in enumerate(values):
        s0 = 16 * (j // 16) + (j % 16) // 2
        for s in (s0, s0 + 8):
            out[2 * s + (j % 2)] = v
    return out


def emit(name: str, physical: np.ndarray) -> str:
    import ml_dtypes

    w = physical.astype(ml_dtypes.bfloat16).view(np.uint16)
    rows = [
        "  " + ", ".join(f"0x{x:04x}" for x in w[i : i + 8]) + ","
        for i in range(0, len(w), 8)
    ]
    return (
        f"alignas(::aie::vector_decl_align) static const uint16_t {name}[512] = {{\n"
        + "\n".join(rows)
        + "\n};\n"
    )


def main():
    sin_tab, cos_tab = sincos_lut_tables()
    sin_phys, cos_phys = place(sin_tab), place(cos_tab)
    OUT.write_text(
        "// Auto-generated sin/cos gather-LUT tables (int8 key, bias=128, 256 logical entries).\n"
        "// L[idx] = sin/cos((idx-128)*pi/128).\n"
        "//\n"
        "// PLACEMENT is the measured aie::lut<4,bfloat16> read pattern, not a linear array: for\n"
        "// logical entry j the hardware reads s0 = 16*(j//16) + (j%16)//2 and s0+8, in BOTH ab and\n"
        "// cd, taking the low uint16 of the 4-byte slot for even j and the high one for odd j. All\n"
        "// four slots therefore carry the same value. Regenerate with gen_rope_lut_tables.py.\n"
        + emit("kSinLutAb", sin_phys)
        + emit("kSinLutCd", sin_phys)
        + emit("kCosLutAb", cos_phys)
        + emit("kCosLutCd", cos_phys)
    )
    # Round-trip the placement to prove every logical entry is recoverable from all four slots.
    import ml_dtypes

    for name, log, phys in (("sin", sin_tab, sin_phys), ("cos", cos_tab, cos_phys)):
        q = phys.astype(ml_dtypes.bfloat16).astype(np.float32)
        for j, v in enumerate(log.astype(ml_dtypes.bfloat16).astype(np.float32)):
            s0 = 16 * (j // 16) + (j % 16) // 2
            for s in (s0, s0 + 8):
                assert q[2 * s + (j % 2)] == v, f"{name}[{j}] lost at slot {s}"
    print(f"wrote {OUT} (256 logical entries, 4 slots each, round-trip verified)")


if __name__ == "__main__":
    main()
