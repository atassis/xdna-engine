#!/usr/bin/env python3
"""Regenerate gen/ramp_tables.inc, the identity-ramp LUT the gather probes read.

probe_gather_ramp.py and probe_gather_width.py both `#include` this table and neither writes it,
so it was scratch that got lost with gen/. The table makes a returned value name its own key:
L[idx] = idx, so an int8 key k under bias=128 reads L[k+128] = k+128. Any deviation from the
128..(128+n-1) ramp is a gather-stage defect, and a value that comes back twice (or not at all)
is the extract<16>-over-32-lanes signature directly.

Layout matches rope_lut_tables.inc exactly: each logical value duplicated into its 4-byte slot
(arr[2j] = arr[2j+1] = L[j]), 512 uint16, ab == cd.
"""
import pathlib

import ml_dtypes
import numpy as np

OUT = pathlib.Path(__file__).parent / "gen" / "ramp_tables.inc"


def emit(name, values):
    words = np.repeat(values.view(np.uint16), 2)  # arr[2j] = arr[2j+1] = L[j]
    rows = [
        "  " + ", ".join(f"0x{w:04x}" for w in words[i : i + 8]) + ","
        for i in range(0, len(words), 8)
    ]
    return (
        f"alignas(::aie::vector_decl_align) static const uint16_t {name}[{len(words)}] = {{\n"
        + "\n".join(rows)
        + "\n};\n"
    )


def main():
    ramp = np.arange(256, dtype=np.float32).astype(ml_dtypes.bfloat16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "// Auto-generated identity-ramp gather-LUT (int8 key, bias=128, 256 logical entries).\n"
        "// L[idx] = idx, so a fetched value names its own key: key k reads L[k+128] = k+128.\n"
        "// Same layout as rope_lut_tables.inc: arr[2j]=arr[2j+1]=L[j], 512 uint16, ab==cd.\n"
        "// Regenerate with gen_ramp_tables.py.\n" + emit("kRampAb", ramp)
    )
    # bf16 has 8 mantissa bits, so 0..255 is exact and the ramp is a faithful oracle.
    back = ramp.astype(np.float32)
    assert np.array_equal(back, np.arange(256)), "bf16 cannot represent the ramp exactly"
    print(f"wrote {OUT} ({len(ramp)} entries, exact in bf16)")


if __name__ == "__main__":
    main()
