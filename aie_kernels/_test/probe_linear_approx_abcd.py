#!/usr/bin/env python3
"""Which ab/cd layout does aie::linear_approx<bfloat16, lut<4,float,bfloat16>> actually read?

There is no working linear_approx call site in this repo to copy. rope-lut uses parallel_lookup (a
different primitive) and probe_lut_ab_cd.py probes THAT primitive's banks. The linear_approx packing
lives only in the shuffle patterns of aie_api/detail/aie2/linear_approx.hpp:346-420:

    load_lut_2x_float(LUT_ab_, LUT_cd_, index, coeff0, coeff1)
    offset = shuffle(coeff0, coeff1, T32_16x2_hi)   // f32 lanes, "hi"
    slope  = shuffle(coeff0, coeff1, T16_16x4_lo)   // bf16 lanes, "lo"
    result = mac_elem(slope, delay_in_, offset)     // offset + slope * DELAYED input

This probe breaks the degeneracy the same way probe_lut_ab_cd.py did: make the intended offset and
slope of each entry recognizable constants, feed keys whose intended entry is unambiguous, and read
back what compute() returns. Three configs, each isolating one unknown:

  A. slope[e] = 0, offset[e] = 1000 + e   -> output should be exactly 1000 + entry(u).
     Reveals the INDEX mapping (and whether bias/step_bits mean what we think).
  B. slope[e] = 1, offset[e] = 0          -> output should be exactly u (identity).
     Reveals the slope lane extraction AND the delay-line shift.
  C. slope[e] = 0, offset[e] = 1000 + e, but written with the pair order SWAPPED in each bank.
     If A is wrong and C is right, the fix is the pair order.

Nothing here is a gate; it prints what came back so the layout can be read off. Device required.
"""
import numpy as np
import ml_dtypes

import bricklib
from bricklib import GEN, _build_oneshot

N = 64                 # elements per call (multiple of 16)
ENTRIES = 64           # small table so the whole sweep is inspectable
BIAS = ENTRIES // 2


def write_tables(name, offsets, slopes, swap_pair):
    """Emit ab/cd as raw f32 words, 4 copies per entry, even entries -> ab, odd -> cd.

    Each entry contributes a (offset, slope) pair per copy; `swap_pair` writes (slope, offset)
    instead so config C can distinguish pair order.
    """
    ab, cd = [], []
    for e in range(ENTRIES):
        pair = [slopes[e], offsets[e]] if swap_pair else [offsets[e], slopes[e]]
        (ab if e % 2 == 0 else cd).extend(pair)
    ab = ab * 4
    cd = cd * 4

    def fmt(v):
        return ", ".join(f"{x:.8e}f" for x in v)

    p = GEN / f"linapprox_{name}_tables.inc"
    p.write_text(
        f"// AUTO-GENERATED discriminating linear_approx table: {name}\n"
        "#pragma once\n"
        f"static const float kProbeAb[] = {{{fmt(ab)}}};\n"
        f"static const float kProbeCd[] = {{{fmt(cd)}}};\n"
        f"static constexpr int kProbeEntries = {ENTRIES};\n"
    )
    return p


def run_config(name, offsets, slopes, swap_pair, u):
    tab = write_tables(name, offsets, slopes, swap_pair)
    shim = GEN / f"linapprox_{name}_shim.cc"
    shim.write_text(f'''// AUTO-GENERATED linear_approx probe shim ({name}).
#include <aie_api/aie.hpp>
#include <stdint.h>
#include "{tab}"
extern "C" void linapprox_probe_{name}(bfloat16 *restrict u, float *restrict out) {{
  const ::aie::lut<4, float, bfloat16> tbl(kProbeEntries, (void *)kProbeAb, (void *)kProbeCd);
  ::aie::linear_approx<bfloat16, ::aie::lut<4, float, bfloat16>> approx(tbl, /*step_bits=*/0,
                                                                       /*bias=*/{BIAS});
  for (int i = 0; i < {N}; i += 16) {{
    ::aie::vector<bfloat16, 16> v = ::aie::load_v<16>(u + i);
    ::aie::accum<accfloat, 16> acc = approx.compute(v);
    ::aie::store_v(out + i, acc.template to_vector<float>(0));
  }}
}}
''')
    design = _build_oneshot(f"linapprox_probe_{name}", shim, [N], N,
                            [ml_dtypes.bfloat16], np.float32, bricklib._AIE_API_INC)
    import aie.iron as iron
    in_t = iron.tensor(np.ascontiguousarray(u.astype(ml_dtypes.bfloat16)),
                       dtype=ml_dtypes.bfloat16, device="npu")
    out_t = iron.zeros((N,), dtype=np.float32, device="npu")
    design(in_t, out_t)
    return out_t.numpy().copy()


if __name__ == "__main__":
    # u sweeps entries -BIAS .. -BIAS+N-1 in steps of 1, i.e. entry index 0..N-1
    u = (np.arange(N, dtype=np.float32) - BIAS).astype(np.float32)
    expected_entry = np.arange(N)

    offs_ramp = (1000.0 + np.arange(ENTRIES)).astype(np.float32)
    zeros = np.zeros(ENTRIES, dtype=np.float32)
    ones = np.ones(ENTRIES, dtype=np.float32)

    print(f"u          = {u[:12]} ...")
    print(f"exp. entry = {expected_entry[:12]} ...  (config A should print 1000+entry)")

    for name, offs, slps, swap in (
        ("A_offset_ramp", offs_ramp, zeros, False),
        ("B_identity", zeros, ones, False),
        ("C_swapped", offs_ramp, zeros, True),
    ):
        try:
            got = run_config(name, offs, slps, swap, u)
        except Exception as exc:  # a probe must report a failure, not die silently
            print(f"\n### {name}: BUILD/RUN FAILED: {type(exc).__name__}: {exc}")
            continue
        print(f"\n### {name}")
        print("  out[:12] :", np.array2string(got[:12], precision=3, suppress_small=True))
        if name.startswith("A") or name.startswith("C"):
            pred = (1000.0 + expected_entry).astype(np.float32)
            hit = int(np.sum(np.isclose(got, pred, atol=0.5)))
            print(f"  matches 1000+entry on {hit}/{N} lanes")
            for shift in (0, 16, -16):
                if shift == 0:
                    continue
                rolled = np.roll(pred, shift)
                h = int(np.sum(np.isclose(got, rolled, atol=0.5)))
                print(f"    with delay shift {shift:+d}: {h}/{N}")
        else:
            hit = int(np.sum(np.isclose(got, u, atol=0.05)))
            print(f"  matches u (identity) on {hit}/{N} lanes")
            for shift in (16, -16):
                h = int(np.sum(np.isclose(got, np.roll(u, shift), atol=0.05)))
                print(f"    with delay shift {shift:+d}: {h}/{N}")
