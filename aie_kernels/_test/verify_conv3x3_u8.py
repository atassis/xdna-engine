#!/usr/bin/env python3
"""conv2d-3x3-u8 device gate: bit-exact against the integer golden.

One streamed tile per output row (its three input rows), weights + bias resident. Covers the
three row positions (top / middle / bottom), widths other than 32 (mlir-aie's vector conv2dk3
is hardwired to 32), and the valid-column mask. The gate is exact equality, since the kernel
and the golden implement the same integer arithmetic.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import bricklib

BRICK = Path(__file__).parent.parent / "conv2d-3x3-u8"
_spec = importlib.util.spec_from_file_location("conv3x3_golden", BRICK / "golden.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)

CIN, COUT, H, SHIFT = 64, 16, 6, 11


def case(name, width, check, valid_lo=0, valid_hi=None, seed=0, signed=False, lut=False,
         gate=None):
    hi = width if valid_hi is None else valid_hi
    rng = np.random.default_rng(seed)
    adt = np.int8 if signed else np.uint8
    lo_v, hi_v = (-128, 128) if signed else (0, 256)
    x = rng.integers(lo_v, hi_v, size=(CIN, H, width), dtype=np.int64).astype(adt)
    w = rng.integers(-127, 128, size=(COUT, CIN, 3, 3), dtype=np.int64).astype(np.int8)
    b = rng.integers(-(1 << 15), 1 << 15, size=(COUT,), dtype=np.int64).astype(np.int32)
    ref = g.conv3x3_u8_ref(x, w, b, SHIFT, valid_lo, hi, signed=signed)
    fn, ct = ("conv3x3_i8", "int8_t") if signed else ("conv3x3_u8", "uint8_t")
    params = g.pack_params(w, b)
    lut_flag = []
    if lut:
        # a random table: a smooth one could hide a lane permutation in the gather
        table = rng.integers(-128, 128, size=256, dtype=np.int64)
        ref = table[ref.astype(np.int64) + 128].astype(np.int8)
        ref[:, :, :valid_lo] = 0
        ref[:, :, hi:] = 0
        fn = "conv3x3_i8_lut"
        lut_flag = [f"-DCONV3X3_LUT_INC=\"{g.lut_inc(table, bricklib.GEN / (name + '_lut.inc'))}\""]
    if gate is not None:
        # SPAN block tail: table = sigmoid(c * 0.05) - 0.5 on a 0.5/127 grid, or random
        ks = np.arange(256) - 128
        table = (rng.integers(-127, 128, size=256, dtype=np.int64) if gate == "random" else
                 np.round((1 / (1 + np.exp(-ks * 0.05)) - 0.5) / (0.5 / 127)).astype(np.int64))
        GA, GB, GS1, GC, GS2 = 64, 45, 6, 100, 9
        xblk = rng.integers(-128, 128, size=(COUT, H, width), dtype=np.int64).astype(np.int8)
        xblk[:, :, :valid_lo] = 0
        xblk[:, :, hi:] = 0
        ref = g.gate_ref(ref, xblk, table, GA, GB, GS1, GC, GS2)
        ref[:, :, :valid_lo] = 0
        ref[:, :, hi:] = 0
        fn = "conv3x3_i8_gate"
        lut_flag = [f"-DCONV3X3_LUT_INC=\"{g.lut_inc(table, bricklib.GEN / (name + '_lut.inc'))}\""]
        xrows = g.pack_rows(xblk).reshape(H, -1)

    wp = width + 2 * g.PAD
    rows = g.pack_rows(x).reshape(H, -1)
    blank = np.zeros_like(rows[0])
    if check == 0:
        ys = [0]
        tiles = [np.concatenate([blank, rows[0], rows[1]])]
    elif check == 2:
        ys = [H - 1]
        tiles = [np.concatenate([rows[H - 2], rows[H - 1], blank])]
    else:
        ys = list(range(1, H - 1))
        tiles = [np.concatenate([rows[y - 1], rows[y], rows[y + 1]]) for y in ys]
    if gate is not None:
        tiles = [np.concatenate([t, xrows[y]]) for t, y in zip(tiles, ys)]
    tiles = np.stack(tiles)

    shim = bricklib.GEN / f"{name}_shim.cc"
    sym = f"conv3x3_verify_{name}"
    shim.write_text(
        f'#include <stdint.h>\n#include "{BRICK / "conv3x3_u8.cc"}"\n'
        f'extern "C" void {sym}({ct} *t, int8_t *p, {ct} *o) {{\n'
        + (f'  {fn}(t, t + {wp * CIN}, t + {2 * wp * CIN}, t + {3 * wp * CIN}, p, o, {width},'
           f' {check}, {SHIFT}, {valid_lo}, {hi}, 64, 45, 6, 100, 9);\n}}\n' if gate is not None else
           f'  {fn}(t, t + {wp * CIN}, t + {2 * wp * CIN}, p, o, {width}, {check},'
           f' {SHIFT}, {valid_lo}, {hi});\n}}\n'))
    exp = np.stack([ref[:, y, :] for y in ys])            # [n, COUT, W]

    margin_nz = []

    def unpack(dev):
        full = np.stack([g.unpack_rows(r, COUT, 1, width, margins=True)[:, 0, :] for r in dev])
        margin_nz.append(int(np.count_nonzero(full[:, :, :g.PAD]) +
                             np.count_nonzero(full[:, :, -g.PAD:])))
        return full[:, :, g.PAD:-g.PAD]

    res = bricklib.verify_streamed(
        name, shim, sym, tiles, wp * COUT, params, unpack, exp, gate=0.0,
        in_dt=adt, out_dt=adt, resident_dt=np.int8,
        compile_flags=[f"-DCONV3X3_CIN={CIN}", f"-DCONV3X3_COUT={COUT}"] + lut_flag,
        stack_size=2048 if (lut or gate is not None) else None)  # the LUT path measures 1984 B; aiecc re-checks
    got = np.asarray(res["got"]).astype(np.int64)
    mism = int((got != exp.astype(np.int64)).sum())
    res["mismatches"] = mism
    res["margin_nonzero"] = margin_nz[-1]
    res["ok"] = bool(res["ok"] and mism == 0 and margin_nz[-1] == 0)
    res["status"] = "PASS" if res["ok"] else (
        f"FAIL({mism}/{exp.size} mismatched, {margin_nz[-1]} nonzero margin bytes)")
    print(f"[{name:22s}] exact: {exp.size - mism}/{exp.size}, margin nonzero "
          f"{margin_nz[-1]} -> {res['status']}")
    return res


CASES = [
    ("c3_w32_mid", 32, 1),
    ("c3_w64_mid", 64, 1),
    ("c3_w48_mid", 48, 1),
    ("c3_w64_top", 64, 0),
    ("c3_w64_bot", 64, 2),
    ("c3_w64_mask", 64, 1, 5, 59),
    ("c3s_w64_mid", 64, 1, 0, None, 1, True),
    ("c3s_w48_top", 48, 0, 0, None, 2, True),
    ("c3s_w64_mask", 64, 1, 5, 59, 3, True),
    ("c3l_w64_mid", 64, 1, 0, None, 4, True, True),
    ("c3l_w48_mask", 48, 1, 5, 43, 5, True, True),
    ("c3g_w64_sig", 64, 1, 0, None, 6, True, False, "sigmoid"),
    ("c3g_w48_rand_mask", 48, 1, 5, 43, 7, True, False, "random"),
]

if __name__ == "__main__":
    results = []
    for c in CASES:
        try:
            results.append(case(*c))
        except Exception as e:  # keep going; one bad case must not hide the others
            print(f"[{c[0]:22s}] ERROR: {e}")
            results.append(dict(name=c[0], status="ERROR", ok=False, err=str(e)))
    print("\n==== conv2d-3x3-u8 ====")
    for r in results:
        print(f"  {r['name']:22s} {r['status']}")
    passed = sum(1 for r in results if r.get("ok"))
    print(f"conv2d-3x3-u8: {passed}/{len(results)} PASS")
    print("JSON " + json.dumps([{k: v for k, v in r.items() if k != 'got'} for r in results]))
    sys.exit(0 if passed == len(results) else 1)
