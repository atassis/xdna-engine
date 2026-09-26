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


def case(name, width, check, valid_lo=0, valid_hi=None, seed=0):
    hi = width if valid_hi is None else valid_hi
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, size=(CIN, H, width), dtype=np.int64).astype(np.uint8)
    w = rng.integers(-127, 128, size=(COUT, CIN, 3, 3), dtype=np.int64).astype(np.int8)
    b = rng.integers(-(1 << 15), 1 << 15, size=(COUT,), dtype=np.int64).astype(np.int32)
    ref = g.conv3x3_u8_ref(x, w, b, SHIFT, valid_lo, hi)

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
    tiles = np.stack(tiles)

    shim = bricklib.GEN / f"{name}_shim.cc"
    sym = f"conv3x3_verify_{name}"
    shim.write_text(
        f'#include <stdint.h>\n#include "{BRICK / "conv3x3_u8.cc"}"\n'
        f'extern "C" void {sym}(uint8_t *t, int8_t *p, uint8_t *o) {{\n'
        f'  conv3x3_u8(t, t + {width * CIN}, t + {2 * width * CIN}, p, o, {width}, {check},'
        f' {SHIFT}, {valid_lo}, {hi});\n}}\n')
    exp = np.stack([ref[:, y, :] for y in ys])            # [n, COUT, W]

    def unpack(dev):
        return np.stack([g.unpack_rows(r, COUT, 1, width)[:, 0, :] for r in dev])

    res = bricklib.verify_streamed(
        name, shim, sym, tiles, width * COUT, g.pack_params(w, b), unpack, exp, gate=0.0,
        in_dt=np.uint8, out_dt=np.uint8, resident_dt=np.int8,
        compile_flags=[f"-DCONV3X3_CIN={CIN}", f"-DCONV3X3_COUT={COUT}"])
    got = np.asarray(res["got"]).astype(np.int64)
    mism = int((got != exp.astype(np.int64)).sum())
    res["mismatches"] = mism
    res["ok"] = bool(res["ok"] and mism == 0)
    res["status"] = "PASS" if res["ok"] else f"FAIL({mism}/{exp.size} mismatched)"
    print(f"[{name:22s}] exact: {exp.size - mism}/{exp.size} -> {res['status']}")
    return res


CASES = [
    ("c3_w32_mid", 32, 1),
    ("c3_w64_mid", 64, 1),
    ("c3_w24_mid", 24, 1),
    ("c3_w64_top", 64, 0),
    ("c3_w64_bot", 64, 2),
    ("c3_w64_mask", 64, 1, 5, 59),
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
