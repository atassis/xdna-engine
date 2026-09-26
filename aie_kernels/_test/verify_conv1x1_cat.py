#!/usr/bin/env python3
"""conv2d-1x1-cat device gate: bit-exact against the integer golden.

One streamed tile per output row, holding the NSRC source rows back to back (what a MemTile join
delivers); the kernel reads them through four pointers, so no concatenated copy exists.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import bricklib

BRICK = Path(__file__).parent.parent / "conv2d-1x1-cat"
ROWS_BRICK = Path(__file__).parent.parent / "conv2d-3x3-u8"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


g = _load(BRICK / "golden.py", "conv1x1_golden")
rows = _load(ROWS_BRICK / "golden.py", "conv3x3_golden")

NSRC, CSRC, COUT, H, PRE, SHIFT = 4, 48, 48, 4, 3, 14


def case(name, width, valid_lo=0, valid_hi=None, seed=0):
    hi = width if valid_hi is None else valid_hi
    rng = np.random.default_rng(seed)
    xs = [rng.integers(-128, 128, size=(CSRC, H, width), dtype=np.int64).astype(np.int8)
          for _ in range(NSRC)]
    w = rng.integers(-127, 128, size=(COUT, NSRC * CSRC), dtype=np.int64).astype(np.int8)
    b = rng.integers(-(1 << 15), 1 << 15, size=(COUT,), dtype=np.int64).astype(np.int32)
    mult = rng.integers(64, 256, size=COUT, dtype=np.int64)
    ref = g.conv1x1_cat_ref(xs, w, b, SHIFT, valid_lo, hi, pre_shift=PRE, mult=mult)
    wp = width + 2 * rows.PAD
    src_rows = [rows.pack_rows(a).reshape(H, -1) for a in xs]
    tiles = np.stack([np.concatenate([r[y] for r in src_rows]) for y in range(H)])
    step = wp * CSRC
    shim = bricklib.GEN / f"{name}_shim.cc"
    sym = f"c1cat_{name}"
    shim.write_text(
        f'#include <stdint.h>\n#include "{BRICK / "conv1x1_cat.cc"}"\n'
        f'extern "C" void {sym}(int8_t *t, int8_t *p, int8_t *o) {{\n'
        f'  conv1x1_cat_i8(t, t + {step}, t + {2 * step}, t + {3 * step}, p, o, {width},'
        f' {PRE}, {SHIFT}, {valid_lo}, {hi});\n}}\n')
    exp = np.stack([ref[:, y, :] for y in range(H)])

    def unpack(dev):
        return np.stack([rows.unpack_rows(r, COUT, 1, width)[:, 0, :] for r in dev])

    res = bricklib.verify_streamed(
        name, shim, sym, tiles, wp * COUT, g.pack_params(w, b, NSRC, mult), unpack, exp, gate=0.0,
        in_dt=np.int8, out_dt=np.int8, resident_dt=np.int8,
        compile_flags=[f"-DCONV1X1_NSRC={NSRC}", f"-DCONV1X1_CSRC={CSRC}",
                       f"-DCONV1X1_COUT={COUT}"])
    got = np.asarray(res["got"]).astype(np.int64)
    mism = int((got != exp.astype(np.int64)).sum())
    res["mismatches"] = mism
    res["ok"] = bool(res["ok"] and mism == 0)
    res["status"] = "PASS" if res["ok"] else f"FAIL({mism}/{exp.size} mismatched)"
    print(f"[{name:22s}] exact: {exp.size - mism}/{exp.size} -> {res['status']}")
    return res


CASES = [("c1_w32", 32), ("c1_w48_mask", 48, 5, 43, 1)]

if __name__ == "__main__":
    results = []
    for c in CASES:
        try:
            results.append(case(*c))
        except Exception as e:
            print(f"[{c[0]:22s}] ERROR: {e}")
            results.append(dict(name=c[0], status="ERROR", ok=False, err=str(e)))
    passed = sum(1 for r in results if r.get("ok"))
    print(f"conv2d-1x1-cat: {passed}/{len(results)} PASS")
    print("JSON " + json.dumps([{k: v for k, v in r.items() if k != 'got'} for r in results]))
    sys.exit(0 if passed == len(results) else 1)
