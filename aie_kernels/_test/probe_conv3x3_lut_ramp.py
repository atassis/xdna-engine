#!/usr/bin/env python3
"""Ramp-table probe for conv3x3_i8_lut: with table[k] = k - 128 the gathered value IS the slot read,
so comparing it against the pre-LUT conv output reveals the key->slot map. Prints, exits 0."""
import numpy as np
import verify_conv3x3_u8 as v

g = v.g
orig_pack = g.pack_lut
captured = {}


def ramp_case():
    rng = np.random.default_rng(4)
    width, H = 64, v.H
    x = rng.integers(-128, 128, size=(v.CIN, H, width), dtype=np.int64).astype(np.int8)
    w = rng.integers(-127, 128, size=(v.COUT, v.CIN, 3, 3), dtype=np.int64).astype(np.int8)
    b = rng.integers(-(1 << 15), 1 << 15, size=(v.COUT,), dtype=np.int64).astype(np.int32)
    ref = g.conv3x3_u8_ref(x, w, b, v.SHIFT, 0, width, signed=True)
    params = g.pack_params(w, b)
    inc = g.lut_inc(np.arange(256) - 128, v.bricklib.GEN / "c3lramp_lut.inc")
    wp = width + 2 * g.PAD
    rows = g.pack_rows(x).reshape(H, -1)
    tiles = np.stack([np.concatenate([rows[y - 1], rows[y], rows[y + 1]]) for y in range(1, H - 1)])
    shim = v.bricklib.GEN / "c3lramp_shim.cc"
    sym = "conv3x3_lutramp"
    shim.write_text(
        f'#include <stdint.h>\n#include "{v.BRICK / "conv3x3_u8.cc"}"\n'
        f'extern "C" void {sym}(int8_t *t, int8_t *p, int8_t *o) {{\n'
        f'  conv3x3_i8_lut(t, t + {wp * v.CIN}, t + {2 * wp * v.CIN}, p, o, {width}, 1, {v.SHIFT}, 0, {width});\n}}\n')
    res = v.bricklib.verify_streamed(
        "c3lramp", shim, sym, tiles, wp * v.COUT, params,
        lambda dev: dev, np.zeros(1), gate=0.0, in_dt=np.int8, out_dt=np.int8, resident_dt=np.int8,
        compile_flags=[f"-DCONV3X3_CIN={v.CIN}", f"-DCONV3X3_COUT={v.COUT}",
                       f'-DCONV3X3_LUT_INC="{inc}"'], stack_size=2048)
    dev = np.asarray(res["got"]).astype(np.int64)[0]            # first row, raw padded layout
    exp = g.pack_rows(ref[:, 1:2, :].astype(np.int8)).reshape(-1).astype(np.int64)  # pre-LUT, same layout
    print("identity-table exact:", int((dev == exp).sum()), "/", dev.size)
    # map: for the first 64-lane vector of real data (skip the 64-byte left margin)
    o = 64
    print("expected keys :", exp[o:o + 32].tolist())
    print("gathered vals :", dev[o:o + 32].tolist())


ramp_case()
