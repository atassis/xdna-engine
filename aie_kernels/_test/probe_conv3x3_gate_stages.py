#!/usr/bin/env python3
"""Stage-by-stage probe for conv3x3_i8_gate (SPAN's block tail), plus the LUT path on the same data.

Each run swaps in a table / constants that make one stage observable:
  sum     table = 1, gc = 1, gs2 = 0          -> out = sat8(sum)
  two     table = 2                           -> out = sat8(2 * sum)          (multiply path)
  gc      table = 1, gc = 100, gs2 = 9        -> out = the final requant
  lutonly conv3x3_i8_lut, identity table      -> out = conv output (LUT path, same data)
  idt     identity table                      -> out = sat8(sum * c)          (gather on the gate path)
Prints match counts, exits 0.
"""
import numpy as np

import bricklib
import verify_conv3x3_u8 as v

g = v.g
CIN, COUT, H, SH = v.CIN, v.COUT, v.H, v.SHIFT
rng = np.random.default_rng(6)
width = 64
wp = width + 16
x = rng.integers(-128, 128, size=(CIN, H, width)).astype(np.int8)
w = rng.integers(-127, 128, size=(COUT, CIN, 3, 3)).astype(np.int8)
b = rng.integers(-(1 << 15), 1 << 15, size=(COUT,)).astype(np.int32)
c = g.conv3x3_u8_ref(x, w, b, SH, 0, width, signed=True)
xb = rng.integers(-128, 128, size=(COUT, H, width)).astype(np.int8)
rows, xr = g.pack_rows(x).reshape(H, -1), g.pack_rows(xb).reshape(H, -1)
tiles = np.stack([np.concatenate([rows[y - 1], rows[y], rows[y + 1], xr[y]]) for y in range(1, H - 1)])
cc = c[:, 1:H - 1, :].transpose(1, 0, 2).astype(np.int64)
xx = xb[:, 1:H - 1, :].transpose(1, 0, 2).astype(np.int64)
s = np.clip(g.rshift_round(cc * 64 + xx * 45, 6), -128, 127)
ks = np.arange(256) - 128


def run(tag, table, ga=64, gb=45, gs1=6, gc=1, gs2=0, extra=(), lutonly=False):
    inc = g.lut_inc(table, bricklib.GEN / f"gd_{tag}.inc")
    sh = bricklib.GEN / f"gd_{tag}_shim.cc"
    sym = f"gatedbg_{tag}"
    a0, a1, a2, a3 = 0, wp * CIN, 2 * wp * CIN, 3 * wp * CIN
    call = (f"conv3x3_i8_lut(t + {a0}, t + {a1}, t + {a2}, p, o, {width}, 1, {SH}, 0, {width});"
            if lutonly else
            f"conv3x3_i8_gate(t + {a0}, t + {a1}, t + {a2}, t + {a3}, p, o, {width}, 1, {SH}, 0, "
            f"{width}, {ga}, {gb}, {gs1}, {gc}, {gs2});")
    sh.write_text(f'#include <stdint.h>\n#include "{v.BRICK / "conv3x3_u8.cc"}"\n'
                  f'extern "C" void {sym}(int8_t *t, int8_t *p, int8_t *o) {{ {call} }}\n')
    r = bricklib.verify_streamed(
        tag, sh, sym, tiles, wp * COUT, g.pack_params(w, b),
        lambda d: np.stack([g.unpack_rows(q, COUT, 1, width)[:, 0, :] for q in d]), np.zeros(1),
        gate=0.0, in_dt=np.int8, out_dt=np.int8, resident_dt=np.int8,
        compile_flags=[f"-DCONV3X3_CIN={CIN}", f"-DCONV3X3_COUT={COUT}",
                       f'-DCONV3X3_LUT_INC="{inc}"', *extra],
        stack_size=3072)
    return np.asarray(r["got"]).astype(np.int64)


def report(label, got, exp):
    m = got != exp
    print(f"{label:34s} match {int((~m).sum())} / {got.size}", flush=True)
    if m.any():
        print(f"   got {got[m][:8].tolist()}  exp {exp[m][:8].tolist()}  c {cc[m][:8].tolist()}")


ones = np.ones(256, np.int64)
report("sum (table 1)", run("sum", ones), s)
report("two (table 2)", run("two", np.full(256, 2)), np.clip(2 * s, -128, 127))
report("gc (table 1, gc 100 >> 9)", run("gc", ones, gc=100, gs2=9),
       np.clip(g.rshift_round(s * 100, 9), -128, 127))
report("lutonly identity (LUT path)", run("lutonly", ks, lutonly=True), cc)
report("idt identity (gate path)", run("idt", ks), np.clip(s * cc, -128, 127))
# shift interplay: with ga=1, gb=0, gs1=0 the sum is c itself and no nonzero shift precedes fetch()
report("idt, gs1=0 (sum = c)", run("idt0", ks, ga=1, gb=0, gs1=0), np.clip(cc * cc, -128, 127))
report("idt, gs1=0, gc=1 gs2=2", run("idt2", ks, ga=1, gb=0, gs1=0, gc=1, gs2=2),
       np.clip(g.rshift_round(np.clip(cc * cc, -32768, 32767), 2), -128, 127))
