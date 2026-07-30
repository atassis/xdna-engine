#!/usr/bin/env python3
"""Gate the fused residual unit on a WINDOW of the real decoder stream. Gate 3e-2.

Extends the windowing proof from the upsample stage (reach 3, trivially satisfied) to the op where
it actually bites. A residual unit is snake -> conv(k=7, dilation d) -> snake -> conv(k=1) -> add,
so its output at t reaches back (7-1)*d = 6d samples: 6 at dilation 1, 18 at dilation 3, 54 at
dilation 9.

WHAT IS AND IS NOT COVERED HERE, stated up front because the gap is the point.

t is capped at 16, and that is MEASURED, not estimated. An arithmetic estimate said 37 and was
wrong: the two static [c, t] f32 scratches share the data region with the objectFIFO buffers, and
the link fails outright rather than degrading. Overflow of `.bss`, by t:

    t=24  scratch 18432 B  overflowed by  4800 B
    t=32  scratch 24576 B  overflowed by 10944 B
    t=48  scratch 36864 B  overflowed by 16640 B

The overflow is not linear in t, so the allocator is doing more than "statics against a fixed
budget" and the exact model is not worth further link attempts. What matters is the measurement:
t=16 links, t=24 does not.

At t=16, against a reach of (k-1)*dilation:

    dilation 1   context  6  -> window 10 of 16   COVERED
    dilation 3   context 18  -> does not fit      NOT COVERED
    dilation 9   context 54  -> does not fit      NOT COVERED

So this closes ONE of the three on real data. That is less than it looked like it would be, and the
reason is worth stating plainly: this decomposition cannot window a dilated residual unit at all
beyond dilation 1, so the c_in-chunked decomposition is not an optimisation for stages 1-3, it is
what makes time-blocking possible anywhere in the residual stack. Do not read a pass here as
covering dilation 3 or 9.

The tell that a carry is sufficient is not the aggregate rel-L2 -- a short carry can hide in it --
but that the error is FLAT across the window rather than concentrated in the first columns. Both are
reported.

    python3 verify_residual_realdata.py <stage_io.npz>

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "route_b_kernels" / "bricks" / "_verify"))
sys.path.insert(0, str(ROOT / "scripts"))

import bricklib  # noqa: E402
import gguf_extract as gx  # noqa: E402

GGUF = ("$WORKSPACE/"
        "s2.cpp/models/s2-pro-q6_k.gguf")
C, K = 96, 7
import os
STAGE_T = int(os.environ.get("RU_T", 16))   # measured .bss budget caps this; see below
T0 = 40000                 # mid-stream
GATE = 3e-2

# (tensor prefix, dilation, npz input key, npz output key)
UNITS = [("c.decoder.model.4.block.2", 1, "stage4_upsample", "stage4_res2"),
         ("c.decoder.model.4.block.3", 3, "stage4_res2", "stage4_res3"),
         ("c.decoder.model.4.block.4", 9, "stage4_res3", "stage4_res4")]

npz = np.load(sys.argv[1] if len(sys.argv) > 1 else "stage_io.npz")
ok = True
skipped = []

for prefix, dil, in_key, out_key in UNITS:
    ctx = (K - 1) * dil
    win = STAGE_T - ctx
    if win <= 0:
        skipped.append((dil, ctx))
        print(f"  dilation {dil}: context {ctx} >= t {STAGE_T} -- SKIPPED, needs c_in chunking")
        continue

    x_full, ref_full = npz[in_key], npz[out_key]
    x_win = np.ascontiguousarray(x_full[:, T0 - ctx:T0 + win]).astype(np.float32)
    assert x_win.shape == (C, STAGE_T), x_win.shape
    ref_win = ref_full[:, T0:T0 + win].astype(np.float32)

    a0 = gx.load(GGUF, f"{prefix}.block.0.alpha").astype(np.float32).reshape(-1)
    w1 = gx.load(GGUF, f"{prefix}.block.1.conv.weight").astype(np.float32)
    b1 = gx.load(GGUF, f"{prefix}.block.1.conv.bias").astype(np.float32).reshape(-1)
    a2 = gx.load(GGUF, f"{prefix}.block.2.alpha").astype(np.float32).reshape(-1)
    w3 = gx.load(GGUF, f"{prefix}.block.3.conv.weight").astype(np.float32)
    b3 = gx.load(GGUF, f"{prefix}.block.3.conv.bias").astype(np.float32).reshape(-1)

    TILE = C * K + 4
    tiles = np.zeros((2 * C, TILE), np.float32)
    for co in range(C):
        tiles[co, :C * K] = w1[co].reshape(-1)
        tiles[co, C * K] = b1[co]
        tiles[co, C * K + 3] = co
    tiles[0, C * K + 2] = 1.0
    for co in range(C):
        r = C + co
        tiles[r, :C] = w3[co].reshape(-1)
        tiles[r, C * K] = b3[co]
        tiles[r, C * K + 1] = 1.0
        tiles[r, C * K + 3] = co
    tiles[C, C * K + 2] = 1.0

    resident = np.concatenate([a0, a2, x_win.reshape(-1)]).astype(np.float32)

    _cb = int(time.time() * 1000) % 10**9
    shim = bricklib.GEN / f"residual_real_d{dil}_shim.cc"
    shim.write_text(
        f"// AUTO-GENERATED verify shim, residual unit on real data, dilation {dil}. cb {_cb}\n"
        "#include <stdint.h>\n"
        f'#include "{(HERE / os.environ.get("RU_SRC", "residual_unit.cc")).resolve()}"\n'
        f'extern "C" void residual_real_d{dil}(float *tile, float *resident, float *out) {{\n'
        "  using namespace route_b_bricks;\n"
        f"  {os.environ.get('RU_MACRO', 'ROUTE_B_RESIDUAL_UNIT_BODY')}(tile, resident, out)\n"
        "}\n"
    )

    res = bricklib.verify_streamed(
        name=f"residual_real_dil{dil}",
        shim=shim,
        symbol=f"residual_real_d{dil}",
        in_tiles=tiles,
        out_tile_numel=STAGE_T,
        resident=resident,
        # Phase B carries y; drop the leading `ctx` columns, which reached before the window.
        unpack=lambda d, c=ctx: np.asarray(d)[C:].reshape(C, STAGE_T)[:, c:],
        golden=ref_win,
        gate=GATE,
        in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
        compile_flags=[*os.environ.get("RU_EXTRA","").split(), f"-DRU_C={C}", f"-DRU_T={STAGE_T}", f"-DRU_K={K}",
                       f"-DRU_DILATION={dil}"],
    )
    got = np.asarray(res["got"], np.float32)
    err = np.abs(got - ref_win)
    h = min(4, win)
    print(f"  dilation {dil}: context {ctx:2d} window {win:2d}  rel-L2 {res['rel_l2']:.3e}  "
          f"max abs err {float(err.max()):.3e}  "
          f"err head/rest {float(err[:, :h].max()):.3e}/{float(err[:, h:].max()):.3e}  "
          f"{res['status']}")
    ok = ok and res["ok"]

if skipped:
    print(f"  NOT COVERED: dilation(s) {[d for d, _ in skipped]} "
          f"(context {[c for _, c in skipped]} > t={STAGE_T}) -- blocked on c_in chunking")
assert ok, "residual unit real-data windowed gate failed"
print("PASS (for the dilations that fit)")
