#!/usr/bin/env python3
"""Gate the causal dilated conv-1d on REAL windowed data at t=64, including dilation 9. Gate 3e-2.

This is the result the whole time-blocking question turns on. The FUSED residual unit cannot hold a
dilation-9 window in core L1 at any arrangement measured -- its intermediate needs a [c, t] buffer
and the static budget at t=64 is only ~7.8 KB (see residual_unit_bf16.cc's header for the frontier).
But the conv-1d brick carries NO static buffer, and t=64 is comfortably inside what the streamed
rail delivers. Dilation 9 reaches back (k-1)*9 = 54, so t=64 leaves a 10-sample window: small, but
non-zero, and that is the difference between "time-blocking is possible" and "it is not".

Each unit's first conv is gated on its OWN real input, snake-activated on the host. Snake is gated
separately and is green; what is under test here is the dilated convolution at a real magnitude
distribution and a real left-context boundary.

The pass criterion is not the aggregate rel-L2 alone. A left context one sample too short still
produces a plausible number -- it shows up as error concentrated in the FIRST output columns. Both
are reported, and a flat profile is what says the carry is right.

    python3 verify_conv_1d_realdata.py <stage_io.npz>

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
BRICK = (HERE.parent / "conv-1d").resolve()
sys.path.insert(0, str(ROOT / "scripts"))
import codec_paths  # noqa: E402
import gguf_extract as gx  # noqa: E402

GGUF = codec_paths.gguf()
C, K = 96, 7
T = int(os.environ.get("CV_T", 64))
T0 = 40000
GATE = 3e-2

# (prefix, dilation, npz key holding that unit's INPUT)
UNITS = [("c.decoder.model.4.block.2", 1, "stage4_upsample"),
         ("c.decoder.model.4.block.3", 3, "stage4_res2"),
         ("c.decoder.model.4.block.4", 9, "stage4_res3")]


def _golden(sub, name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "route_b_kernels" / "bricks" / sub / "golden.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


snake_g = _golden("snake", "snake_golden")
conv_g = _golden("conv-1d", "conv1d_golden")

npz = np.load(sys.argv[1] if len(sys.argv) > 1 else "stage_io.npz")
print(f"t={T}; dilation d reaches back (k-1)*d, leaving a window of t-(k-1)*d")
ok = True

for prefix, dil, in_key in UNITS:
    ctx = (K - 1) * dil
    win = T - ctx
    if win <= 0:
        print(f"  dilation {dil}: context {ctx} >= t {T} -- cannot window")
        ok = False
        continue

    a0 = gx.load(GGUF, f"{prefix}.block.0.alpha").astype(np.float32).reshape(-1)
    w = gx.load(GGUF, f"{prefix}.block.1.conv.weight").astype(np.float32)
    bias = gx.load(GGUF, f"{prefix}.block.1.conv.bias").astype(np.float32).reshape(-1)
    assert w.shape == (C, C, K)

    # The unit's real input, snake-activated: this conv's true operand mid-stream.
    x_stream = npz[in_key]
    s_win = snake_g.snake_ref(
        np.ascontiguousarray(x_stream[:, T0 - ctx:T0 + win]).astype(np.float32), a0)
    assert s_win.shape == (C, T)

    # Reference over the same window. Its first `ctx` columns see zero-padding where the real
    # stream has signal, exactly as the device does -- so both are wrong there in the same way, and
    # comparing only columns >= ctx is what makes the window equivalent to the unblocked result.
    ref_full = conv_g.conv_1d_causal_ref(s_win, w, bias, dil)
    ref_win = ref_full[:, ctx:]

    tiles = np.zeros((C, C * K + 1), np.float32)
    for co in range(C):
        tiles[co, :C * K] = w[co].reshape(-1)
        tiles[co, C * K] = bias[co]

    _cb = int(time.time() * 1000) % 10**9
    shim = bricklib.GEN / f"conv1d_real_d{dil}_shim.cc"
    shim.write_text(
        f"// AUTO-GENERATED verify shim, conv-1d on real data, dilation {dil}, t={T}. cb {_cb}\n"
        "#include <stdint.h>\n"
        f'#include "{BRICK / "conv_1d.cc"}"\n'
        f'extern "C" void conv1d_real_d{dil}(float *wtile, float *resident, float *out) {{\n'
        f"  route_b_bricks::conv_1d_causal_core(resident, wtile, wtile[{C * K}], out,\n"
        f"                                      {C}, {K}, {T}, {dil});\n"
        "}\n"
    )

    res = bricklib.verify_streamed(
        name=f"conv1d_real_d{dil}_t{T}",
        shim=shim,
        symbol=f"conv1d_real_d{dil}",
        in_tiles=tiles,
        out_tile_numel=T,
        resident=s_win.reshape(-1).astype(np.float32),
        unpack=lambda d, c=ctx: np.asarray(d).reshape(C, T)[:, c:],
        golden=ref_win,
        gate=GATE,
        in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
    )
    got = np.asarray(res["got"], np.float32)
    err = np.abs(got - ref_win)
    h = min(4, win)
    print(f"  dilation {dil}: context {ctx:2d} window {win:2d}  rel-L2 {res['rel_l2']:.3e}  "
          f"max abs err {float(err.max()):.3e}  "
          f"err head/rest {float(err[:, :h].max()):.3e}/{float(err[:, h:].max()):.3e}  "
          f"{res['status']}")
    ok = ok and res["ok"]

assert ok, "conv-1d real-data windowed gate failed"
print("PASS")
