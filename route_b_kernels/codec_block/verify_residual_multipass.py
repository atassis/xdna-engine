#!/usr/bin/env python3
"""A whole residual unit on device as FOUR passes over already-green bricks, windowed. Gate 3e-2.

    snake(x, a0) -> conv_dilated(w1, d) -> snake(h, a2) -> conv_1x1(w3) + x

The fused kernel cannot hold a dilation-9 window: it needs a [c, t] intermediate in core L1 and the
static budget at t=64 is ~7.8 KB against the 12288 B one bf16 [96,64] buffer costs. Unfused, no pass
carries a static buffer, so every one of them fits at t=64 -- which is how dilation 9 becomes
expressible at all. Fusion was the right first move (it proved the arithmetic and kept the
intermediate off DDR) and is the wrong final one at the window lengths blocking needs.

Each pass is a separate device dispatch; the host carries the intermediate between them. That is the
cost being measured, not hidden: three DDR round-trips per unit. Whether to buy them back with a
MemTile staging path in the rail is the next decision, and it should be made against this number.

WHAT THE GATE COMPARES, and this is the point of the exercise. Not the device against a host replay
of the same window -- both would share the same zero-padding at the window's left edge and agree
while being equally wrong. It compares against the TRUE UNBLOCKED STREAM, `stage4_res*` from the
whole-decoder reference that reproduces codec_audio.bin. A window that matches that is a window that
can be stitched.

    python3 verify_residual_multipass.py <stage_io.npz>

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import importlib.util
import os
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
SNAKE = (ROOT / "route_b_kernels" / "bricks" / "snake" / "snake.cc").resolve()
CONV = (ROOT / "route_b_kernels" / "bricks" / "conv-1d" / "conv_1d.cc").resolve()
C, K = 96, 7
T = int(os.environ.get("MP_T", 64))
T0 = 40000
GATE = 3e-2

UNITS = [("c.decoder.model.4.block.2", 1, "stage4_upsample", "stage4_res2"),
         ("c.decoder.model.4.block.3", 3, "stage4_res2", "stage4_res3"),
         ("c.decoder.model.4.block.4", 9, "stage4_res3", "stage4_res4")]

_cb = int(time.time() * 1000) % 10**9


def snake_pass(tag, x, alpha):
    """Per-channel snake. alpha rides in the tile (one row per tile), so no resident operand and no
    per-channel shim rebuild -- the pure-buffer ABI carries no scalars."""
    tiles = np.zeros((C, T + 1), np.float32)
    tiles[:, :T] = x
    tiles[:, T] = alpha
    shim = bricklib.GEN / f"mp_snake_{tag}_shim.cc"
    shim.write_text(
        f"// multipass snake {tag} cb {_cb}\n#include <stdint.h>\n"
        f'#include "{SNAKE}"\n'
        f'extern "C" void mp_snake_{tag}(float *tile, float *out) {{\n'
        f"  snake_f32(tile, out, {T}, tile[{T}]);\n}}\n")
    r = bricklib.verify_streamed(
        name=f"snake_{tag}", shim=shim, symbol=f"mp_snake_{tag}",
        in_tiles=tiles, out_tile_numel=T, resident=None,
        unpack=lambda d: np.asarray(d).reshape(C, T),
        golden=np.zeros((C, T)), gate=np.inf,          # not gated here; the unit's output is
        in_dt=np.float32, out_dt=np.float32)
    return np.asarray(r["got"], np.float32)


def conv_pass(tag, s, w, bias, k, dil, x_add=None):
    """One output channel per tile. When x_add is given the residual row rides in the tile too --
    a single row, which is why the second pass needs no second resident operand."""
    wide = C * k + 1 + (T if x_add is not None else 0)
    tiles = np.zeros((C, wide), np.float32)
    for co in range(C):
        tiles[co, :C * k] = w[co].reshape(-1)
        tiles[co, C * k] = bias[co]
        if x_add is not None:
            tiles[co, C * k + 1:] = x_add[co]
    body = (f"  route_b_bricks::conv_1d_causal_core(resident, tile, tile[{C * k}], out,\n"
            f"                                      {C}, {k}, {T}, {dil});\n")
    if x_add is not None:
        body += (f"  for (int p = 0; p < {T}; p++) out[p] += tile[{C * k + 1} + p];\n")
    shim = bricklib.GEN / f"mp_conv_{tag}_shim.cc"
    shim.write_text(
        f"// multipass conv {tag} cb {_cb}\n#include <stdint.h>\n"
        f'#include "{CONV}"\n'
        f'extern "C" void mp_conv_{tag}(float *tile, float *resident, float *out) {{\n{body}}}\n')
    r = bricklib.verify_streamed(
        name=f"conv_{tag}", shim=shim, symbol=f"mp_conv_{tag}",
        in_tiles=tiles, out_tile_numel=T, resident=s.reshape(-1).astype(np.float32),
        unpack=lambda d: np.asarray(d).reshape(C, T),
        golden=np.zeros((C, T)), gate=np.inf,
        in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32)
    return np.asarray(r["got"], np.float32)


npz = np.load(sys.argv[1] if len(sys.argv) > 1 else "stage_io.npz")
print(f"multipass residual unit, t={T}, 4 device dispatches per unit")
ok = True

for prefix, dil, in_key, out_key in UNITS:
    ctx = (K - 1) * dil
    win = T - ctx
    if win <= 0:
        print(f"  dilation {dil}: context {ctx} >= t {T} -- cannot window")
        ok = False
        continue

    a0 = gx.load(GGUF, f"{prefix}.block.0.alpha").astype(np.float32).reshape(-1)
    w1 = gx.load(GGUF, f"{prefix}.block.1.conv.weight").astype(np.float32)
    b1 = gx.load(GGUF, f"{prefix}.block.1.conv.bias").astype(np.float32).reshape(-1)
    a2 = gx.load(GGUF, f"{prefix}.block.2.alpha").astype(np.float32).reshape(-1)
    w3 = gx.load(GGUF, f"{prefix}.block.3.conv.weight").astype(np.float32)
    b3 = gx.load(GGUF, f"{prefix}.block.3.conv.bias").astype(np.float32).reshape(-1)

    x_win = np.ascontiguousarray(npz[in_key][:, T0 - ctx:T0 + win]).astype(np.float32)
    truth = npz[out_key][:, T0:T0 + win].astype(np.float32)

    s1 = snake_pass(f"a_d{dil}", x_win, a0)
    h = conv_pass(f"dil_d{dil}", s1, w1, b1, K, dil)
    s2 = snake_pass(f"b_d{dil}", h, a2)
    y = conv_pass(f"1x1_d{dil}", s2, w3, b3, 1, 1, x_add=x_win)

    got = y[:, ctx:]
    num = np.linalg.norm((got - truth).astype(np.float64))
    den = np.linalg.norm(truth.astype(np.float64))
    rl2 = float(num / den)
    err = np.abs(got - truth)
    h4 = min(4, win)
    status = "PASS" if rl2 <= GATE else "FAIL"
    print(f"  dilation {dil}: context {ctx:2d} window {win:2d}  rel-L2 vs TRUE stream {rl2:.3e}  "
          f"max abs err {float(err.max()):.3e}  "
          f"err head/rest {float(err[:, :h4].max()):.3e}/{float(err[:, h4:].max()):.3e}  {status}")
    ok = ok and rl2 <= GATE

assert ok, "multipass residual unit gate failed"
print("PASS")
