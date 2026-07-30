#!/usr/bin/env python3
"""Device rel-L2 verify for the causal dilated conv-1d brick. Gate 3e-2. Run under the device lock.

Gated at the REAL residual-unit shapes of codec decoder stage 4 -- c_in 96, c_out 96, k 7, t 32 --
with weights, bias and dilation taken from the S2-Pro GGUF, at all three dilations the codec uses
(1, 3, 9 for .block.2/.3/.4). Dilation is the whole point of this brick, so gating one value of it
would not mean much.

Streamed rail, one output channel per tile: [c_in, c_out, k] f32 is 258 KB and no core tile holds
it, while the [c_in, t] activation is 12 KB and does. Each tile carries that channel's weights and
its own bias, so the kernel never needs to know which call it is on. Same split as the fused
upsample stage, which is deliberate -- the residual unit composes these two.

Run with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
BRICK = (HERE.parent / "conv-1d").resolve()
sys.path.insert(0, str(ROOT / "scripts"))
import gguf_extract as gx  # noqa: E402

GGUF = Path(
    "$WORKSPACE/s2.cpp/models/s2-pro-q6_k.gguf")
C_IN, C_OUT, K, T = 96, 96, 7, 32
GATE = 3e-2
# .block.2/.3/.4 of a decoder stage are the three residual units, at dilations 1/3/9.
UNITS = [("c.decoder.model.4.block.2.block.1.conv", 1),
         ("c.decoder.model.4.block.3.block.1.conv", 3),
         ("c.decoder.model.4.block.4.block.1.conv", 9)]

spec = importlib.util.spec_from_file_location("conv1d_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(0)
x = (rng.standard_normal((C_IN, T)).astype(np.float32) * 0.5)
resident = x.reshape(-1).astype(np.float32)

ok = True
for tensor, dilation in UNITS:
    w = gx.load(str(GGUF), f"{tensor}.weight").astype(np.float32)
    bias = gx.load(str(GGUF), f"{tensor}.bias").astype(np.float32).reshape(-1)
    assert w.shape == (C_IN, C_OUT, K), f"{tensor}: weight shape {w.shape}"

    ref = golden.conv_1d_causal_ref(x, w, bias, dilation)

    TILE = C_IN * K + 1
    tiles = np.zeros((C_OUT, TILE), np.float32)
    for co in range(C_OUT):
        tiles[co, :C_IN * K] = w[:, co, :].reshape(-1)
        tiles[co, C_IN * K] = bias[co]

    _cb = int(time.time() * 1000) % 10**9
    shim = bricklib.GEN / f"conv_1d_d{dilation}_shim.cc"
    shim.write_text(
        f"// AUTO-GENERATED verify shim for the conv-1d brick, dilation {dilation}. cb {_cb}\n"
        "#include <stdint.h>\n"
        f'#include "{BRICK / "conv_1d.cc"}"\n'
        f'extern "C" void conv_1d_verify_d{dilation}(float *wtile, float *resident, float *out) {{\n'
        f"  route_b_bricks::conv_1d_causal_core(resident, wtile, wtile[{C_IN * K}], out,\n"
        f"                                      {C_IN}, {K}, {T}, {dilation});\n"
        "}\n"
    )

    res = bricklib.verify_streamed(
        name=f"conv_1d_dil{dilation}",
        shim=shim,
        symbol=f"conv_1d_verify_d{dilation}",
        in_tiles=tiles,
        out_tile_numel=T,
        resident=resident,
        unpack=lambda d: np.asarray(d).reshape(C_OUT, T),
        golden=ref,
        gate=GATE,
        in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
    )
    got = np.asarray(res["got"], np.float32)
    print(f"  dilation {dilation}: rel-L2 {res['rel_l2']:.3e}  "
          f"max abs err {float(np.max(np.abs(got - ref))):.3e}  "
          f"range [{got.min():+.4f}, {got.max():+.4f}]  {res['status']}")
    ok = ok and res["ok"]

assert ok, "conv_1d device gate failed"
print("PASS")
