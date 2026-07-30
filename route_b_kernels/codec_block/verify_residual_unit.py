#!/usr/bin/env python3
"""Device rel-L2 verify for one fused codec residual unit at REAL checkpoint shapes. Gate 3e-2.

    y = x + conv1d_1x1( snake( conv1d_dilated( snake(x, a0), w1, d ), a2 ), w3 )

s2.cpp/src/s2_codec.cpp:381-393. Gated at stage 4's channel shapes (c 96, k 7) with every tensor from
the S2-Pro GGUF, at all three dilations a decoder stage uses (1, 3, 9 for .block.2/.3/.4). The
reference composes the two already-device-green bricks on the host, so what this gates is the
FUSION -- the two-phase stream, the L1-resident intermediate, and the residual add.

t is 16, and that is an L1 bound rather than a choice: the unit needs two [c, t] f32 scratches
resident at once (the activation and the intermediate) and at t=32 that overflowed .bss by 10944 B.
KNOWN COVERAGE GAP as a result: a dilated tap reaches back (k-1)*d, which is 54 at dilation 9, so
at t=16 only the near taps of the dilation-3 and dilation-9 units see real input and the rest read
causal padding. The arithmetic those units DO perform is gated exactly; their long taps are not
exercised here. Real decoder inputs are far longer than 16, so closing that needs the blocked form
that carries left context between tiles -- tracked separately, not papered over by loosening this
gate.

The unit runs as ONE stream of 2*c_out tiles because the 1x1 conv needs every channel of the
dilated conv's output: phase A (first c_out tiles) fills an L1 intermediate, phase B (second c_out)
consumes it and adds the residual. Only phase B's output tiles carry y.

Run under the device lock with PYTHONPATH at instance 7d8a49b5d7a0.
"""
import importlib.util
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

GGUF = Path(
    "$WORKSPACE/s2.cpp/models/s2-pro-q6_k.gguf")
C, K, T = 96, 7, 16
GATE = 3e-2
UNITS = [("c.decoder.model.4.block.2", 1),
         ("c.decoder.model.4.block.3", 3),
         ("c.decoder.model.4.block.4", 9)]


def _golden(sub, name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "route_b_kernels" / "bricks" / sub / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


snake_g = _golden("snake", "snake_golden")
conv_g = _golden("conv-1d", "conv1d_golden")

rng = np.random.default_rng(0)
x = (rng.standard_normal((C, T)).astype(np.float32) * 0.5)

ok = True
for prefix, dilation in UNITS:
    a0 = gx.load(str(GGUF), f"{prefix}.block.0.alpha").astype(np.float32).reshape(-1)
    w1 = gx.load(str(GGUF), f"{prefix}.block.1.conv.weight").astype(np.float32)
    b1 = gx.load(str(GGUF), f"{prefix}.block.1.conv.bias").astype(np.float32).reshape(-1)
    a2 = gx.load(str(GGUF), f"{prefix}.block.2.alpha").astype(np.float32).reshape(-1)
    w3 = gx.load(str(GGUF), f"{prefix}.block.3.conv.weight").astype(np.float32)
    b3 = gx.load(str(GGUF), f"{prefix}.block.3.conv.bias").astype(np.float32).reshape(-1)
    assert w1.shape == (C, C, K) and w3.shape == (C, C, 1), f"{prefix}: {w1.shape} {w3.shape}"

    h = conv_g.conv_1d_causal_ref(snake_g.snake_ref(x, a0), w1, b1, dilation)
    y = conv_g.conv_1d_causal_ref(snake_g.snake_ref(h, a2), w3, b3, 1)
    ref = (x + y).astype(np.float32)

    # One stream, two phases. Slots after the weights: bias, phase, activate-now, channel.
    TILE = C * K + 4
    tiles = np.zeros((2 * C, TILE), np.float32)
    for co in range(C):                                     # phase A: the dilated conv
        tiles[co, :C * K] = w1[:, co, :].reshape(-1)
        tiles[co, C * K] = b1[co]
        tiles[co, C * K + 3] = co
    tiles[0, C * K + 2] = 1.0                               # activate x with alpha0 on tile 0
    for co in range(C):                                     # phase B: the 1x1 conv + residual
        r = C + co
        tiles[r, :C * 1] = w3[:, co, :].reshape(-1)
        tiles[r, C * K] = b3[co]
        tiles[r, C * K + 1] = 1.0
        tiles[r, C * K + 3] = co
    tiles[C, C * K + 2] = 1.0                               # activate h with alpha2 on the first B

    resident = np.concatenate([a0, a2, x.reshape(-1)]).astype(np.float32)

    _cb = int(time.time() * 1000) % 10**9
    shim = bricklib.GEN / f"residual_unit_d{dilation}_shim.cc"
    shim.write_text(
        f"// AUTO-GENERATED verify shim, residual unit dilation {dilation}. cachebust {_cb}\n"
        "#include <stdint.h>\n"
        f'#include "{(HERE / "residual_unit.cc").resolve()}"\n'
        f'extern "C" void residual_unit_verify_d{dilation}('
        "float *tile, float *resident, float *out) {\n"
        "  using namespace route_b_bricks;\n"
        "  ROUTE_B_RESIDUAL_UNIT_BODY(tile, resident, out)\n"
        "}\n"
    )

    res = bricklib.verify_streamed(
        name=f"residual_unit_dil{dilation}",
        shim=shim,
        symbol=f"residual_unit_verify_d{dilation}",
        in_tiles=tiles,
        out_tile_numel=T,
        resident=resident,
        # Only phase B carries y; phase A's tiles are the L1 fill and are defined-but-ignored.
        unpack=lambda d: np.asarray(d)[C:].reshape(C, T),
        golden=ref,
        gate=GATE,
        in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
        compile_flags=[f"-DRU_C={C}", f"-DRU_T={T}", f"-DRU_K={K}",
                       f"-DRU_DILATION={dilation}"],
    )
    got = np.asarray(res["got"], np.float32)
    # A unit whose conv output vanished would still track x closely, since y is a correction on top
    # of the residual path -- so check the correction is actually present.
    corr = float(np.linalg.norm((got - x).ravel()) / np.linalg.norm(x.ravel()))
    print(f"  dilation {dilation}: rel-L2 {res['rel_l2']:.3e}  "
          f"max abs err {float(np.max(np.abs(got - ref))):.3e}  "
          f"|y|/|x| {corr:.3e}  {res['status']}")
    ok = ok and res["ok"]

assert ok, "residual unit device gate failed"
print("PASS")
