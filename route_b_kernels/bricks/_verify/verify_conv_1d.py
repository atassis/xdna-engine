#!/usr/bin/env python3
"""Device rel-L2 verify for the causal dilated conv-1d brick. Gate 3e-2. Run under the device lock.

Gated at the REAL residual-unit shapes of codec decoder stage 4 -- c_in 96, c_out 96, k 7, t 32 --
with weights, bias and dilation taken from the S2-Pro GGUF, at all three dilations the codec uses
(1, 3, 9 for .block.2/.3/.4). Dilation is the whole point of this brick, so gating one value of it
would not mean much.

Streamed rail, one output channel per tile: [c_out, c_in, k] f32 is 258 KB and no core tile holds
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
import codec_paths  # noqa: E402
import gguf_extract as gx  # noqa: E402

GGUF = Path(codec_paths.gguf())
import os
# CV_T overrides t: this brick has NO static scratches, so sweeping it measures what the streamed
# rail alone can deliver, independent of any kernel's own buffers. Measured: t=64 passes at
# rel-L2 4.592e-07, t=128 exceeds available memory.
C_IN, C_OUT, K, T = 96, 96, 7, int(os.environ.get('CV_T', 32))
GATE = 3e-2
# .block.2/.3/.4 of a decoder stage are the three residual units, at dilations 1/3/9.
UNITS = [("c.decoder.model.4.block.2.block.1.conv", 1),
         ("c.decoder.model.4.block.3.block.1.conv", 3),
         ("c.decoder.model.4.block.4.block.1.conv", 9)]

# ADDITIVE: the same three dilated tensors plus a real 1x1 (block.3.conv, a residual unit's own
# pointwise conv -- what the S2 quantizer's projections use), now against conv_1d_causal_core_vec
# instead of conv_1d_causal_core. Same golden, same GATE, same activation -- only the kernel symbol
# and k (for the 1x1 case) differ, so a failure here isolates to the vector core, never to the
# tensors or the harness. conv_1d_causal_core_vec requires t a multiple of 16 (see its own comment);
# CV_T's default (32) and its one other measured value (64) both satisfy that.
assert T % 16 == 0, f"conv_1d_causal_core_vec requires t a multiple of 16, got {T}"
VEC_UNITS = [("c.decoder.model.4.block.2.block.1.conv", 7, 1),
             ("c.decoder.model.4.block.3.block.1.conv", 7, 3),
             ("c.decoder.model.4.block.4.block.1.conv", 7, 9),
             ("c.decoder.model.4.block.2.block.3.conv", 1, 1)]

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
    # [c_out, c_in, k]: ggml_conv_1d's layout, the TRANSPOSE of conv_transpose_1d's.
    assert w.shape == (C_OUT, C_IN, K), f"{tensor}: weight shape {w.shape}"

    ref = golden.conv_1d_causal_ref(x, w, bias, dilation)

    TILE = C_IN * K + 1
    tiles = np.zeros((C_OUT, TILE), np.float32)
    for co in range(C_OUT):
        tiles[co, :C_IN * K] = w[co].reshape(-1)   # already contiguous [c_in, k]
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

for tensor, k, dilation in VEC_UNITS:
    w = gx.load(str(GGUF), f"{tensor}.weight").astype(np.float32)
    bias = gx.load(str(GGUF), f"{tensor}.bias").astype(np.float32).reshape(-1)
    assert w.shape == (C_OUT, C_IN, k), f"{tensor}: weight shape {w.shape}"

    ref = golden.conv_1d_causal_ref(x, w, bias, dilation)

    tile_w = C_IN * k + 1
    tiles = np.zeros((C_OUT, tile_w), np.float32)
    for co in range(C_OUT):
        tiles[co, :C_IN * k] = w[co].reshape(-1)
        tiles[co, C_IN * k] = bias[co]

    _cb = int(time.time() * 1000) % 10**9
    shim = bricklib.GEN / f"conv_1d_vec_d{dilation}_k{k}_shim.cc"
    shim.write_text(
        f"// AUTO-GENERATED verify shim for the conv-1d VECTORISED brick, dilation {dilation} "
        f"k {k}. cb {_cb}\n"
        "#include <stdint.h>\n"
        f'#include "{BRICK / "conv_1d.cc"}"\n'
        f'extern "C" void conv_1d_vec_verify_d{dilation}_k{k}(float *wtile, float *resident, '
        f'float *out) {{\n'
        f"  route_b_bricks::conv_1d_causal_core_vec<16>(resident, wtile, wtile[{C_IN * k}], out,\n"
        f"                                              {C_IN}, {k}, {T}, {dilation});\n"
        "}\n"
    )

    res = bricklib.verify_streamed(
        name=f"conv_1d_vec_dil{dilation}_k{k}",
        shim=shim,
        symbol=f"conv_1d_vec_verify_d{dilation}_k{k}",
        in_tiles=tiles,
        out_tile_numel=T,
        resident=resident,
        unpack=lambda d: np.asarray(d).reshape(C_OUT, T),
        golden=ref,
        gate=GATE,
        in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
    )
    got = np.asarray(res["got"], np.float32)
    print(f"  VEC dilation {dilation} k={k}: rel-L2 {res['rel_l2']:.3e}  "
          f"max abs err {float(np.max(np.abs(got - ref))):.3e}  "
          f"range [{got.min():+.4f}, {got.max():+.4f}]  {res['status']}")
    ok = ok and res["ok"]


# ADDITIVE: same VEC_UNITS tensors/shapes, now against conv_1d_causal_core_vec_acc<16, NACC> --
# the multi-accumulator form that splits the c_in axis into NACC independent lanes to break the
# single serial accumulator recurrence conv_1d_causal_core_vec pays for (see that kernel's own
# comment in conv_1d.cc for the measured cause: an unpipelined loop plus a ~6-cycle-latency
# emulated f32 vector MAC, per llvm-aie's AIE2PGenSchedule.td). Swept at NACC 2/4/8 -- the only
# values the kernel accepts (it static_asserts NACC a power of two) -- so device numbers can settle
# what compile-time evidence alone could not: NACC=2 compiles with ZERO vector-register spill
# (frame stays at the single-accumulator kernel's own 64 bytes), while NACC=4 and NACC=8 both spill
# 8 vector (not accfloat) registers into a 704- and 960-byte frame respectively, INSIDE the hot
# (oc, ci0, j) loop (confirmed via -fstack-size-section + llvm-readelf and reading the compiled
# asm for "Folded Spill/Reload" tags on vst/vlda) -- more parallelism bought with register pressure
# whose net effect on cycles is unmeasured. All four units have c_in=96, divisible by 2/4/8, so
# this gate does NOT exercise the remainder-channel path (c_in % NACC != 0); that path is unverified
# by any device run and rests on code inspection + the shared conv1d_vec_acc_tap logic only.
ACC_UNITS = [("c.decoder.model.4.block.2.block.1.conv", 7, 1),
             ("c.decoder.model.4.block.3.block.1.conv", 7, 3),
             ("c.decoder.model.4.block.4.block.1.conv", 7, 9),
             ("c.decoder.model.4.block.2.block.3.conv", 1, 1)]
NACC_VALUES = (2, 4, 8)

for nacc in NACC_VALUES:
    for tensor, k, dilation in ACC_UNITS:
        w = gx.load(str(GGUF), f"{tensor}.weight").astype(np.float32)
        bias = gx.load(str(GGUF), f"{tensor}.bias").astype(np.float32).reshape(-1)
        assert w.shape == (C_OUT, C_IN, k), f"{tensor}: weight shape {w.shape}"

        ref = golden.conv_1d_causal_ref(x, w, bias, dilation)

        tile_w = C_IN * k + 1
        tiles = np.zeros((C_OUT, tile_w), np.float32)
        for co in range(C_OUT):
            tiles[co, :C_IN * k] = w[co].reshape(-1)
            tiles[co, C_IN * k] = bias[co]

        _cb = int(time.time() * 1000) % 10**9
        shim = bricklib.GEN / f"conv_1d_acc{nacc}_d{dilation}_k{k}_shim.cc"
        shim.write_text(
            f"// AUTO-GENERATED verify shim for the conv-1d MULTI-ACCUMULATOR brick, NACC {nacc} "
            f"dilation {dilation} k {k}. cb {_cb}\n"
            "#include <stdint.h>\n"
            f'#include "{BRICK / "conv_1d.cc"}"\n'
            f'extern "C" void conv_1d_acc{nacc}_verify_d{dilation}_k{k}(float *wtile, '
            f'float *resident, float *out) {{\n'
            f"  route_b_bricks::conv_1d_causal_core_vec_acc<16, {nacc}>(resident, wtile, "
            f"wtile[{C_IN * k}], out,\n"
            f"                                                          {C_IN}, {k}, {T}, {dilation});\n"
            "}\n"
        )

        res = bricklib.verify_streamed(
            name=f"conv_1d_acc{nacc}_dil{dilation}_k{k}",
            shim=shim,
            symbol=f"conv_1d_acc{nacc}_verify_d{dilation}_k{k}",
            in_tiles=tiles,
            out_tile_numel=T,
            resident=resident,
            unpack=lambda d: np.asarray(d).reshape(C_OUT, T),
            golden=ref,
            gate=GATE,
            in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
        )
        got = np.asarray(res["got"], np.float32)
        print(f"  ACC NACC={nacc} dilation {dilation} k={k}: rel-L2 {res['rel_l2']:.3e}  "
              f"max abs err {float(np.max(np.abs(got - ref))):.3e}  "
              f"range [{got.min():+.4f}, {got.max():+.4f}]  {res['status']}")
        ok = ok and res["ok"]

assert ok, "conv_1d device gate failed"
print("PASS")
