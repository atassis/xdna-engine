#!/usr/bin/env python3
"""Device rel-L2 verify for one fused codec upsample stage at REAL checkpoint shapes. Gate 3e-2.

Stage 4 of the fish-audio DAC decoder: c_in 192 -> c_out 96, k 4, stride 2, weights and per-channel
alpha pulled straight from the S2-Pro GGUF -- no synthetic tensors anywhere in the path. The
reference is the two already-device-green bricks composed on the host (snake_ref then
conv_transpose_1d_ref), so this gates the FUSION and the real shapes, not the arithmetic, which
each brick already gated on its own.

Delivery follows the weights, which decide it: [c_in, c_out, k] f32 is 294 KB and no core tile
holds it, while the [c_in, t] activation is 12 KB and does. So the activation is the resident
operand and the stream is one output channel's weights per tile, each tile carrying its own bias so
the kernel never needs to know which call it is on. snake's output stays in an L1 scratch across
all 96 conv calls -- the intermediate never goes to DDR, which is the property the stage exists to
demonstrate.

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
import codec_paths  # noqa: E402
import gguf_extract as gx  # noqa: E402

GGUF = Path(codec_paths.gguf())
PREFIX = "c.decoder.model.4"
C_IN, C_OUT, K, STRIDE, T = 192, 96, 4, 2, 16
OUT_LEN = T * STRIDE                       # crop_right == stride, so out_len is exactly t*stride
GATE = 3e-2


def _golden(sub, name):
    p = ROOT / "route_b_kernels" / "bricks" / sub / "golden.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


snake_g = _golden("snake", "snake_golden")
ct_g = _golden("conv-transpose-1d", "ct_golden")

alpha = gx.load(str(GGUF), f"{PREFIX}.block.0.alpha").astype(np.float32).reshape(-1)
w = gx.load(str(GGUF), f"{PREFIX}.block.1.conv.weight").astype(np.float32)
bias = gx.load(str(GGUF), f"{PREFIX}.block.1.conv.bias").astype(np.float32).reshape(-1)
assert w.shape == (C_IN, C_OUT, K), f"weight shape {w.shape} != {(C_IN, C_OUT, K)}"
assert alpha.shape == (C_IN,) and bias.shape == (C_OUT,)

rng = np.random.default_rng(0)
x = (rng.standard_normal((C_IN, T)).astype(np.float32) * 0.5)

ref = ct_g.conv_transpose_1d_ref(snake_g.snake_ref(x, alpha), w, bias, STRIDE, crop_right=STRIDE)
assert ref.shape == (C_OUT, OUT_LEN)

# One streamed tile per output channel: that channel's weights over every input channel, then its
# bias, then a first-tile flag. Laid out [ci*K + j] so the kernel walks it contiguously per input
# channel. The flag tells the kernel to compute snake into its L1 scratch on tile 0 only -- it is
# data rather than a static bool because an uninitialised static read non-zero on device and the
# stage returned NaN.
TILE = C_IN * K + 2
tiles = np.zeros((C_OUT, TILE), np.float32)
for co in range(C_OUT):
    tiles[co, :C_IN * K] = w[:, co, :].reshape(-1)
    tiles[co, C_IN * K] = bias[co]
tiles[0, C_IN * K + 1] = 1.0

resident = np.concatenate([alpha, x.reshape(-1)]).astype(np.float32)

_cb = int(time.time() * 1000) % 10**9
shim = bricklib.GEN / "upsample_stage_shim.cc"
shim.write_text(
    f"// AUTO-GENERATED verify shim for the fused codec upsample stage. cachebust {_cb}\n"
    "#include <stdint.h>\n"
    f'#include "{(HERE / "upsample_stage.cc").resolve()}"\n'
    'extern "C" void upsample_stage_verify(float *wtile, float *resident, float *out) {\n'
    "  route_b_bricks::upsample_stage_core(wtile, resident, out);\n"
    "}\n"
)

res = bricklib.verify_streamed(
    name="upsample_stage4",
    shim=shim,
    symbol="upsample_stage_verify",
    in_tiles=tiles,
    out_tile_numel=OUT_LEN,
    resident=resident,
    unpack=lambda d: np.asarray(d).reshape(C_OUT, OUT_LEN),
    golden=ref,
    gate=GATE,
    in_dt=np.float32,
    out_dt=np.float32,
    resident_dt=np.float32,
    compile_flags=[f"-DSTAGE_C_IN={C_IN}", f"-DSTAGE_T={T}",
                   f"-DSTAGE_K={K}", f"-DSTAGE_STRIDE={STRIDE}"],
)

got = np.asarray(res["got"], np.float32)
print(f"  stage-4 device vs host-composed ref rel-L2: {res['rel_l2']:.3e}")
print(f"  max abs err                              : {float(np.max(np.abs(got - ref))):.3e}")
print(f"  out shape {got.shape} range [{got.min():+.4f}, {got.max():+.4f}]")
# A stage that silently produced only its bias would still look plausible in rel-L2 if the
# activations were small, so check the conv actually contributed.
print(f"  per-channel spread (min over co)         : {float(np.ptp(got, axis=1).min()):.3e}")
assert res["ok"], f"upsample stage gate failed: {res['status']} rel_l2={res['rel_l2']}"
print("PASS")
