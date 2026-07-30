#!/usr/bin/env python3
"""Gate the fused upsample stage on a WINDOW of the real decoder stream. Gate 3e-2.

Two things this adds over verify_upsample_stage.py, which feeds random normal activations:

1. REAL ACTIVATIONS. The arithmetic was already gated; the regime was not. snake's argument fold is
   magnitude-sensitive by construction (alpha*x spans many periods, and that is exactly where an f32
   fold loses digits), so a kernel can be correct on N(0,1) and drift on the tensor the stage really
   receives. The input here is stage 4's actual input, taken from the whole-decoder reference that
   reproduces codec_audio.bin at 3.919e-04.

2. LEFT CONTEXT, which is the mechanism the time-blocked driver is built on. A window taken out of
   the middle of a stream is NOT the same as a stream that starts there: the kernel treats window
   index 0 as if zeros preceded it, while the real stream has signal. For this stage the dependency
   is short -- out[p] draws on input ti in [(p-k+1)/stride, p/stride], so only outputs p < k-1 = 3
   reach before the window -- so L=2 input samples of carry is enough and the first L*stride outputs
   are discarded. Proving that here, at a stage whose reach is small, is what makes the same argument
   trustworthy at the residual units, where the reach is 54.

If this passes, a window is bit-comparable to the unblocked result and the blocked driver is sound
for this op. If it fails only in the first few samples, the carry is too short. Those are different
diagnoses, so the report below separates them.

    python3 verify_stage4_realdata.py <stage_io.npz>

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
PREFIX = "c.decoder.model.4"
C_IN, C_OUT, K, STRIDE = 192, 96, 4, 2
L = 2                      # input samples of left context; out[p] reaches back only for p < k-1 = 3
T_WIN = 14                 # useful window; L + T_WIN must be the kernel's STAGE_T
STAGE_T = L + T_WIN        # 16, one whole 16-float vector per channel row
T0 = 20000                 # well inside the stream, so the window is genuinely mid-stream
GATE = 3e-2

npz = np.load(sys.argv[1] if len(sys.argv) > 1 else "stage_io.npz")
x_full = npz["stage4_in"]              # [192, 56320]
up_full = npz["stage4_upsample"]       # [96, 112640]
print(f"real stage-4 in {x_full.shape}, upsample out {up_full.shape}")
print(f"window: input [{T0 - L}, {T0 + T_WIN}) = {STAGE_T} cols, "
      f"compare outputs [{T0 * STRIDE}, {(T0 + T_WIN) * STRIDE}) = {T_WIN * STRIDE} cols")

x_win = np.ascontiguousarray(x_full[:, T0 - L:T0 + T_WIN]).astype(np.float32)
assert x_win.shape == (C_IN, STAGE_T)
ref_win = up_full[:, T0 * STRIDE:(T0 + T_WIN) * STRIDE].astype(np.float32)

alpha = gx.load(GGUF, f"{PREFIX}.block.0.alpha").astype(np.float32).reshape(-1)
w = gx.load(GGUF, f"{PREFIX}.block.1.conv.weight").astype(np.float32)
bias = gx.load(GGUF, f"{PREFIX}.block.1.conv.bias").astype(np.float32).reshape(-1)
assert w.shape == (C_IN, C_OUT, K)     # conv_transpose_1d layout, [c_in, c_out, k]

TILE = C_IN * K + 2
tiles = np.zeros((C_OUT, TILE), np.float32)
for co in range(C_OUT):
    tiles[co, :C_IN * K] = w[:, co, :].reshape(-1)
    tiles[co, C_IN * K] = bias[co]
tiles[0, C_IN * K + 1] = 1.0
resident = np.concatenate([alpha, x_win.reshape(-1)]).astype(np.float32)

_cb = int(time.time() * 1000) % 10**9
shim = bricklib.GEN / "stage4_realdata_shim.cc"
shim.write_text(
    f"// AUTO-GENERATED verify shim, stage-4 upsample on real data. cachebust {_cb}\n"
    "#include <stdint.h>\n"
    f'#include "{(HERE / "upsample_stage.cc").resolve()}"\n'
    'extern "C" void stage4_real_verify(float *wtile, float *resident, float *out) {\n'
    "  route_b_bricks::upsample_stage_core(wtile, resident, out);\n"
    "}\n"
)

res = bricklib.verify_streamed(
    name="stage4_realdata",
    shim=shim,
    symbol="stage4_real_verify",
    in_tiles=tiles,
    out_tile_numel=STAGE_T * STRIDE,
    resident=resident,
    # Drop the first L*stride outputs: those are the ones that would have reached before the window.
    unpack=lambda d: np.asarray(d).reshape(C_OUT, STAGE_T * STRIDE)[:, L * STRIDE:],
    golden=ref_win,
    gate=GATE,
    in_dt=np.float32, out_dt=np.float32, resident_dt=np.float32,
    compile_flags=[f"-DSTAGE_C_IN={C_IN}", f"-DSTAGE_T={STAGE_T}",
                   f"-DSTAGE_K={K}", f"-DSTAGE_STRIDE={STRIDE}"],
)

got = np.asarray(res["got"], np.float32)
err = np.abs(got - ref_win)
print(f"  device vs real stage-4 upsample rel-L2: {res['rel_l2']:.3e}")
print(f"  max abs err                           : {float(err.max()):.3e}")
print(f"  activation magnitude (real)           : |x| max {float(np.abs(x_win).max()):.3f}, "
      f"alpha*|x| max {float((np.abs(x_win) * alpha.reshape(-1, 1)).max()):.3f} "
      f"({float((np.abs(x_win) * alpha.reshape(-1, 1)).max()) / (2 * np.pi):.1f} periods)")
# Separate "the carry is too short" from "the kernel is wrong": a short carry shows up as error
# concentrated in the FIRST columns and clean everywhere after.
head, tail = err[:, :4], err[:, 4:]
print(f"  err in first 4 cols / rest            : {float(head.max()):.3e} / {float(tail.max()):.3e}")
assert res["ok"], f"stage-4 real-data gate failed: {res['status']} rel_l2={res['rel_l2']}"
print("PASS")
