#!/usr/bin/env python3
"""Device rel-L2 verify for the conv-transpose-1d brick. Gate 3e-2. Run under the device lock.

ONE input buffer, ONE output buffer, both 64 floats -- the only configuration measured to transfer
intact on this harness:
  * 3 separate inputs is rejected outright (a core tile has 2 input / 2 output DMA channels).
  * With 2 inputs, in0 arrives as ZEROS and in1 arrives with a single element replicated
    (probe_ct1d_inputs.py). So the multi-input oneshot path does not deliver.
  * An 8-float (32 B) buffer arrives as zeros even when it is the only input.
So x, w and bias are packed into one 64-float buffer and sliced in the shim. verify_rmsnorm packs
gamma|beta the same way for the same DMA reason; this just goes one step further.

stride 4 exercises the misalignment every codec rate ([8,8,4,2]) has.
"""
import importlib.util
import time
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = (HERE.parent / "conv-transpose-1d").resolve()

C_IN, C_OUT, K, T, STRIDE = 1, 2, 8, 4, 4
OUT_LEN = (T - 1) * STRIDE + K                 # 20
X_OFF, W_OFF = 0, C_IN * T                     # 0, 4
B_OFF = W_OFF + C_IN * C_OUT * K               # 20
PACK = 64
GATE = 3e-2

spec = importlib.util.spec_from_file_location("ct1d_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(0)
x = rng.standard_normal((C_IN, T)).astype(np.float32)
w = (rng.standard_normal((C_IN, C_OUT, K)).astype(np.float32) * 0.5)
b = (rng.standard_normal(C_OUT).astype(np.float32) * 0.1)
ref = golden.conv_transpose_1d_ref(x, w, b, STRIDE)

packed = np.zeros(PACK, np.float32)
packed[X_OFF:X_OFF + x.size] = x.ravel()
packed[W_OFF:W_OFF + w.size] = w.ravel()
packed[B_OFF:B_OFF + b.size] = b

_cb = int(time.time() * 1000) % 10**9
res = bricklib.verify_oneshot(
    name="conv_transpose_1d",
    brick_cc=BRICK / "conv_transpose_1d.cc",
    shim_body=(
        f"// cachebust {_cb}\n"
        'extern "C" void ct1d_verify(float *p, float *out) {\n'
        f"  conv_transpose_1d_f32(p + {X_OFF}, p + {W_OFF}, p + {B_OFF}, out,\n"
        f"                        {C_IN}, {C_OUT}, {K}, {T}, {STRIDE}, {OUT_LEN});\n"
        "}\n"
    ),
    symbol="ct1d_verify",
    inputs=[(packed, np.float32)],
    out_numel=PACK,
    out_shape=(PACK,),
    unpack=lambda d: np.asarray(d)[:C_OUT * OUT_LEN].reshape(C_OUT, OUT_LEN),
    golden=ref,
    gate=GATE,
    out_dt=np.float32,
)
got = np.asarray(res["got"], np.float32)
print(f"  device vs ref rel-L2: {res['rel_l2']:.3e}")
print(f"  max abs err         : {float(np.max(np.abs(got - ref))):.3e}")
assert res["ok"], f"conv_transpose_1d device gate failed: {res['status']} rel_l2={res['rel_l2']}"
print("PASS")
