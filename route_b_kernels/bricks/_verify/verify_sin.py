#!/usr/bin/env python3
"""Device rel-L2 verify for the sin brick. Gate 3e-2. Run under the device lock.

Gates against np.sin (ground truth), and separately reports the device-vs-emulation delta so a
kernel bug is distinguishable from a golden bug. bricklib runs the design twice and reports
run-to-run determinism, which is the CLFLUSH read-race guard.
"""
import importlib.util
from pathlib import Path

import numpy as np

import bricklib

HERE = Path(__file__).parent
BRICK = HERE.parent / "sin"

N = 1024   # 4 KB in + 4 KB out; double-buffered that is 16 KB of the core tile 64 KB L1
GATE = 3e-2

spec = importlib.util.spec_from_file_location("sin_golden", BRICK / "golden.py")
golden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(golden)

rng = np.random.default_rng(0)
# +/-64 spans about 10 periods so the argument fold is actually exercised, not just the polynomial.
x = (rng.random(N, dtype=np.float32) * 128.0 - 64.0).astype(np.float32)
ref = golden.sin_ref(x)

res = bricklib.verify_oneshot(
    name="sin",
    brick_cc=BRICK / "sin.cc",
    shim_body=(
        'extern "C" void sin_verify(float *x, float *out) {\n'
        f"  sin_f32(x, out, {N});\n"
        "}\n"
    ),
    symbol="sin_verify",
    inputs=[(x, np.float32)],
    out_numel=N,
    out_shape=(N,),
    unpack=lambda d: d,
    golden=ref,
    gate=GATE,
    out_dt=np.float32,
)

got = res["got"]
print(f"  device vs np.sin    rel-L2: {res['rel_l2']:.3e}")
print(f"  device vs emulation rel-L2: {golden.rel_l2(got, golden.sin_poly_emulated(x)):.3e}")
print(f"  max abs err vs np.sin     : {float(np.max(np.abs(got - ref))):.3e}")
assert res["ok"], f"sin device gate failed: {res['status']} rel_l2={res['rel_l2']}"
print("PASS")
