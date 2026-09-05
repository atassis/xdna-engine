#!/usr/bin/env python3
"""What exactly is wrong in ARM0? Shape the defect so it can be filed.

The isolation probe put the trigger on `float s = beta * kv[i]` -- a scalar f32 multiply whose
operand is a lane EXTRACTED from an aie::vector. Extract-and-broadcast alone is clean, and a vector
mul is clean, so this is a narrow codegen defect rather than a precision effect (rel-L2 9.2e-02
against 2.9e-08 for both controls).

This recovers the effective per-row scale the device actually used. Each output row is
  S_out[i,:] = alpha*S_in[i,:] + scale_i * err[:]
so scale_i is recoverable by least squares against err, and comparing it to the intended beta*k[i]
says whether the multiply is dropped, mis-rounded, reading a wrong lane, or something else.

Run:  ./run.sh probe_iso_extract_dump.py
"""
from pathlib import Path

import numpy as np

import bricklib

DK = DV = 32
CC = Path(__file__).parent / "gen" / "iso_extract_roundtrip.cc"
SHIM = ('extern "C" void iso_verify(float* packed, float* s_in, float* s_out) {\n'
        '  iso_step_impl(packed, s_in, s_out);\n'
        '}\n')

rng = np.random.default_rng(0)
k = rng.standard_normal(DK).astype(np.float32)
err = rng.standard_normal(DV).astype(np.float32)
s_in = rng.standard_normal((DK, DV)).astype(np.float32)
alpha, beta = np.float32(0.9), np.float32(0.7)
packed = np.concatenate([k, err, np.array([alpha, beta], np.float32)])
golden = (alpha * s_in + (beta * k)[:, None] * err[None, :]).astype(np.float32)

res = bricklib.verify_oneshot(
    name="iso_dump", brick_cc=CC, shim_body=SHIM, symbol="iso_verify",
    inputs=[(packed, np.float32), (s_in.reshape(-1), np.float32)],
    out_numel=DK * DV, out_shape=(DK, DV),
    unpack=lambda flat: np.asarray(flat, np.float32).reshape(DK, DV),
    golden=golden, gate=3e-2,
    compile_flags=[f"-DISO_DK={DK}", f"-DISO_DV={DV}", "-DISO_ARM=0"],
    out_dt=np.float32,
)
got = np.asarray(res.get("got"), np.float32).reshape(DK, DV)

# recover the scale the device actually applied on each row
resid = got - alpha * s_in                      # should equal scale_i * err
scale_dev = (resid @ err) / float(err @ err)    # least squares per row
want = (beta * k).astype(np.float32)

np.set_printoptions(precision=5, suppress=True, linewidth=150)
bad = np.abs(scale_dev - want) > 1e-4 * np.maximum(1.0, np.abs(want))
print(f"status={res.get('status')} rel_l2={res.get('rel_l2'):.3e}   rows wrong: {int(bad.sum())}/{DK}")
print("\nrow : k[i]        want=beta*k[i]   device scale    ratio dev/want")
for i in range(DK):
    flag = "  <-- WRONG" if bad[i] else ""
    r = scale_dev[i] / want[i] if abs(want[i]) > 1e-9 else float("nan")
    print(f"{i:3d} : {k[i]:+9.5f}   {want[i]:+9.5f}     {scale_dev[i]:+9.5f}   {r:+8.5f}{flag}")

print(f"\nwrong-row indices: {np.nonzero(bad)[0].tolist()}")
if bad.any():
    print(f"ratio dev/want on wrong rows: min {(scale_dev/want)[bad].min():.5f} "
          f"max {(scale_dev/want)[bad].max():.5f}")
    print(f"is the device scale just k[i] (i.e. beta dropped)? "
          f"{np.allclose(scale_dev[bad], k[bad], rtol=1e-4)}")
    print(f"is it k[i] * beta^2? {np.allclose(scale_dev[bad], (beta*beta*k)[bad], rtol=1e-4)}")
