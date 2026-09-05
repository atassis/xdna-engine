#!/usr/bin/env python3
"""Isolate the gatedeltanet trigger: scalar OP, or the vector-extract round trip?

The 3-part bisect showed `float bk = beta * k[i]` (k an aie::vector) must become a vector mul or
gatedeltanet is NaN, while a standalone scalar f32 multiply was already shown fine on device. The
distinguishing feature is that the scalar operand is EXTRACTED FROM A VECTOR and BROADCAST BACK.

Three arms, differing only in how the per-row scale reaches aie::broadcast:

  ARM 0  extract + SCALAR MUL + broadcast   -- the failing shape
  ARM 1  one vector mul, then extract + broadcast   -- the fix
  ARM 2  extract + broadcast, NO scalar op          -- the discriminator (host sends beta=1)

  0 dirty, 1 clean, 2 CLEAN  => the scalar OP on an extracted lane is the trigger.
  0 dirty, 1 clean, 2 DIRTY  => the extract/broadcast round trip alone is enough; the multiply is
                               incidental, and the defect is broader than the gatedeltanet fix implies.
  all clean                  => it does not reproduce standalone, so gatedeltanet's context (loop
                               nesting, register pressure, the surrounding recurrence) is required,
                               and this is not yet a filable repro.

Run:  ./run.sh probe_iso_extract_roundtrip.py
"""
from pathlib import Path

import numpy as np

import bricklib

DK = DV = 32
CC = Path(__file__).parent / "gen" / "iso_extract_roundtrip.cc"

LABEL = {0: "extract + SCALAR MUL + broadcast (failing shape)",
         1: "vector mul, then extract + broadcast (the fix)",
         2: "extract + broadcast, NO scalar op   (discriminator)"}

SHIM = (
    'extern "C" void iso_verify(float* packed, float* s_in, float* s_out) {\n'
    '  iso_step_impl(packed, s_in, s_out);\n'
    '}\n'
)

rng = np.random.default_rng(0)
k = rng.standard_normal(DK).astype(np.float32)
err = rng.standard_normal(DV).astype(np.float32)
s_in = rng.standard_normal((DK, DV)).astype(np.float32)
alpha = np.float32(0.9)

print(f"===== extract-round-trip isolation (DK={DK} DV={DV}) =====", flush=True)
for arm in (0, 1, 2):
    beta = np.float32(1.0) if arm == 2 else np.float32(0.7)
    packed = np.concatenate([k, err, np.array([alpha, beta], np.float32)])
    # golden: S[i,:] = alpha*S[i,:] + (beta*k[i])*err[:]   (arm 2 has beta = 1)
    golden = (alpha * s_in + (beta * k)[:, None] * err[None, :]).astype(np.float32)
    try:
        res = bricklib.verify_oneshot(
            name=f"iso_arm{arm}",
            brick_cc=CC,
            shim_body=SHIM,
            symbol="iso_verify",
            inputs=[(packed, np.float32), (s_in.reshape(-1), np.float32)],
            out_numel=DK * DV,
            out_shape=(DK, DV),
            unpack=lambda flat: np.asarray(flat, np.float32).reshape(DK, DV),
            golden=golden,
            gate=3e-2,
            compile_flags=[f"-DISO_DK={DK}", f"-DISO_DV={DV}", f"-DISO_ARM={arm}"],
            out_dt=np.float32,
        )
        got = np.asarray(res.get("got"), np.float32)
        nf = int((~np.isfinite(got)).sum())
        rel = res.get("rel_l2")
        rel_s = f"{rel:.3e}" if isinstance(rel, float) else str(rel)
        print(f"  ARM{arm} {LABEL[arm]:50s} status={res.get('status')} rel_l2={rel_s} "
              f"non-finite {nf}/{got.size}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  ARM{arm} {LABEL[arm]:50s} ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
