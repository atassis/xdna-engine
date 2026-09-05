#!/usr/bin/env python3
"""Sharpen the extract-scalar-mul repro before it is filed.

Established: a scalar f32 mul on a lane extracted from an aie::vector corrupts EXACTLY row 0 of the
loop; 31/32 rows exact, run2run 0. Two things the previous run could not say:

  PART A -- WHICH TERM. The scale recovery assumed alpha was correct and attributed all of row 0's
  error to beta*k[0]. Setting alpha=0 removes the alpha*S[i,:] term entirely, so S_out[i,:] becomes
  exactly (beta*k[i])*err[:]. If row 0 is STILL wrong, the defect is in the beta*k[0] scalar mul.
  If row 0 becomes CLEAN, the defect was in the alpha term and the scalar mul is a bystander.

  PART B -- DOES IT TRACK ROW 0 OR THE LOOP COUNT. Sweep DK. If only row 0 is ever wrong regardless
  of trip count, it is a loop-prologue defect. If the wrong row moves or the count grows, it is
  something else (unroll boundary, pipeline depth).

Run:  ./run.sh probe_iso_extract_sharpen.py
"""
from pathlib import Path

import numpy as np

import bricklib

DV = 32
CC = Path(__file__).parent / "gen" / "iso_extract_roundtrip.cc"
SHIM = ('extern "C" void iso_verify(float* packed, float* s_in, float* s_out) {\n'
        '  iso_step_impl(packed, s_in, s_out);\n'
        '}\n')


def run(dk, alpha, beta, arm, tag):
    rng = np.random.default_rng(0)
    k = rng.standard_normal(dk).astype(np.float32)
    err = rng.standard_normal(DV).astype(np.float32)
    s_in = rng.standard_normal((dk, DV)).astype(np.float32)
    a, b = np.float32(alpha), np.float32(beta)
    # match the kernel's 32-float padded field offsets (see ISO_ERR_OFF): without the pad,
    # aie::load_v<DV>(err) is misaligned whenever DK is not a multiple of 32.
    al = lambda x: ((x + 31) // 32) * 32
    packed = np.zeros(al(dk) + al(DV) + 32, np.float32)
    packed[:dk] = k
    packed[al(dk):al(dk) + DV] = err
    packed[al(dk) + al(DV):al(dk) + al(DV) + 2] = [a, b]
    golden = (a * s_in + (b * k)[:, None] * err[None, :]).astype(np.float32)
    res = bricklib.verify_oneshot(
        name=tag, brick_cc=CC, shim_body=SHIM, symbol="iso_verify",
        inputs=[(packed, np.float32), (s_in.reshape(-1), np.float32)],
        out_numel=dk * DV, out_shape=(dk, DV),
        unpack=lambda flat: np.asarray(flat, np.float32).reshape(dk, DV),
        golden=golden, gate=3e-2,
        compile_flags=[f"-DISO_DK={dk}", f"-DISO_DV={DV}", f"-DISO_ARM={arm}"],
        out_dt=np.float32,
    )
    got = np.asarray(res.get("got"), np.float32).reshape(dk, DV)
    # per-row wrongness, computed directly against the golden (no scale inference needed)
    rowerr = np.linalg.norm(got - golden, axis=1) / np.maximum(np.linalg.norm(golden, axis=1), 1e-12)
    bad = np.nonzero(rowerr > 1e-4)[0]
    return res, got, golden, k, rowerr, bad


print("===== PART A: which term carries row 0's error? (DK=32, the only valid point) =====", flush=True)
for alpha, arm, note in ((0.9, 0, "alpha=0.9 arm0 (both terms live)"),
                         (0.0, 1, "alpha=0   arm1 CONTROL"),
                         (0.0, 0, "alpha=0   arm0 (alpha term REMOVED)")):
    res, got, golden, k, rowerr, bad = run(32, alpha, 0.7, arm, f"iso_a{int(alpha*10)}_m{arm}")
    rel = res.get("rel_l2")
    rel_s = f"{rel:.3e}" if isinstance(rel, float) else str(rel)
    print(f"  {note:38s} status={res.get('status')} rel_l2={rel_s}  "
          f"wrong rows {bad.tolist()}  row0 rel-err {rowerr[0]:.3e}", flush=True)
    if alpha == 0.0 and arm == 0 and len(bad):
        # with alpha=0 the row IS the scale, so read it straight off
        dev0 = got[0] @ golden[0] / max(float(golden[0] @ golden[0]), 1e-30)
        print(f"      alpha=0 -> row0 is purely (beta*k[0])*err; device/want ratio = {dev0:.5f}",
              flush=True)

print("\n===== PART B: DK sweep WITH the arm-1 control at every point =====", flush=True)
print("  (arm1 = vector mul, the known-good spelling. If arm1 is also wrong at a DK,", flush=True)
print("   that DK is a harness/shape problem and says nothing about the extract defect.)", flush=True)
for dk in (8, 16, 32, 64):
    for arm, lbl in ((1, "arm1 CONTROL (vector mul)"), (0, "arm0 extract+scalar mul")):
        try:
            res, got, golden, k, rowerr, bad = run(dk, 0.9, 0.7, arm, f"iso_dk{dk}_a{arm}")
            rel = res.get("rel_l2")
            rel_s = f"{rel:.3e}" if isinstance(rel, float) else str(rel)
            print(f"  DK={dk:3d} {lbl:26s} status={res.get('status')} rel_l2={rel_s}  "
                  f"wrong {len(bad)}/{dk}  rows {bad.tolist()[:6]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  DK={dk:3d} {lbl:26s} ERROR {type(e).__name__}: {str(e)[:90]}", flush=True)
