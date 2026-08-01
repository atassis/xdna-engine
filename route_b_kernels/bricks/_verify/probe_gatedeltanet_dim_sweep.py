#!/usr/bin/env python3
"""Is gatedeltanet's fix principled, or did it just move a marginal case under a threshold?

WHY. gatedeltanet went NaN -> 0.000e+00 bit-exact when `float bk = beta * k[i]` became a vector op.
An isolated probe then showed the bare scalar multiply is FINE (32/32 lanes bit-exact, both forms),
so that was never the mechanism. The surviving explanation is one this project already measured:
[[kernel-internal-loops-miscompile-put-volume-in-the-worker]] -- a loop INSIDE an AIE kernel body
miscompiles, with error growing in BOTH live-vector count and iteration count (gelu-erf: clean at 1
chunk, 7.077e-01 at 2). gatedeltanet's NaN entered at t=2 of 64 with steps 0-1 clean, which is that
exact signature, and hoisting the multiply out of the inner DK loop reduces per-iteration live
values -- the KB's failure axis.

If that is right, the fix is PRESSURE-DEPENDENT and green at one shape says nothing about another.
This sweeps the two axes the KB names:
  * GDN_T   -- iteration count of the recurrence loop
  * GDN_DK/DV -- live vectors and work inside each iteration

PREDICTION IF THE HYPOTHESIS HOLDS: rel-L2 stays ~0 for small shapes and blows up (or NaNs) as T
and/or DK/DV grow. PREDICTION IF IT IS WRONG: every shape stays at the noise floor, and the fix is
principled after all -- in which case the mechanism is still open and must NOT be recorded as the
internal-loop bug.

Run:  ./run.sh probe_gatedeltanet_dim_sweep.py
"""
import importlib.util
from pathlib import Path

import ml_dtypes
import numpy as np

HERE = Path(__file__).parent
BRICK = (HERE.parent / "gatedeltanet").resolve()
BRICK_CC = BRICK / "gatedeltanet.cc"


def _load_golden():
    spec = importlib.util.spec_from_file_location("gdn_golden", BRICK / "golden.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_shape(T, DK, DV):
    import bricklib
    g = _load_golden()
    rng = np.random.default_rng(0)
    k, v, q, gates = g.make_inputs(rng, T, DK, DV)
    ker, _ = g.kernel_model(k, v, q, gates, DK, DV)

    kb = np.ascontiguousarray(k).astype(ml_dtypes.bfloat16)
    vb = np.ascontiguousarray(v).astype(ml_dtypes.bfloat16)
    qb = np.ascontiguousarray(q).astype(ml_dtypes.bfloat16)
    gf = np.ascontiguousarray(gates).astype(np.float32)
    s_in = np.zeros((DK, DV), dtype=np.float32)

    k_off = 0
    v_off = k_off + T * DK * 2
    q_off = v_off + T * DV * 2
    g_off = q_off + T * DK * 2
    packed = np.concatenate([
        kb.reshape(-1).view(np.int8), vb.reshape(-1).view(np.int8),
        qb.reshape(-1).view(np.int8), gf.reshape(-1).view(np.int8),
    ])
    shim = (
        'extern "C" void gdn_verify(int8_t* packed, float* s_in, bfloat16* o) {\n'
        f'  bfloat16* k = (bfloat16*)(packed + {k_off});\n'
        f'  bfloat16* v = (bfloat16*)(packed + {v_off});\n'
        f'  bfloat16* q = (bfloat16*)(packed + {q_off});\n'
        f'  float*    g = (float*)(packed + {g_off});\n'
        '  static float s_out_scratch[GDN_DK * GDN_DV];\n'
        '  gatedeltanet_step(k, v, q, g, s_in, o, s_out_scratch);\n'
        '}\n'
    )
    return bricklib.verify_oneshot(
        name=f"gdn_T{T}_DK{DK}_DV{DV}",
        brick_cc=BRICK_CC,
        shim_body=shim,
        symbol="gdn_verify",
        inputs=[(packed, np.int8), (s_in.reshape(-1), np.float32)],
        out_numel=T * DV,
        out_shape=(T, DV),
        unpack=lambda flat: np.asarray(flat).reshape(T, DV).astype(np.float32),
        golden=ker.astype(np.float32),
        gate=3e-2,
        compile_flags=[f"-DGDN_T={T}", f"-DGDN_DK={DK}", f"-DGDN_DV={DV}"],
        out_dt=ml_dtypes.bfloat16,
    )


# HOLD THE L1 FOOTPRINT ROUGHLY CONSTANT WHILE VARYING THE ITERATION COUNT. The first version of
# this sweep raised T at DK=DV=32 and raised DK/DV at fixed T, and FOUR of five shapes died in
# aiecc with "'aie.tile' op allocated buffers exceeded available memory" -- the whole T-sequence
# (k, v, q, gates, out) is resident in L1, so T and D both scale the footprint directly. A build
# that never ran says NOTHING about a numerics hypothesis, so those arms were not evidence.
#
# Bytes scale as T*D (packed operands and output) plus D^2*4 (the state). Halving D and doubling T
# therefore keeps the operand bytes identical while DOUBLING the recurrence's iteration count --
# which isolates the axis the NaN-at-t=2 signature actually points at.
SHAPES = [
    (64, 32, 32),    # the shape the brick gates at today -- control
    (128, 16, 16),   # same operand bytes as the control, 2x the iterations
    (256, 16, 16),   # 2x operand bytes, 4x the iterations
    (32, 32, 32),    # half the iterations, same D -- the other direction
]

rows = []
for T, DK, DV in SHAPES:
    try:
        r = run_shape(T, DK, DV)
        rows.append((T, DK, DV, r.get("status"), r.get("rel_l2")))
    except Exception as e:
        # Distinguish a BUILD failure from a wrong answer. Conflating them is how the first run of
        # this probe printed a confident verdict off four arms that never reached the device.
        msg = str(e)
        kind = "BUILD-FAIL" if "exceeded available memory" in msg or "aiecc" in msg else "ERROR"
        rows.append((T, DK, DV, kind, None))
        print(f"  T={T} DK={DK} DV={DV}: {kind}", flush=True)

print("\n===== gatedeltanet dim sweep (gate 3e-2) =====", flush=True)
for T, DK, DV, status, rl2 in rows:
    s = f"{rl2:.3e}" if isinstance(rl2, float) else "--"
    print(f"  T={T:4d} DK={DK:3d} DV={DV:3d}  {str(status):12s} rel_l2={s}", flush=True)

ran = [r for r in rows if isinstance(r[4], float)]
bad = [r for r in ran if r[4] > 3e-2 or not np.isfinite(r[4])]
built = len(ran)
print(f"\n{built}/{len(rows)} shapes actually built and ran; {len(bad)} of those are outside gate")
if built < 2:
    print("VERDICT: INCONCLUSIVE -- fewer than two shapes reached the device, so nothing is "
          "comparable. Shrink the shapes until they fit L1 and re-run.")
elif not bad:
    print("VERDICT: every shape that RAN is within gate -- this sweep does NOT support the "
          "internal-loop hypothesis, and the gatedeltanet mechanism stays OPEN")
else:
    print("VERDICT: rel-L2 degrades with iteration count at comparable footprint -- consistent "
          "with the kernel-internal-loop miscompile; the fix is pressure-dependent")
