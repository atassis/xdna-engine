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


# T first (iteration count, the axis the NaN-at-t=2 signature points at), then DK/DV
# (live vectors and work per iteration). 32/32/64 is the shape the brick currently gates at.
SHAPES = [(64, 32, 32), (128, 32, 32), (256, 32, 32), (64, 64, 64), (128, 64, 64)]

rows = []
for T, DK, DV in SHAPES:
    try:
        r = run_shape(T, DK, DV)
        rl2 = r.get("rel_l2")
        rows.append((T, DK, DV, r.get("status"), rl2))
    except Exception as e:
        rows.append((T, DK, DV, f"ERROR {type(e).__name__}", None))
        print(f"  T={T} DK={DK} DV={DV}: {e}", flush=True)

print("\n===== gatedeltanet dim sweep (gate 3e-2) =====", flush=True)
for T, DK, DV, status, rl2 in rows:
    s = f"{rl2:.3e}" if isinstance(rl2, float) else "--"
    print(f"  T={T:4d} DK={DK:3d} DV={DV:3d}  {str(status):16s} rel_l2={s}", flush=True)

clean = [r for r in rows if isinstance(r[4], float) and r[4] <= 3e-2]
print(f"\n{len(clean)}/{len(rows)} shapes within gate")
if len(clean) == len(rows):
    print("VERDICT: no dim dependence found -- the internal-loop hypothesis is NOT supported by "
          "this sweep, and the gatedeltanet mechanism stays OPEN")
else:
    print("VERDICT: error depends on the shape -- consistent with the kernel-internal-loop "
          "miscompile; the fix is pressure-dependent, not principled")
