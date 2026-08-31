#!/usr/bin/env python3
"""Which PART of the 3-part gatedeltanet fix actually mattered?

The fix (a6d6b47) took the kernel from NaN to 0.000e+00 bit-exact by moving both gates into the
vector domain. But its stated mechanism -- "the aie2p scalar-f32 path miscompiles" -- was REFUTED in
isolation: a standalone scalar f32 multiply is fine on device.
So the fix works and
its explanation does not, which means the responsible edit has never been named.

The commit changes three things at once:
  (a) alpha's broadcast moves from inside gdn_state_write to the caller
  (b) beta*k moves from a per-i scalar f32 mul inside the loop to ONE vector mul in the caller
  (c) gdn_state_write's signature changes to take the two vectors

(c) is forced by (a) and (b), so the question is which of (a)/(b) carries the fix. GDN_VARIANT
compiles the SHIPPED kernel with each half independently disabled:

  0 = pre-fix    (scalar alpha, scalar beta*k[i])   -- expect FAIL
  1 = alpha only (vector alpha, scalar beta*k[i])
  2 = bk only    (scalar alpha, vector beta*k)
  3 = shipped    (both)                             -- expect PASS

If 2 passes and 1 fails, the scalar beta*k[i] is the trigger and the isolation result needs
explaining (fine alone, not here). If BOTH 1 and 2 pass, neither edit is individually necessary and
the rewrite fixed it incidentally -- a very different conclusion, and one worth having before this
pattern gets copied into other kernels.

Run:  ./run.sh probe_gdn_part_bisect.py
"""
import ml_dtypes
import numpy as np

import bricklib
import verify_gatedeltanet as V

LABEL = {0: "pre-fix    (scalar alpha, scalar beta*k)",
         1: "alpha only (vector alpha, scalar beta*k)",
         2: "bk only    (scalar alpha, vector beta*k)",
         3: "shipped    (both vector)"}


def run_variant(variant):
    g = V._load_golden()
    T, DK, DV = V.GDN_T, V.GDN_DK, V.GDN_DV
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
    packed = np.concatenate([kb.reshape(-1).view(np.int8), vb.reshape(-1).view(np.int8),
                             qb.reshape(-1).view(np.int8), gf.reshape(-1).view(np.int8)])

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
        name=f"gdn_v{variant}",
        brick_cc=V.BRICK_CC,
        shim_body=shim,
        symbol="gdn_verify",
        inputs=[(packed, np.int8), (s_in.reshape(-1), np.float32)],
        out_numel=T * DV,
        out_shape=(T, DV),
        unpack=lambda flat: np.asarray(flat).reshape(T, DV).astype(np.float32),
        golden=ker.astype(np.float32),
        gate=3e-2,
        compile_flags=[f"-DGDN_VARIANT={variant}"],
        out_dt=ml_dtypes.bfloat16,
    )


print("===== gatedeltanet 3-part fix bisect =====", flush=True)
for variant in (0, 1, 2, 3):
    try:
        res = run_variant(variant)
        got = np.asarray(res.get("got"), np.float32)
        nf = int((~np.isfinite(got)).sum())
        rel = res.get("rel_l2")
        rel_s = f"{rel:.3e}" if isinstance(rel, float) else str(rel)
        print(f"  V{variant} {LABEL[variant]:42s} status={res.get('status')} "
              f"rel_l2={rel_s} non-finite {nf}/{got.size}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  V{variant} {LABEL[variant]:42s} ERROR {type(e).__name__}: {str(e)[:120]}",
              flush=True)
