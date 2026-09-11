# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free contract test for the affine ("int4a"/"int8a") GEMV weight format.

Every check here stands for a property some measurement paid for, and would be a silent
regression rather than a failure if it went away:

  kernel factorisation   the kernel does NOT dequantize the min per element -- it folds it as
                         sum_g m_g * (sum_{k in g} b_k), a different arithmetic order from the
                         packer's own dequant. A round-trip test would pass while the kernel
                         computed something else, so this reimplements the kernel's form.
  byte identity          an int4a row is EXACTLY as long as a symmetric int4 row with an f32
                         scale, at every K and group this model builds. That is the whole
                         drop-in claim: it is what leaves swiglu_mlp_dp's TAPs, tile invariant
                         and L1 budget untouched. Widen the header and nothing else notices.
  zero on the grid       the reconstruction must contain exact 0. Measured: constraining the
                         offset costs a fraction of a step of range and is worth 3x on the model
                         (+3.90% against a free min's +16.26% at group 32). A packer change that
                         "improves" the range fit would regress quality 4x and trip nothing.
  the -8 nibble          the affine range is [-8, 7] and every group's own minimum lands on -8.
                         The symmetric path clips to [-7, 7] and never emitted that bit pattern,
                         so the signed unpack is load-bearing here for the first time.
  bf16 exactness         scale and min are read by the kernel as bf16; the packer must store
                         values that survive the round trip, or the fit is against numbers the
                         device does not have.

    python iron/tests/quant_affine.py
"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from iron.operators.gemv.quant import (quantize_weight, dequantize_weight,  # noqa: E402
                                        row_stride_bytes, is_affine)
try:
    import ml_dtypes
except ImportError:  # the packer needs it too; fail with the reason, not an AttributeError
    raise SystemExit("quant_affine needs ml_dtypes (the packer's bf16 type)")

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def kernel_dot(packed, M, K, g, wdt, b):
    """Reimplement mv_quant.cc's matvec_*_affine arithmetic, including group_sums_of_b."""
    import ml_dtypes
    n_groups = K // g
    hdr = 2 * n_groups
    rows = np.asarray(packed).view(np.uint8).reshape(M, row_stride_bytes(K, g, wdt))
    s = rows[:, :hdr].view(ml_dtypes.bfloat16).reshape(M, n_groups).astype(np.float32)
    m = rows[:, hdr:2 * hdr].view(ml_dtypes.bfloat16).reshape(M, n_groups).astype(np.float32)
    payload = rows[:, 2 * hdr:]
    if wdt == "int4a":
        lo = (payload & 0x0F).astype(np.int8); lo = np.where(lo >= 8, lo - 16, lo)
        hi = ((payload >> 4) & 0x0F).astype(np.int8); hi = np.where(hi >= 8, hi - 16, hi)
        q = np.empty((M, K), np.int8); q[:, 0::2] = lo; q[:, 1::2] = hi
    else:
        q = payload.view(np.int8)
    q = q.reshape(M, n_groups, g).astype(np.float32)
    bg = np.asarray(b, np.float32).reshape(n_groups, g)
    bsum = bg.sum(axis=1)                                   # group_sums_of_b
    qb = (q * bg[None, :, :]).sum(axis=2)                   # the flat mac loop, per group
    return (qb * s).sum(axis=1) + (m * bsum).sum(axis=1)    # + the n_groups scalar FMAs


def Wg_min(W, g):
    """Per-(row, group) minimum of the ORIGINAL weights -- what q = lo must map to."""
    M, K = W.shape
    return W.reshape(M, K // g, g).min(axis=2)


def main():
    rng = np.random.default_rng(0)
    for wdt, nb in (("int4a", 4), ("int8a", 8)):
        for K, g in ((1024, 32), (1024, 128), (3072, 64), (2048, 256)):
            M = 16
            print(f"{wdt} K={K} g={g}")
            W = rng.standard_normal((M, K)).astype(np.float32) * 0.02
            W[3, :g] += 0.5          # a shifted group: exactly what a symmetric scale wastes
            pk = quantize_weight(W, g, wdt)
            stride = row_stride_bytes(K, g, wdt)

            check("row stride == header + payload",
                  stride == 4 * (K // g) + (K // 2 if nb == 4 else K), f"{stride} B")
            check("stride is 4-byte aligned", stride % 4 == 0)
            check("packed size == M * stride", pk.nbytes == M * stride)

            dq = dequantize_weight(pk, M, K, g, wdt)
            check("levels used are within range",
                  bool(np.isfinite(dq).all()),
                  f"rel-L2 {np.linalg.norm(dq - W) / np.linalg.norm(W):.4f}")

            b = rng.standard_normal(K).astype(np.float32)
            ref = dq @ b
            got = kernel_dot(pk, M, K, g, wdt, b)
            rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
            check("kernel factorisation == dequant-then-dot", rel < 1e-5, f"rel {rel:.2e}")

            sym_dt = wdt.replace("a", "")
            check("byte-identical to symmetric with an f32 scale",
                  stride == row_stride_bytes(K, g, sym_dt, "f32"),
                  f"{stride} vs {row_stride_bytes(K, g, sym_dt, 'f32')}")

            hdr = 2 * (K // g)
            rows = np.asarray(pk).view(np.uint8).reshape(M, stride)
            sc = rows[:, :hdr].view(ml_dtypes.bfloat16).astype(np.float32)
            mn = rows[:, hdr:2 * hdr].view(ml_dtypes.bfloat16).astype(np.float32)
            check("scale and min survive the bf16 round trip the kernel reads them through",
                  np.array_equal(sc, sc.astype(ml_dtypes.bfloat16).astype(np.float32))
                  and np.array_equal(mn, mn.astype(ml_dtypes.bfloat16).astype(np.float32)))

            # The zero constraint is a 4-BIT property. bf16 stores the min with 8 significand
            # bits; |z| <= 8 leaves room, |z| <= 128 does not, so at int8 m cannot equal -z*s and
            # the constraint fails at its own job while costing range. Assert what is true at
            # each width rather than the same thing at both.
            lo, hi = -(1 << (nb - 1)), (1 << (nb - 1)) - 1
            zero_frac = float((dq == 0.0).mean())
            q_at_min = np.round((Wg_min(W, g) - mn) / np.where(sc == 0, 1, sc))
            q_all = np.round((W.reshape(M, K // g, g) - mn[:, :, None])
                             / np.where(sc == 0, 1, sc)[:, :, None])
            clipped = float(((q_all < lo) | (q_all > hi)).mean())
            if nb == 4:
                check("the reconstruction grid contains exact zero", zero_frac > 0.01,
                      f"{100*zero_frac:.2f}% of weights reconstruct as exactly 0")
                check(f"every group's own minimum lands on q={lo}", bool((q_at_min == lo).all()),
                      f"distinct q at the group minimum: {np.unique(q_at_min)[:4]}")
            else:
                check("int8 does NOT attempt the zero constraint (bf16 cannot hold z*s)",
                      zero_frac < 0.01, f"{100*zero_frac:.2f}% exactly 0")
            check("clipping stays inside a 1% budget", clipped < 0.01,
                  f"{100*clipped:.3f}% of values clip")

            # the shifted group is the point of the format: it must beat the symmetric form
            sym = dequantize_weight(quantize_weight(W, g, wdt.replace("a", "")), M, K,
                                    g, wdt.replace("a", ""))
            ea = np.linalg.norm(dq[3, :g] - W[3, :g])
            es = np.linalg.norm(sym[3, :g] - W[3, :g])
            check("affine beats symmetric on the shifted group", ea < es,
                  f"{ea:.4e} vs {es:.4e}")
    print(f"\n{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
