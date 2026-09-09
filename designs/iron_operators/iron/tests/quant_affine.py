# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-free contract test for the affine ("int4a"/"int8a") GEMV weight format.

Three things are checked, and the third is the one that matters: the kernel does NOT dequantize
the min per element. It folds it as sum_g m_g * (sum_{k in g} b_k), which is a different
arithmetic order from the packer's own dequant. A test that only round-trips the packer would
pass while the kernel computed something else, so the third check reimplements the kernel's
factorisation and compares it to the dot product of the dequantized row.

    python iron/tests/quant_affine.py
"""
import sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from iron.operators.gemv.quant import (quantize_weight, dequantize_weight,  # noqa: E402
                                        row_stride_bytes, is_affine)

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
