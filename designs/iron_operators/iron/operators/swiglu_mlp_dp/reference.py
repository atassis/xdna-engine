# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16


def reference(cur, a, n_pf, Wg, Wu, Wd, D, FF, epsilon=1e-5):
    """CPU (f32) reference for the data-parallel decode SwiGLU MLP block. Same math as
    swiglu_mlp_fused's reference (fusion/parallelism strategy does not change the numerics):
        x1  = cur + a
        hf  = RMSNorm_weighted(x1, n_pf, epsilon)
        nxt = x1 + Wd @ (SiLU(Wg @ hf) * (Wu @ hf))
    Wg, Wu are flat [FF*D] row-major (FF rows, D cols); Wd is flat [D*FF] row-major (D rows,
    FF cols) -- the on-wire layout every core's own column slice indexes into.
    """
    cur = np.asarray(cur, np.float32)
    a = np.asarray(a, np.float32)
    n_pf = np.asarray(n_pf, np.float32)
    Wg = np.asarray(Wg, np.float32).reshape(FF, D)
    Wu = np.asarray(Wu, np.float32).reshape(FF, D)
    Wd = np.asarray(Wd, np.float32).reshape(D, FF)

    x1 = cur + a
    rms = np.sqrt(np.mean(x1 * x1) + epsilon)
    hf = (x1 / rms) * n_pf

    g = Wg @ hf
    sig = np.empty_like(g)
    pos = g >= 0
    sig[pos] = 1.0 / (1.0 + np.exp(-g[pos]))
    sig[~pos] = np.exp(g[~pos]) / (1.0 + np.exp(g[~pos]))
    silu_g = g * sig
    u = Wu @ hf
    gh = silu_g * u
    d = Wd @ gh
    nxt = x1 + d
    return nxt.astype(bfloat16)


def generate_golden_reference(D, FF, seed=42):
    rng = np.random.default_rng(seed)
    val_range = 1.0
    cur = (rng.standard_normal(D) * val_range).astype(bfloat16)
    a = (rng.standard_normal(D) * val_range).astype(bfloat16)
    n_pf = (rng.standard_normal(D) * val_range + 1.0).astype(bfloat16)
    Wg = (rng.standard_normal(FF * D) * val_range).astype(bfloat16)
    Wu = (rng.standard_normal(FF * D) * val_range).astype(bfloat16)
    Wd = (rng.standard_normal(D * FF) * val_range).astype(bfloat16)
    nxt = reference(cur, a, n_pf, Wg, Wu, Wd, D, FF)
    return {"cur": cur, "a": a, "n_pf": n_pf, "Wg": Wg, "Wu": Wu, "Wd": Wd, "nxt": nxt}


def reference_fused_o(cur, cx, n_pf, Wo, Wg, Wu, Wd, D, FF, QD, epsilon=1e-5):
    """CPU (f32) reference for the fuse_o arm: same math as `reference()`, except `a` is computed
    on-chip from `Wo @ cx` instead of arriving as an external input.

    `Wo` may be the PADDED [D+pad, QD] buffer design.py's device path actually loads (see
    op.py's `_wo_rows_padded` / design.py's FUSE_O module docstring) -- only rows [0:D) are real,
    the rest are zero-padding added solely so the on-chip TSI_O-row tiling divides evenly, and
    their computed (but never drained) contribution is exactly zero. Slicing to the first D rows
    here reproduces that math exactly regardless of how much padding was applied.
    """
    cx = np.asarray(cx, np.float32)
    Wo = np.asarray(Wo, np.float32).reshape(-1, QD)[:D]
    a = (Wo @ cx).astype(bfloat16)
    return reference(cur, a, n_pf, Wg, Wu, Wd, D, FF, epsilon)


def generate_golden_reference_fused_o(D, FF, QD, wo_rows_padded=None, seed=42):
    """Golden inputs/output for the fuse_o arm. `wo_rows_padded` defaults to D (no padding) --
    callers exercising the real device path pass op.py's `_wo_rows_padded` so the returned `Wo`
    buffer matches the arg-spec size the device actually reads, with the tail rows zero (see
    design.py's FUSE_O overlap/pad derivation)."""
    rng = np.random.default_rng(seed)
    val_range = 1.0
    rows = wo_rows_padded if wo_rows_padded is not None else D
    assert rows >= D
    cur = (rng.standard_normal(D) * val_range).astype(bfloat16)
    cx = (rng.standard_normal(QD) * val_range).astype(bfloat16)
    n_pf = (rng.standard_normal(D) * val_range + 1.0).astype(bfloat16)
    Wo_real = (rng.standard_normal(D * QD) * val_range).astype(bfloat16).reshape(D, QD)
    Wo = np.zeros((rows, QD), dtype=bfloat16)
    Wo[:D] = Wo_real
    Wo = Wo.reshape(-1)
    Wg = (rng.standard_normal(FF * D) * val_range).astype(bfloat16)
    Wu = (rng.standard_normal(FF * D) * val_range).astype(bfloat16)
    Wd = (rng.standard_normal(D * FF) * val_range).astype(bfloat16)
    nxt = reference_fused_o(cur, cx, n_pf, Wo, Wg, Wu, Wd, D, FF, QD)
    return {
        "cur": cur, "cx": cx, "n_pf": n_pf, "Wo": Wo, "Wg": Wg, "Wu": Wu, "Wd": Wd, "nxt": nxt,
    }
