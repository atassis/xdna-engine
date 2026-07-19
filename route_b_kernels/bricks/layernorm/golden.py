#!/usr/bin/env python3
"""Host numpy golden reference for route_b_kernels/bricks/layernorm/layernorm.cc.

Generic norm{LN|RMS} brick reference -- matches layernorm_core<N,UseMean,Affine> exactly
(2-pass, mean-centered variance for LN; single-pass Sigma(x^2) for RMS; invsqrt(var+eps) tail;
optional affine applied in the same pass as the normalize).

Later DEVICE pass: run the compiled kernel on-device over the same random [rows, cols] input,
compare against `layernorm_ln_normalize/layernorm_rms_normalize` etc. below with rel-L2, gate at
3e-2 (matches the project's established rel-L2 bf16-path threshold, e.g. verify_fused_ffn).

Usage: python3 golden.py            # runs a self-check against a NON-cancelling
                                     # (E[x^2]-mean^2) formulation, prints rel-L2 per row.
"""
import numpy as np

EPS = 1e-5


def layernorm_ln_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """LayerNorm, normalize-only. x: [rows, cols] f32. Per-row 2-pass, mean-centered.

    Matches layernorm_core<N, UseMean=true, Affine=false>.
    """
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return ((x - mean) * inv_std).astype(np.float32)


def layernorm_ln_affine(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                         eps: float = EPS) -> np.ndarray:
    """LayerNorm with on-device affine. Matches layernorm_core<N, true, true>."""
    normed = layernorm_ln_normalize(x, eps)
    return (normed * gamma + beta).astype(np.float32)


def layernorm_rms_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """RMSNorm, normalize-only (mean=0, var=E[x^2]).

    Matches layernorm_core<N, UseMean=false, Affine=false>.
    """
    x = np.asarray(x, dtype=np.float32)
    var = np.mean(x ** 2, axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x * inv_std).astype(np.float32)


def layernorm_rms_affine(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                          eps: float = EPS) -> np.ndarray:
    """RMSNorm with on-device affine. Matches layernorm_core<N, false, true>."""
    normed = layernorm_rms_normalize(x, eps)
    return (normed * gamma + beta).astype(np.float32)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, cols = 8, 768  # cols must be a multiple of N=16 for the on-device kernel
    x = rng.standard_normal((rows, cols)).astype(np.float32) * 3.0 + 1.0
    gamma = rng.standard_normal(cols).astype(np.float32) * 0.1 + 1.0
    beta = rng.standard_normal(cols).astype(np.float32) * 0.1

    ln = layernorm_ln_normalize(x)
    ln_aff = layernorm_ln_affine(x, gamma, beta)
    rms = layernorm_rms_normalize(x)
    rms_aff = layernorm_rms_affine(x, gamma, beta)

    # Self-check: the 2-pass centered formulation vs the naive E[x^2]-mean^2 shortcut the
    # ln_2pass.cc header rejects for bf16 cancellation. In f64 host numpy both should still agree
    # closely (this just confirms the reference is internally consistent, not a bf16 regression
    # test -- that only shows up on the actual bf16 device path).
    mean = x.mean(axis=-1, keepdims=True)
    var_naive = (x ** 2).mean(axis=-1, keepdims=True) - mean ** 2
    ln_naive = (x - mean) / np.sqrt(var_naive + EPS)
    print("LN 2-pass vs naive E[x^2]-mean^2 rel-L2:", rel_l2(ln, ln_naive))

    print("ln_normalize shape", ln.shape, "sample row0[:4]", ln[0, :4])
    print("ln_affine    shape", ln_aff.shape, "sample row0[:4]", ln_aff[0, :4])
    print("rms_normalize shape", rms.shape, "sample row0[:4]", rms[0, :4])
    print("rms_affine    shape", rms_aff.shape, "sample row0[:4]", rms_aff[0, :4])
