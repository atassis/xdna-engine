#!/usr/bin/env python3
"""
golden.py -- host numpy reference for the qk-norm brick (RMSNorm on Q/K vectors
pre-attention), for the later DEVICE pass to rel-L2-check the compiled kernel
against.

Matches qk_norm.cc exactly:
    per row r of `cols` values (input[r, :]):
        ms      = mean(x**2)                 (over `cols`, axis=-1)
        inv_rms = 1 / sqrt(ms + eps)          (eps default 1e-6, matches
                                                QKNORM_EPS in qk_norm.cc)
        out[r, j] = x[r, j] * inv_rms * gamma[j]     (gamma optional)

Shapes:
    input : [rows, cols]  f32   (rows = resident tile row count, e.g.
                                  num_heads or num_heads * tile_tokens;
                                  cols = head_dim D)
    gamma : [cols]        f32, or None for the weight-free variant
    output: [rows, cols]  f32

No commit; this is a plain host-side script, no device/toolchain dependency.
"""
import numpy as np


def qk_norm(x: np.ndarray, gamma: np.ndarray | None = None,
            eps: float = 1e-6) -> np.ndarray:
    """RMSNorm over the last axis of x ([rows, cols] or [cols]), optional
    per-column gamma broadcast over rows. Computed in f64 as the "true" math
    reference; caller casts down for comparison against the f32 device path."""
    x = np.asarray(x, dtype=np.float64)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ms + eps)
    out = x * inv_rms
    if gamma is not None:
        out = out * np.asarray(gamma, dtype=np.float64)
    return out


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def make_golden_case(rows: int = 8, cols: int = 64, seed: int = 0,
                      with_gamma: bool = True, eps: float = 1e-6):
    """Deterministic test vector generator: returns (input, gamma_or_None, output).

    cols must be a multiple of 16 (the kernel's vector width N) -- 64 and 128
    (typical Q/K head_dim) both satisfy this.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    gamma = None
    if with_gamma:
        # learned scale, centered near 1.0 like a trained RMSNorm weight
        gamma = (1.0 + 0.1 * rng.standard_normal(cols)).astype(np.float32)
    out = qk_norm(x, gamma, eps=eps).astype(np.float32)
    return x, gamma, out


if __name__ == "__main__":
    # Self-check: degenerate rows==1 (decode M=1 case) and a small tile, both
    # gamma and no-gamma, cols=64 and cols=128 (multiples of N=16).
    for cols in (64, 128):
        for rows in (1, 8):
            for with_gamma in (True, False):
                x, gamma, out = make_golden_case(rows, cols,
                                                  with_gamma=with_gamma)
                # sanity: RMS of the normalized-only (pre-gamma) output is ~1
                ms_check = qk_norm(x, gamma=None)
                rms_of_norm = np.sqrt(np.mean(ms_check ** 2, axis=-1))
                assert np.allclose(rms_of_norm, 1.0, atol=1e-3), rms_of_norm
                print(f"cols={cols:4d} rows={rows:2d} gamma={with_gamma!s:5s} "
                      f"out.shape={out.shape} rms(normalize-only)~=1 OK")
    print("golden.py self-check PASS")
