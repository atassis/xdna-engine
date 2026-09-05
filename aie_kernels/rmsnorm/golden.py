#!/usr/bin/env python3
"""Host numpy golden reference for the rmsnorm AIE2P brick.

Matches rmsnorm.cc exactly (single-pass, no re-centering, affine, optional beta):
    ms      = mean(x**2)              over the `cols` axis, per row
    inv_rms = 1 / sqrt(ms + eps)
    out     = (x * inv_rms) * gamma [+ beta]

Contract: input is a resident [tile, D] stream -- shape (rows, cols). gamma/beta are
(cols,) affine params broadcast across rows (matching the caller-loop broadcast the
kernel assumes: gamma/beta are NOT baked per-row into the .cc, they're loaded fresh
each row by the calling wrapper/host in the same layout).

Later DEVICE pass: run rmsnorm_f32 (or rmsnorm_f32_nobeta) on-device over the same
(rows, cols) input/gamma/beta and compare against `rmsnorm_ref` below via rel-L2.
Suggested gate: rel_l2 < 3e-2 (consistent with the other bricks in this catalog;
f32-in/f32-out single-pass SFU invsqrt should track numpy far tighter than that in
practice -- 3e-2 is a generous device-noise ceiling, not the expected error).
"""
import numpy as np


def rmsnorm_ref(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray | None = None,
                eps: float = 1e-6) -> np.ndarray:
    """x: (rows, cols) f32. gamma: (cols,) f32. beta: (cols,) f32 or None."""
    x = x.astype(np.float32)
    gamma = gamma.astype(np.float32)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    inv_rms = 1.0 / np.sqrt(ms + eps)
    out = (x * inv_rms) * gamma
    if beta is not None:
        out = out + beta.astype(np.float32)
    return out.astype(np.float32)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den != 0.0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rows, cols = 8, 768  # D=768 matches the encoder's per-row width used by ln_2pass.cc
    x = rng.standard_normal((rows, cols)).astype(np.float32)
    gamma = rng.standard_normal((cols,)).astype(np.float32) * 0.1 + 1.0
    beta = rng.standard_normal((cols,)).astype(np.float32) * 0.01

    out_affine = rmsnorm_ref(x, gamma, beta)
    out_nobeta = rmsnorm_ref(x, gamma, None)

    # sanity: bias-free variant differs from beta-variant exactly by beta
    assert np.allclose(out_affine - beta[None, :], out_nobeta, atol=1e-5)

    # sanity: RMS of the normalized-only signal (x*inv_rms, before affine) should be ~1
    ms = np.mean(x * x, axis=-1)
    inv_rms = 1.0 / np.sqrt(ms + 1e-6)
    normed = x * inv_rms[:, None]
    rms_of_normed = np.sqrt(np.mean(normed * normed, axis=-1))
    assert np.allclose(rms_of_normed, 1.0, atol=1e-3)

    print(f"rows={rows} cols={cols}")
    print(f"out_affine sample[0,:5] = {out_affine[0, :5]}")
    print(f"out_nobeta sample[0,:5] = {out_nobeta[0, :5]}")
    print(f"rms_of_normed (expect ~1.0) sample = {rms_of_normed[:3]}")
    print("golden self-check OK")
