#!/usr/bin/env python3
"""Host numpy golden for route_b_kernels/bricks/snake/snake.cc.

Snake activation as the DAC codec decoder uses it (s2.cpp/src/s2_codec.cpp:197-204):
    y = x + sin^2(alpha * x) / alpha
alpha is per-CHANNEL (one scalar per channel), broadcast along the time axis. The fish-audio codec
applies Snake at every upsample stage, and the same codec.pth serves S2-Pro and s1-mini.

`snake_emulated` composes the sin brick's own emulation, so this predicts the DEVICE path rather
than just defining the maths.

Usage: python3 golden.py
"""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("sin_golden", Path(__file__).parent.parent / "sin" / "golden.py")
_sin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sin)


def snake_ref(x, alpha):
    """Ground truth. x: [channels, time] f32. alpha: [channels] f32, all > 0."""
    x = np.asarray(x, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32).reshape(-1, 1).astype(np.float64)
    s = np.sin(a * x.astype(np.float64))
    return (x + (s * s) / a).astype(np.float32)


def snake_emulated(x, alpha):
    """Predicts the device: the kernel's own sin path, then the same combine."""
    x = np.asarray(x, dtype=np.float32)
    a = np.asarray(alpha, dtype=np.float32).reshape(-1, 1)
    s = _sin.sin_poly_emulated((a * x).astype(np.float32))
    return (x + (s * s).astype(np.float32) / a).astype(np.float32)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ch, t = 8, 64
    x = (rng.standard_normal((ch, t)).astype(np.float32) * 3.0)
    alpha = (rng.random(ch, dtype=np.float32) * 4.0 + 0.25).astype(np.float32)
    y = snake_ref(x, alpha)
    e = snake_emulated(x, alpha)
    print("shape", y.shape, "row0[:4]", np.round(y[0, :4], 5).tolist())
    print("emulated vs ref rel-L2:", rel_l2(e, y))
    # Snake never decreases its input for positive alpha, since sin^2 >= 0. A real invariant.
    assert np.all(y >= x - 1e-5), "snake must not decrease its input for positive alpha"
    print("invariant y >= x holds")
