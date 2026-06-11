"""Relative-error comparison helpers vs ONNX reference tensors."""
import numpy as np
from .dtypes import f32


def rel(a, b):
    """max|a-b| / max|b| — scale-invariant per-tensor error."""
    a, b = f32(a), f32(b)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))


def report(name, got, ref, tol):
    got, ref = f32(got), f32(ref)
    if got.shape != ref.shape:
        print(f"  {name:14s} SHAPE {got.shape} vs {ref.shape} ***"); return False
    d = np.abs(got - ref)
    r = d.max() / (np.abs(ref).max() + 1e-9)
    ok = r < tol
    print(f"  {name:14s} max|Δ|={d.max():.4e}  mean={d.mean():.4e}  rel={r:.2e}  {'ok' if ok else '**OFF**'}")
    return ok
