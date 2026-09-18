#!/usr/bin/env python3
"""Host references for conv_transpose_channel_bf16.cc.

Same two-golden split as bricks/conv-1d/golden_bf16.py (read that file's header for the rationale):
an f32-truth reference (reusing the existing conv-transpose-1d brick's golden, unmodified) and a
bf16-quantized MODEL that isolates device correctness from bf16's own quantization loss.

Usage: python3 golden_conv_transpose_bf16.py
"""
import sys
from pathlib import Path

import numpy as np
import ml_dtypes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bricks" / "conv-transpose-1d"))
from golden import conv_transpose_1d_ref  # noqa: E402

BF16 = ml_dtypes.bfloat16


def _to_bf16(a):
    return np.asarray(a, np.float32).astype(BF16)


def conv_transpose_1d_bf16_model(x, w, bias, stride, crop_right):
    """x: [c_in, t] f32. w: [c_in, c_out, k] f32 (conv_transpose layout). bias: [c_out] f32.
    Returns [c_out, t*stride - crop_right] f32, the bf16-quantized output widened back to f32.
    Same accumulate-wide-never-in-bf16 rule as conv_1d's model."""
    xb = _to_bf16(x).astype(np.float64)
    wb = _to_bf16(w).astype(np.float64)
    bb = _to_bf16(bias).astype(np.float64)
    c_in, c_out, k = wb.shape
    c_in_x, t = xb.shape
    assert c_in == c_in_x, f"channel mismatch: x has {c_in_x}, w has {c_in}"
    out_len = (t - 1) * stride + k
    y = np.zeros((c_out, out_len), np.float64)
    contrib = np.einsum("ioj,it->jot", wb, xb, optimize=True)
    for j in range(k):
        y[:, j::stride][:, :t] += contrib[j]
    y += bb.reshape(-1, 1)
    y = y[:, :out_len - crop_right] if crop_right else y
    return _to_bf16(y.astype(np.float32)).astype(np.float32)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Stage-4 shape (the smallest, what upsample_stage.cc gates at by default).
    c_in, c_out, k, stride, t = 192, 96, 4, 2, 16
    x = rng.standard_normal((c_in, t)).astype(np.float32)
    w = (rng.standard_normal((c_in, c_out, k)).astype(np.float32) * 0.1)
    b = (rng.standard_normal(c_out).astype(np.float32) * 0.1)

    y_f32 = conv_transpose_1d_ref(x, w, b, stride, crop_right=stride)
    y_bf16 = conv_transpose_1d_bf16_model(x, w, b, stride, crop_right=stride)
    assert y_bf16.shape == y_f32.shape, f"{y_bf16.shape} vs {y_f32.shape}"
    rl2 = rel_l2(y_bf16, y_f32)
    assert 1e-4 < rl2 < 5e-1, f"bf16 model rel-L2 {rl2:.3e} outside sanity band"
    print(f"stage4 shape: bf16-model vs f32-truth rel-L2 {rl2:.3e}")
    print("conv_transpose golden_bf16 OK")
