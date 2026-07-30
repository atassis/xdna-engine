#!/usr/bin/env python3
"""Host numpy golden for route_b_kernels/bricks/conv-transpose-1d/conv_transpose_1d.cc.

Causal transposed 1-D convolution as the fish-audio DAC codec decoder uses it
(s2.cpp/src/s2_codec.cpp:236-250): conv_transpose_1d(stride, pad=0, dilation=1), then per-output-
channel bias, then crop `crop_right` from the right. The codec's decoder_rates are [8, 8, 4, 2],
512x total upsample to 44.1 kHz.

Usage: python3 golden.py
"""
import numpy as np


def conv_transpose_1d_ref(x, w, bias, stride, crop_right=0):
    """x: [c_in, t] f32. w: [c_in, c_out, k] f32. bias: [c_out] f32.

    Output length before crop is (t - 1) * stride + k.
    """
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    c_in, t = x.shape
    c_in_w, c_out, k = w.shape
    assert c_in == c_in_w, f"channel mismatch: x has {c_in}, w has {c_in_w}"
    out_len = (t - 1) * stride + k
    y = np.zeros((c_out, out_len), dtype=np.float64)
    for ci in range(c_in):
        for ti in range(t):
            base = ti * stride
            y[:, base:base + k] += w[ci] * x[ci, ti]
    y += np.asarray(bias, dtype=np.float64).reshape(-1, 1)
    if crop_right > 0:
        y = y[:, :out_len - crop_right]
    return y.astype(np.float32)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    c_in, c_out, k, t, stride = 2, 2, 16, 4, 4
    x = rng.standard_normal((c_in, t)).astype(np.float32)
    w = (rng.standard_normal((c_in, c_out, k)).astype(np.float32) * 0.1)
    b = (rng.standard_normal(c_out).astype(np.float32) * 0.01)
    y = conv_transpose_1d_ref(x, w, b, stride)
    assert y.shape == (c_out, (t - 1) * stride + k), f"unexpected shape {y.shape}"
    print("shape", y.shape, "row0[:4]", np.round(y[0, :4], 5).tolist())
    # A transposed conv is the ADJOINT of a strided conv, so <conv_T(x), g> == <x, conv(g)> + bias term
    # for random g. This catches an off-by-one in the scatter, which a shape assert cannot.
    g = rng.standard_normal(y.shape).astype(np.float32)
    lhs = float(np.sum(y.astype(np.float64) * g.astype(np.float64)))
    conv_g = np.zeros_like(x, dtype=np.float64)
    for ci in range(c_in):
        for ti in range(t):
            conv_g[ci, ti] = np.sum(w[ci] * g[:, ti * stride:ti * stride + k])
    rhs = float(np.sum(x.astype(np.float64) * conv_g)) + float(
        np.sum(b.astype(np.float64).reshape(-1, 1) * g.astype(np.float64)))
    print("adjoint lhs/rhs:", round(lhs, 6), round(rhs, 6), "rel:", abs(lhs - rhs) / abs(rhs))
    assert abs(lhs - rhs) / abs(rhs) < 1e-4, "adjoint identity violated -- reference is wrong"
    print("adjoint identity holds")
