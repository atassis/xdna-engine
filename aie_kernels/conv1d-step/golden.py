#!/usr/bin/env python3
"""Numpy golden for conv1d_step.cc: one step of a depthwise causal conv with carried history.

kernel_model is bit-faithful to the kernel (bf16 operands, f32 accumulate, one bf16 round);
host_reference is the same step in f32 on unrounded inputs.
"""
import numpy as np
from ml_dtypes import bfloat16


def bf16(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


def host_reference(x, hist, w):
    """x [C], hist [K-1, C] oldest-first, w [K, C] tap-major -> (y [C], hist_out [K-1, C])."""
    y = (w[:-1] * hist).sum(0) + w[-1] * x
    return y.astype(np.float32), np.concatenate([hist[1:], x[None]], 0)


def kernel_model(x, hist, w):
    xb, hb, wb = bf16(x), bf16(hist), bf16(w)
    y, h = host_reference(xb, hb, wb)
    return bf16(y), h


def conv_as_torch(xs, w):
    """The same op the way HF runs it for a whole sequence (causal_conv1d_fn without activation):
    xs [T, C], w [K, C]; used to check that stepping reproduces the sequence form."""
    T, C = xs.shape
    K = w.shape[0]
    xp = np.concatenate([np.zeros((K - 1, C), np.float32), xs], 0)
    return np.stack([(w * xp[t:t + K]).sum(0) for t in range(T)])


def rel_l2(a, b):
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)).ravel()) /
                 (np.linalg.norm(np.asarray(b).ravel()) + 1e-12))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, C, K = 12, 256, 4
    xs = rng.standard_normal((T, C)).astype(np.float32)
    w = (rng.standard_normal((K, C)) * 0.5).astype(np.float32)
    hist = np.zeros((K - 1, C), np.float32)
    ys = []
    for t in range(T):
        y, hist = host_reference(xs[t], hist, w)
        ys.append(y)
    print(f"[conv1d-step] stepped vs sequence form rel-L2 {rel_l2(np.stack(ys), conv_as_torch(xs, w)):.3e}")
