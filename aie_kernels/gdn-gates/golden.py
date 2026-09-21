#!/usr/bin/env python3
"""Golden for gdn_gates.cc: the gated delta rule's gate activation, computed in f64.

Matches Qwen3.5's Qwen3_5GatedDeltaNet: beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias),
alpha = exp(g). The kernel takes neg_a = -exp(A_log) precomputed and bf16 a/b, so the golden rounds
a/b to bf16 first and is otherwise exact.
"""
import numpy as np
from ml_dtypes import bfloat16


def bf16(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


def gates(a, b, a_log, dt_bias):
    """-> interleaved [N, 2] (alpha, beta) float64."""
    a, b = bf16(a).astype(np.float64), bf16(b).astype(np.float64)
    neg_a = -np.exp(np.asarray(a_log, np.float64))
    sp = np.logaddexp(0.0, a + np.asarray(dt_bias, np.float32).astype(np.float64))
    alpha = np.exp(neg_a * sp)
    beta = 1.0 / (1.0 + np.exp(-b))
    return np.stack([alpha, beta], -1)


def max_rel(got, ref):
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)))
