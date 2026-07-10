#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Numpy golden for the Gemma gated-GeGLU FFN sub-block (the r1-spike reference math).

Reconstructs, from the oracle-dumped weights, the exact sub-block the NPU kernel reproduces:

    normed = gemma_rmsnorm(x_in, pre_norm)                 # RMSNorm: f32 sum-of-squares, *(1+gamma)
    h      = gelu_tanh(normed @ gate.T) * (normed @ up.T)  # gated GeGLU
    out    = x_in + gemma_rmsnorm(h @ down.T, post_norm)   # sandwich post-norm + residual

Gemma specifics that MUST match (audit): RMSNorm accumulates sum-of-squares in float32 and scales by
(1 + gamma) (gamma initialised to 0); activation is gelu_pytorch_tanh. `compute_dtype=float32` proves
the formula matches HF exactly; a bf16 compute path is the golden the NPU bf16 kernel is gated against.
"""
import json
import os

import numpy as np


def gelu_tanh(x):
    x = x.astype(np.float32)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def gemma_rmsnorm(x, gamma, eps):
    """RMSNorm with EFFECTIVE gamma (the oracle already folds the family convention: Gemma3 1+w vs
    Gemma4 w). Sum-of-squares in float32 (audit: bf16 ssq is a bug)."""
    x = x.astype(np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    xn = x / np.sqrt(var + eps)
    return xn * gamma.astype(np.float32)


def _maybe_bf16(x, compute_dtype):
    """Emulate bf16 rounding by truncating the float32 mantissa to 7 bits (round-to-nearest-even-ish)."""
    if compute_dtype == "float32":
        return x.astype(np.float32)
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x8000) & 0xFFFF0000  # round + truncate lower 16 mantissa bits
    return u.view(np.float32)


def ffn_forward(x_in, wdir, eps=1e-6, compute_dtype="float32"):
    """Gemma FFN resident sub-block: pre_norm -> gate/up -> GeGLU -> down (= the dense mlp output).

    Boundary matches scripts/gemma_ffn_oracle.py: input = pre_feedforward_layernorm INPUT, output = mlp
    module OUTPUT. Excludes post_norm / residual / MoE / PLE (layer-level plumbing). wdir holds
    gate_proj/up_proj/down_proj/pre_norm .npy.
    """
    g = np.load(os.path.join(wdir, "gate_proj.npy"))   # [I, D]
    u = np.load(os.path.join(wdir, "up_proj.npy"))     # [I, D]
    d = np.load(os.path.join(wdir, "down_proj.npy"))   # [D, I]
    pre = np.load(os.path.join(wdir, "pre_norm.npy"))  # [D]
    bf = lambda t: _maybe_bf16(t, compute_dtype)

    normed = bf(gemma_rmsnorm(x_in.astype(np.float32), pre, eps))  # RMSNorm always in f32
    gate = bf(normed @ bf(g).T)          # [.., I]
    up = bf(normed @ bf(u).T)            # [.., I]
    h = bf(gelu_tanh(gate) * up)         # [.., I]
    down = bf(h @ bf(d).T)               # [.., D]  = dense mlp output (sub-block output)
    return down.astype(np.float32)


def rel_l2(a, b):
    a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def corr(a, b):
    a, b = a.astype(np.float32).ravel(), b.astype(np.float32).ravel()
    return float(np.corrcoef(a, b)[0, 1])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, help="dir with ffn_in.npy, ffn_out.npy, weights/, meta.json")
    ap.add_argument("--compute-dtype", default="float32", choices=["float32", "bf16"])
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.oracle, "meta.json")))
    x = np.load(os.path.join(a.oracle, "ffn_in.npy"))
    ref = np.load(os.path.join(a.oracle, "ffn_out.npy"))
    got = ffn_forward(x, os.path.join(a.oracle, "weights"), meta["rms_norm_eps"], a.compute_dtype)
    print(f"[golden] {meta['model']} L{meta['layer']} dtype={a.compute_dtype}: "
          f"rel_L2={rel_l2(got, ref):.2e}  corr={corr(got, ref):.6f}")
