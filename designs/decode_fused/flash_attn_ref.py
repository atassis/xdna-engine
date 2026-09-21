# SPDX-License-Identifier: Apache-2.0
"""Block model of the fused prefill attention operator (online softmax), and its references.

The model does what the device workers do per block of `b_kv` keys: f32 scores, hidden keys
SELECTED to -inf (never masked by addition, which lets a NaN key through), a running max, sum
and accumulator in f32, and a row with no visible key in a block keeping its state untouched.
"""
import numpy as np


def visible_from_widths(widths, cols):
    """[R] widths -> [R, cols] bool: row r sees columns [0, widths[r])."""
    return np.arange(cols)[None, :] < np.asarray(widths)[:, None]


def visible_from_ring_rows(rows, cols):
    """[R, 3] (hole_lo, hole_hi, width) -> [R, cols] bool, the rows_hole softmax's mask."""
    c = np.arange(cols)[None, :]
    lo, hi, wid = (np.asarray(rows)[:, j:j + 1] for j in range(3))
    return ~(((c >= lo) & (c < hi)) | (c >= wid))


def full_attention(q, k, v, visible, scale):
    """One-shot softmax attention in f64 over the visible keys. Every row must see a key."""
    if not visible.any(axis=1).all():
        raise ValueError("a row with no visible key has no defined softmax")
    s = (q.astype(np.float64) @ k.astype(np.float64).T) * scale
    s = np.where(visible, s, -np.inf)
    p = np.exp(s - s.max(axis=1, keepdims=True))
    return (p @ v.astype(np.float64)) / p.sum(axis=1, keepdims=True)


def flash_attention(q, k, v, visible, scale, b_kv=64, guard_empty=True, rescale=True,
                    additive_mask=False):
    """Online-softmax attention over blocks of `b_kv` keys, state in f32.

    The new block's probabilities are taken against the MERGED max and only the old state is
    rescaled -- the form mha_decode.cc's flash loop uses.
    guard_empty=False, rescale=False and additive_mask=True are known-wrong variants that exist
    only for the negative controls.
    """
    q, k, v = (x.astype(np.float32) for x in (q, k, v))
    rows_n, cols = visible.shape
    if cols % b_kv:
        raise ValueError(f"{cols} columns is not a whole number of {b_kv}-key blocks")
    m = np.full(rows_n, -np.inf, np.float32)
    l = np.zeros(rows_n, np.float32)
    acc = np.zeros((rows_n, v.shape[1]), np.float32)
    for c0 in range(0, cols, b_kv):
        vis = visible[:, c0:c0 + b_kv]
        s = (q @ k[c0:c0 + b_kv].T) * np.float32(scale)
        if additive_mask:
            s = s + np.where(vis, np.float32(0), np.float32(-np.inf))
        else:
            s = np.where(vis, s, np.float32(-np.inf))
        live = vis.any(axis=1) if guard_empty else np.ones(rows_n, bool)
        m_new = np.maximum(m[live], s[live].max(axis=1))
        corr = np.exp(m[live] - m_new) if rescale else np.ones_like(m_new)
        p = np.exp(s[live] - m_new[:, None])
        l[live] = l[live] * corr + p.sum(axis=1)
        acc[live] = acc[live] * corr[:, None] + p @ v[c0:c0 + b_kv]
        m[live] = m_new
    return acc / l[:, None]
