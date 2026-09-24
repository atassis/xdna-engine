# SPDX-License-Identifier: Apache-2.0
"""Block model of the decode global-layer flash-attention operator (attn_global_flash).

64-position blocks dealt round-robin to `columns` workers; per block: bf16 scores, hidden keys
selected to -inf, f32 running max/sum/context, bf16 probabilities, P*V over the unmasked rows only.
A block wholly past n_live is skipped. Partials merge in f32 with an empty partial weighted zero.
The device's exp2 is an SFU LUT; this model uses float exp, so compare within that band (K020).
"""
import numpy as np
from ml_dtypes import bfloat16


def bf16(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float32)


def blocks_per_column(n_live, block=64, columns=8):
    if n_live < 1:
        raise ValueError("n_live must be >= 1")
    return -(-(-(-n_live // block)) // columns)


def last_block_index(n_live, col, block=64, columns=8):
    return col + columns * (blocks_per_column(n_live, block, columns) - 1)


def column_partial(q, k, v, n_live, col, block=64, columns=8):
    hq, hd = q.shape
    m = np.full(hq, -np.inf, np.float32)
    l = np.zeros(hq, np.float32)
    acc = np.zeros((hq, hd), np.float32)
    for j in range(blocks_per_column(n_live, block, columns)):
        lo = (col + columns * j) * block
        live = min(max(n_live - lo, 0), block)
        if live == 0:
            continue
        with np.errstate(invalid="ignore"):
            s = bf16(q @ k[lo:lo + block].T)
        s[:, live:] = -np.inf
        m_new = np.maximum(m, s.max(axis=1))
        corr = np.exp(m - m_new)
        p = bf16(np.exp(s[:, :live] - m_new[:, None]))
        l = l * corr + p.sum(axis=1, dtype=np.float32)
        acc = acc * corr[:, None] + p @ v[lo:lo + live]
        m = m_new
    return m, l, acc


def merge(parts):
    hq, hd = parts[0][2].shape
    m = np.full(hq, -np.inf, np.float32)
    l = np.zeros(hq, np.float32)
    acc = np.zeros((hq, hd), np.float32)
    for pm, pl, pacc in parts:
        m_new = np.maximum(m, pm)
        with np.errstate(invalid="ignore"):
            cp = np.where(m == -np.inf, 0.0, np.exp(m - m_new)).astype(np.float32)
            cc = np.where(pm == -np.inf, 0.0, np.exp(pm - m_new)).astype(np.float32)
        l = l * cp + pl * cc
        acc = acc * cp[:, None] + pacc * cc[:, None]
        m = m_new
    return bf16(acc / l[:, None])


def decode_flash_attention(q, k, v, n_live, block=64, columns=8):
    return merge([column_partial(q, k, v, n_live, c, block, columns) for c in range(columns)])
