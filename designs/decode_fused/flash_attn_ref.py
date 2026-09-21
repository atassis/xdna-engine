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
    raise NotImplementedError


def flash_attention(q, k, v, visible, scale, b_kv=64, guard_empty=True, rescale=True,
                    additive_mask=False):
    raise NotImplementedError
