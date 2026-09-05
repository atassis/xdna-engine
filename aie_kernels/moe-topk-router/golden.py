#!/usr/bin/env python3
"""Host numpy golden for the moe-topk-router brick (moe_topk_router.cc).

Generic MoE gate GEMM + top-K expert select, M=1 padded to MOE_M_PAD rows for
the mmul r-dim (row 0 = live query, rows 1..M-1 are zero padding, same
convention as bricks/lm-head-argmax/golden reasoning and
decode_norm_gemv/norm_gemv_iron.py).

This is the HOST reference the later DEVICE pass checks rel-L2 against -- it
mirrors the kernel's arithmetic exactly:

  1. logits = hidden[0, :] @ Wg      (bf16 in, f32 accumulate; only row 0 of
     the M-padded tile is the live query, so only row 0 of the GEMM matters)
  2. (top_values, top_indices) = topk(logits, K), DESCENDING, via
     SELECTION-SORT-BY-ARGMAX: take the max, mask it to -inf, repeat K times
     -- this exactly mirrors the kernel's iterated argmax_full_reduce loop
     (the "extend max_cmp beyond k=1" primitive), not np.argpartition/np.sort,
     so ties break the SAME way (first index wins) as the device's
     left-to-right max_cmp/select reduction tree.

Usage (self-check when run directly):
    python3 golden.py
"""
from __future__ import annotations

import numpy as np

# Same default tile shape as the kernel's compile-time macros.
MOE_HIDDEN = 256   # HIDDEN dim (K of the gate GEMM)
MOE_EXPERTS = 64    # NUM_EXPERTS (N of the gate GEMM)
MOE_TOPK = 2        # experts routed to per token (e.g. Mixtral-style top-2)
MOE_M_PAD = 8       # padded decode row count (row 0 = live query)


def gate_logits(hidden_row: np.ndarray, gate_weight: np.ndarray) -> np.ndarray:
    """hidden_row: [HIDDEN] (bf16-range values, kept f32 in numpy).
    gate_weight: [HIDDEN, EXPERTS].
    returns: [EXPERTS] f32 logits, mirrors the row-0 slice of the on-device
    bf16-in/f32-accumulate gate GEMM (gemm_expert_tile + row0 extract)."""
    assert hidden_row.ndim == 1 and gate_weight.ndim == 2
    assert hidden_row.shape[0] == gate_weight.shape[0]
    return (hidden_row.astype(np.float32) @ gate_weight.astype(np.float32)).astype(
        np.float32
    )


def topk_selection_sort(
    logits: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Iterated argmax + mask-to-(-inf), K times -- mirrors
    moe_topk_router_detail::argmax_full_reduce called kTopK times with the
    winner masked out of logits_buf between passes. First-index-wins on ties,
    matching the device's left-to-right max_cmp/select reduction (running_val
    starts as strictly-less-than any real logit, and reduce_chunk only
    replaces the running max on a STRICT greater-than from max_cmp)."""
    work = logits.astype(np.float32).copy()
    n = work.shape[0]
    assert 0 < k <= n
    top_values = np.empty(k, dtype=np.float32)
    top_indices = np.empty(k, dtype=np.int32)
    neg_inf = np.float32(-3.0e38)
    for i in range(k):
        idx = int(np.argmax(work))  # first-max-wins on ties, like max_cmp
        top_values[i] = work[idx]
        top_indices[i] = idx
        work[idx] = neg_inf
    return top_values, top_indices


def make_padded_query(hidden_row: np.ndarray, m_pad: int = MOE_M_PAD) -> np.ndarray:
    """Pad a single [HIDDEN] query row to [m_pad, HIDDEN] with zero rows,
    matching the kernel's M=1-padded-to-kMPad convention."""
    h = hidden_row.shape[0]
    out = np.zeros((m_pad, h), dtype=np.float32)
    out[0, :] = hidden_row
    return out


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    hidden_row = rng.standard_normal(MOE_HIDDEN).astype(np.float32) * 0.1
    padded = make_padded_query(hidden_row, MOE_M_PAD)
    gate_weight = rng.standard_normal((MOE_HIDDEN, MOE_EXPERTS)).astype(np.float32) * 0.1

    # Row 0 is the only live row; rows 1.. dotted with any weight column
    # must reduce to exactly 0, mirroring gemv-int8's padding-row assertion.
    logits_full = padded @ gate_weight
    assert logits_full.shape == (MOE_M_PAD, MOE_EXPERTS)
    assert np.allclose(logits_full[1:, :], 0.0), "padding rows must reduce to ~0"

    logits = gate_logits(hidden_row, gate_weight)
    assert np.allclose(logits, logits_full[0, :], atol=1e-4)

    top_values, top_indices = topk_selection_sort(logits, MOE_TOPK)

    # Cross-check against np.argsort's top-K (order-agnostic set compare;
    # np.argsort tie-breaking can differ from the device's first-wins rule,
    # so compare the SET of selected experts + the max value only).
    ref_order = np.argsort(-logits)[:MOE_TOPK]
    assert set(top_indices.tolist()) == set(ref_order.tolist()), (
        "selection-sort topk must pick the same expert SET as a full sort"
    )
    assert top_values[0] == logits.max(), "top_values[0] must be the global max"
    assert np.all(np.diff(top_values) <= 0), "top_values must be descending"

    err = rel_l2(np.sort(top_values)[::-1], np.sort(logits)[::-1][:MOE_TOPK])
    print(f"self-check rel-L2 (topk values vs full-sort head): {err:.3e}")
    assert err < 1e-6

    print(
        "golden.py self-check OK: "
        f"hidden{hidden_row.shape} @ Wg{gate_weight.shape} -> "
        f"logits{logits.shape}, top{MOE_TOPK} values={top_values}, "
        f"indices={top_indices}"
    )
