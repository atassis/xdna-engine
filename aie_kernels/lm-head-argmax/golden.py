#!/usr/bin/env python3
"""Host-side golden reference for the lm-head-argmax brick.

BRICK: lm-head-argmax  [group: head]

    logits[VOCAB] = hidden[HIDDEN] @ W[HIDDEN, VOCAB]     (bf16 in, f32 accum)
    best_value, best_index = argmax_v(logits)

This mirrors lm_head_argmax.cc exactly: bf16-rounded inputs, f32 GEMM
accumulation, then a value+index co-produced argmax (the device kernel's
running max_cmp/select reduction tree). The DEVICE pass (not this spike)
should compare its on-device (best_value, best_index) against this golden
via rel-L2 on best_value and exact match on best_index (see verify_spec
below / in the task's return payload).

No device/runtime deps -- numpy only, so this is runnable standalone in any
CPU-only dev environment while the leaf that wires up the NPU xclbin/test
harness is still pending.
"""
import numpy as np

# Must match LMHEAD_HIDDEN / LMHEAD_VOCAB / LMHEAD_M_PAD in lm_head_argmax.cc
# (kept as defaults here; a device-pass leaf should parameterize both sides
# from the same source of truth, e.g. a shared config, when it lands).
HIDDEN = 256
VOCAB = 512
M_PAD = 8  # decode row-padding factor; only row 0 (the live query) is real


def to_bf16(x: np.ndarray) -> np.ndarray:
    """Round f32 -> bf16 (round-to-nearest-even) and back to f32, matching
    the device's bf16 storage format for `hidden` and `weight`."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    rounded = (bits.astype(np.uint64) + rounding_bias) & np.uint64(0xFFFF0000)
    return rounded.astype(np.uint32).view(np.float32)


def lm_head_argmax_golden(hidden_row: np.ndarray, weight: np.ndarray):
    """
    hidden_row: [HIDDEN] f32 (a single decode query row; this is the "row 0"
                of the device's [M_PAD, HIDDEN] padded tile -- the other
                M_PAD-1 rows are zero-padding, invisible to this reference).
    weight:     [HIDDEN, VOCAB] f32 (vocab projection matrix, W = lm_head^T
                or whatever layout the model ships; row-major [HIDDEN, VOCAB]
                here for clarity -- the device kernel's on-chip layout tiles
                this by [K_TILE, N_TILE] but is numerically equivalent).

    Returns (best_value: float32 scalar, best_index: int64 scalar).
    """
    assert hidden_row.shape == (HIDDEN,)
    assert weight.shape == (HIDDEN, VOCAB)

    h_bf16 = to_bf16(hidden_row)
    w_bf16 = to_bf16(weight)

    # f32 accumulation over bf16-rounded inputs, exactly like the device's
    # aie::mmul<M,K,N,bfloat16,bfloat16,accauto> accumulator.
    logits = (h_bf16.astype(np.float64) @ w_bf16.astype(np.float64)).astype(np.float32)

    best_index = int(np.argmax(logits))
    best_value = float(logits[best_index])
    return best_value, best_index


def _make_reference_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    hidden_row = rng.standard_normal(HIDDEN).astype(np.float32)
    weight = (rng.standard_normal((HIDDEN, VOCAB)).astype(np.float32) * 0.02)
    return hidden_row, weight


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(b)
    if denom == 0.0:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


if __name__ == "__main__":
    hidden_row, weight = _make_reference_inputs()
    best_value, best_index = lm_head_argmax_golden(hidden_row, weight)

    # Sanity cross-check against a plain f32 (non-bf16-rounded) GEMM+argmax,
    # to bound how much bf16 rounding alone can perturb the winning index.
    logits_f32 = hidden_row.astype(np.float64) @ weight.astype(np.float64)
    ref_index = int(np.argmax(logits_f32))
    ref_value = float(logits_f32[ref_index])

    print(f"HIDDEN={HIDDEN} VOCAB={VOCAB} M_PAD={M_PAD}")
    print(f"bf16-path   best_index={best_index} best_value={best_value:.6f}")
    print(f"f32-ref     best_index={ref_index} best_value={ref_value:.6f}")
    print(f"index match: {best_index == ref_index}")
    print(f"value rel_l2 (bf16 vs f32 ref, same index): "
          f"{rel_l2([best_value], [logits_f32[best_index]]):.3e}")
