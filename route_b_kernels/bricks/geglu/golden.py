#!/usr/bin/env python3
"""Host-side numpy golden reference for the `geglu` brick.

geglu(input) = GELU(gate) * up

Row-stream contract (matches geglu.cc / glu.cc):
  input row layout, per resident [tile, D] row, is the concatenation
    [up (cols) | gate (cols)]
  i.e. input.shape == (tile, 2*cols), output.shape == (tile, cols).

GELU here is the tanh approximation (torch gelu(approximate="tanh")):
  gelu(x) = 0.5*x*(1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))

This is the reference the later DEVICE pass checks the on-chip kernel
against via relative L2 error (see verify_spec in the task return / this
brick's README-less pointer: threshold 3e-2, consistent with the other
route_b_kernels bricks' bf16-tanh-SFU tolerance).
"""
import numpy as np


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """torch gelu(approximate='tanh'), computed in f64/f32 (host precision)."""
    c0 = np.sqrt(2.0 / np.pi)
    c1 = 0.044715
    inner = c0 * (x + c1 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def geglu(up: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """out = gelu(gate) * up, elementwise. up/gate shape (tile, cols)."""
    assert up.shape == gate.shape
    return gelu_tanh(gate) * up


def geglu_row_stream(input_rows: np.ndarray, cols: int) -> np.ndarray:
    """Row-stream form matching the kernel's [up|gate] concatenated-row
    contract: input_rows.shape == (tile, 2*cols) -> output (tile, cols)."""
    assert input_rows.shape[-1] == 2 * cols
    up = input_rows[..., :cols]
    gate = input_rows[..., cols:]
    return geglu(up, gate)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel()) + 1e-12
    return float(num / den)


if __name__ == "__main__":
    # Self-check + shape the later NPU verify pass should reuse.
    rng = np.random.default_rng(0)

    TILE = 32   # rows per resident tile (matches EPI_M-style tile conventions)
    COLS = 64   # per-half width; must be a multiple of 16 (kernel's N=16 lanes)

    up = rng.standard_normal((TILE, COLS)).astype(np.float32)
    gate = rng.standard_normal((TILE, COLS)).astype(np.float32)
    row_input = np.concatenate([up, gate], axis=-1).astype(np.float32)

    out_direct = geglu(up, gate)
    out_row = geglu_row_stream(row_input, COLS)

    assert rel_l2(out_direct, out_row) < 1e-12, "row-stream form must match direct form exactly (both host f64/f32)"

    print(f"geglu golden self-check OK: input {row_input.shape} -> output {out_row.shape}")
    print(f"sample out[0, :4] = {out_row[0, :4]}")
