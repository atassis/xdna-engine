#!/usr/bin/env python3
"""Host numpy golden for aie_kernels/embedding-gather/embedding_gather.cc.

Row gather from an L3-resident table: given `table[n_rows, D]` and integer indices `idx[T]`,
produce `table[idx]` -- exactly `ARWeights.embedding_rows`/`.codebook_embedding_rows`/
`.fast_embedding_rows` (scripts/s2_ar_ref.py:465-481), each `block[ids - lo]`, no compute.

No clamp here, unlike aie_kernels/gather-rows/golden.py: this brick's kernel never
sees an index (the DMA already selected the row via a host-computed offset before the core
runs), so an out-of-range idx is the CALLER's bug, not something this golden should paper over.
`verify_embedding_gather.py`'s driver clamps before writing the offset and this golden takes
the same already-clamped indices, so the two stay comparable.

Usage: python3 golden.py
"""
import numpy as np


def embedding_gather_ref(table, idx):
    """table: [n_rows, D] float. idx: [T] int, already in [0, n_rows). -> [T, D] float."""
    table = np.asarray(table)
    idx = np.asarray(idx).astype(np.int64)
    assert np.all((idx >= 0) & (idx < table.shape[0])), "idx out of range -- clamp before calling"
    return np.take(table, idx, axis=0)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # D=2560 matches all three real AR tables (embeddings.weight, codebook_embeddings.weight,
    # fast_embeddings.weight -- see docs/s2-embedding-gather-design.md); N_ROWS is a small
    # synthetic stand-in, not a production shape (the mechanism is size-independent, see the
    # design doc's table-size section).
    N_ROWS, D, T = 64, 2560, 6
    table = rng.standard_normal((N_ROWS, D)).astype(np.float32)

    idx = rng.integers(0, N_ROWS, size=T).astype(np.int32)
    idx[0] = 0
    idx[1] = N_ROWS - 1
    idx[2] = idx[3] = 17   # deliberate repeat

    got = embedding_gather_ref(table, idx)
    exp = table[idx]
    print("shape:", got.shape)
    print("rel-L2 vs manual index:", rel_l2(got, exp))
    assert np.array_equal(got, exp)
    assert np.array_equal(got[0], table[0]) and np.array_equal(got[1], table[N_ROWS - 1])
    assert np.array_equal(got[2], got[3])
    print("PASS")
