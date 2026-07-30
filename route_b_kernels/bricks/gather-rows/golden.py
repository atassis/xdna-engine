#!/usr/bin/env python3
"""Host numpy golden for route_b_kernels/bricks/gather-rows/gather_rows.cc.

N-row vector gather: given a codebook [n_rows, D] and integer indices [T], produce [T, D] =
numpy.take(codebook, idx, axis=0), after clamping idx to [0, n_rows-1].

This is the RVQ codebook lookup in scripts/codec_quantizer_ref.py::rvq_lookup (~line 442):
    idx = np.clip(codes[cb].astype(np.int64), 0, size - 1)
    gathered = codebook[idx]                                 # [T, codebook_dim]
Real shapes (codec_quantizer_ref.py:205-207, kernel-map line 85): codebook_dim (D) = 8;
n_rows = SEMANTIC_CODEBOOK_SIZE = 4096 for codebook 0 (semantic), n_rows =
RESIDUAL_CODEBOOK_SIZE = 1024 for codebooks 1..9 (residual). This golden is the gather step
alone, matched to the brick's contract (one streamed operand + one resident operand) -- the
out_proj GEMM that follows in rvq_lookup is a separate, already-existing brick
(gemm-bf16xbfp16 / gemm-bfp16-ebs8), not reimplemented here.

Usage: python3 golden.py
"""
import numpy as np


def gather_rows_ref(codebook, idx):
    """codebook: [n_rows, D] float. idx: [T] int (any integer dtype). -> [T, D] float.

    Clamp matches AudioCodec::decode's inline clamp_code lambda / rvq_lookup's np.clip: an
    out-of-range index is clamped to the nearest valid row, not an error -- a silent OOB read
    on device is the alternative, so the kernel implements this unconditionally and this
    golden must match that, not just the "codes are always valid" happy path.
    """
    codebook = np.asarray(codebook)
    n_rows = codebook.shape[0]
    idx = np.clip(np.asarray(idx).astype(np.int64), 0, n_rows - 1)
    return np.take(codebook, idx, axis=0)


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else float(num)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Same shapes verify_gather_rows.py gates at: the residual-codebook regime (n_rows=1024,
    # D=8) is what fits a core tile resident in f32 -- see gather_rows.cc's header.
    N_ROWS, D, T = 1024, 8, 48
    codebook = rng.standard_normal((N_ROWS, D)).astype(np.float32)

    # Non-trivial index pattern: repeats, row 0, row n_rows-1, and both an out-of-range and a
    # negative index to exercise the clamp in both directions -- an identity/ramp pattern
    # would pass even if the gather silently degenerated to something index-independent.
    idx = rng.integers(0, N_ROWS, size=T).astype(np.int32)
    idx[0] = 0
    idx[1] = N_ROWS - 1
    idx[2] = idx[5] = 17            # deliberate repeat
    idx[3] = N_ROWS + 50            # out-of-range -> must clamp to N_ROWS - 1
    idx[4] = -3                     # negative -> must clamp to 0 (codes are >=0 in practice;
                                     # clamp_code's clamp is unconditional in the C++, so this
                                     # golden and the kernel both handle it regardless)

    got = gather_rows_ref(codebook, idx)
    exp = codebook[np.clip(idx, 0, N_ROWS - 1)]
    print("shape:", got.shape)
    print("rel-L2 vs manual clip+index:", rel_l2(got, exp))
    assert np.array_equal(got, exp), "gather_rows_ref must match numpy.take + clip exactly"

    # clamp behaviour, explicitly, matching rvq_lookup's np.clip(idx, 0, size - 1)
    assert np.array_equal(got[3], codebook[N_ROWS - 1]), "out-of-range index must clamp to last row"
    assert np.array_equal(got[4], codebook[0]), "negative index must clamp to row 0"
    assert np.array_equal(got[2], got[5]), "repeated index must gather identical rows"
    assert np.array_equal(got[0], codebook[0]) and np.array_equal(got[1], codebook[N_ROWS - 1])
    print("clamp behaviour OK (idx[3] -> last row, idx[4] -> row 0, idx[2]==idx[5] repeat)")
    print("PASS")
