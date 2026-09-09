# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def generate_golden_reference_tmatvec(
    M=128, K=2048, num_batches=16, batch_group=2, alloc_K=None, seed=42
):
    """Golden for the transposed-A contraction: C[b][j] = sum_p W[b][p] * A[b//batch_group][p][j].

    Note the contraction is over ROWS of A, so this is NOT gemv's golden with A transposed on the
    host -- writing it that way would test the host's transpose, not the kernel's.

    With alloc_K set, A is allocated with alloc_K rows per matrix and only the first K are reduced.
    Rows past K are POISONED: unlike gemv's window, these DO reach every output if the reduction
    extent is wrong, so here the poison catches both a wrong extent AND a wrong per-matrix stride.
    """
    import torch

    assert num_batches % batch_group == 0
    _AK = K if alloc_K is None else alloc_K
    assert _AK >= K
    torch.manual_seed(seed)
    n_matrices = num_batches // batch_group
    A = torch.randn(n_matrices, _AK, M, dtype=torch.bfloat16)
    A[:, K:, :] = 1000.0
    W = torch.randn(num_batches, K, dtype=torch.bfloat16)
    C = torch.empty(num_batches, M, dtype=torch.bfloat16)
    for b in range(num_batches):
        # f32 accumulation, matching the kernel's accum<accfloat> -- a bf16 accumulation here
        # would make the golden itself the least accurate thing in the comparison.
        C[b] = (W[b].float() @ A[b // batch_group, :K].float()).to(torch.bfloat16)
    return {"A": A, "W": W, "C": C}
