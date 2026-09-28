# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from iron.common.quant import quantize_weight
import weight_views as wv


@pytest.mark.parametrize("N,K", [(64, 3840), (32, 15360)])
def test_planar_blob_round_trips_to_header_first_rows(N, K):
    rng = np.random.default_rng(0)
    W = rng.standard_normal((N, K)).astype(np.float32)
    rows = quantize_weight(W, 32, "int4", scale_dtype="bf16")
    G = wv.row_group_for(K)
    planar = quantize_weight(W, 32, "int4", scale_dtype="bf16", layout="row_group_planar", row_group=G)
    got = wv.planar_to_rows(planar, N, K, G)
    assert np.array_equal(got.view(np.uint8).reshape(-1), np.asarray(rows).view(np.uint8).reshape(-1))


def test_codes_and_scales_split():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((8, 256)).astype(np.float32)
    rows = quantize_weight(W, 32, "int4", scale_dtype="bf16")
    codes, scales = wv.codes_and_scales(rows, 8, 256, scale_dtype="bf16")
    assert codes.shape == (8, 256) and codes.dtype == np.int8 and codes.min() >= -8 and codes.max() <= 7
    assert scales.shape == (8, 8) and scales.dtype == np.float32


def test_row_group_for_matches_served_artifact_shapes():
    assert wv.row_group_for(3840) == 2
    assert wv.row_group_for(15360) == 1
    assert wv.row_group_for(8192) == 1


def test_qkv_rows_slices_q_k_v_in_row_order():
    rows = np.arange(20 * 4, dtype=np.uint8).reshape(20, 4)
    parts = wv.qkv_rows(rows, qd=12, kvd=4, has_v=True)
    assert np.array_equal(parts["q"], rows[:12])
    assert np.array_equal(parts["k"], rows[12:16])
    assert np.array_equal(parts["v"], rows[16:20])


def test_qkv_rows_no_v_on_global_layers():
    rows = np.arange(16 * 4, dtype=np.uint8).reshape(16, 4)
    parts = wv.qkv_rows(rows, qd=12, kvd=4, has_v=False)
    assert set(parts) == {"q", "k"}
