# SPDX-License-Identifier: Apache-2.0
"""Host tests for decode_flash_ref: the global-layer decode flash operator's block model.

  PYTHONPATH=designs/decode_fused .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_decode_flash_ref.py -v
"""
import numpy as np
import pytest

from decode_flash_ref import (blocks_per_column, decode_flash_attention, flash_slot_writes,
                              last_block_index, merge)
from flash_attn_ref import full_attention

HQ, HD, CAP = 16, 512, 4096


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def qkv(seed, cap=CAP):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((HQ, HD), dtype=np.float32) / np.sqrt(HD)
    return q, rng.standard_normal((cap, HD), dtype=np.float32), rng.standard_normal((cap, HD), dtype=np.float32)


@pytest.mark.parametrize("n_live,want", [(1, 1), (64, 1), (65, 1), (512, 1), (513, 2), (262144, 512)])
def test_blocks_per_column(n_live, want):
    assert blocks_per_column(n_live) == want


@pytest.mark.parametrize("cap", [4096, 262144])
def test_last_block_never_passes_capacity(cap):
    for n_live in list(range(1, 2049)) + [cap - 1, cap]:
        assert last_block_index(n_live, 7) <= cap // 64 - 1


@pytest.mark.parametrize("n_live", [1, 63, 64, 447, 448, 1000, 4096])
def test_matches_f64_attention(n_live):
    q, k, v = qkv(n_live)
    got = decode_flash_attention(q, k, v, n_live)
    ref = full_attention(q, k[:n_live], v[:n_live], np.ones((HQ, n_live), bool), 1.0)
    assert rel_l2(got, ref) < 2e-2


@pytest.mark.parametrize("n_live", [1, 100, 447, 1000])
@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
def test_poison_past_n_live_does_not_leak(n_live, poison):
    q, k, v = qkv(7)
    clean = decode_flash_attention(q, k, v, n_live)
    k2, v2 = k.copy(), v.copy()
    k2[n_live:] = poison
    v2[n_live:] = poison
    np.testing.assert_array_equal(decode_flash_attention(q, k2, v2, n_live), clean)


def test_column_count_changes_order_not_result():
    q, k, v = qkv(3)
    assert rel_l2(decode_flash_attention(q, k, v, 3000, columns=1),
                  decode_flash_attention(q, k, v, 3000, columns=8)) < 1e-2


def test_merge_skips_an_empty_leading_partial():
    hq, hd = HQ, HD
    empty = (np.full(hq, -np.inf, np.float32), np.zeros(hq, np.float32), np.zeros((hq, hd), np.float32))
    rng = np.random.default_rng(9)
    real = (rng.standard_normal(hq, dtype=np.float32),
            rng.random(hq, dtype=np.float32) + 0.1,
            rng.standard_normal((hq, hd), dtype=np.float32))
    got = merge([empty, real])
    want = merge([real])
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got, want)


def test_columns_without_a_live_block_fold_as_no_ops():
    q, k, v = qkv(5)
    got = decode_flash_attention(q, k, v, 100)       # blocks 0,1 live; columns 2..7 empty
    ref = full_attention(q, k[:100], v[:100], np.ones((HQ, 100), bool), 1.0)
    assert np.isfinite(got).all() and rel_l2(got, ref) < 2e-2


def _slots(block=64, columns=8, capacity=CAP):
    return [{"len_param": "gf_len0", "loop_param": "gf_loop0",
             "block": block, "columns": columns, "capacity": capacity}]


def test_flash_slot_writes_empty_list():
    assert flash_slot_writes([], 0) == []


def test_flash_slot_writes_pos_0():
    assert flash_slot_writes(_slots(), 0) == [("gf_len0", 0), ("gf_loop0", 1)]


def test_flash_slot_writes_pos_512():
    assert flash_slot_writes(_slots(block=64, columns=8), 512) == [("gf_len0", 1), ("gf_loop0", 2)]


def test_flash_slot_writes_last_position_in_capacity():
    nb = CAP // 512
    assert flash_slot_writes(_slots(), CAP - 1) == [("gf_len0", nb - 1), ("gf_loop0", nb)]


def test_flash_slot_writes_past_capacity_raises():
    with pytest.raises(ValueError):
        flash_slot_writes(_slots(), CAP)
