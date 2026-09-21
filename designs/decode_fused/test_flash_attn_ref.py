# SPDX-License-Identifier: Apache-2.0
"""Host tests for flash_attn_ref: the fused prefill attention's block model against one-shot
softmax attention, with the prefill generator's own mask functions.

Run inside the IRON env (the mask functions live in gen_llm_prefill):
  source scripts/amd_paths.sh
  PYTHONPATH=designs/decode_fused:$IRON_DIR .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_flash_attn_ref.py -v
"""
import numpy as np
import pytest

gen = pytest.importorskip("gen_llm_prefill")
from flash_attn_ref import (flash_attention, full_attention,  # noqa: E402
                            visible_from_ring_rows, visible_from_widths)
from test_prefill_ring import expand_mask  # noqa: E402

M, B_KV, RING = 64, 64, 1024
TOL = 1e-5


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def qkv(seed, rows, cols, hd):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, hd), dtype=np.float32),
            rng.standard_normal((cols, hd), dtype=np.float32),
            rng.standard_normal((cols, hd), dtype=np.float32))


def global_visible(base, w, heads):
    return visible_from_widths(gen.causal_widths(base, M, w, heads), w)


def sliding_visible(base, grp):
    return visible_from_ring_rows(np.tile(gen.ring_mask_rows(base, M, RING), (grp, 1)), RING + M)


@pytest.mark.parametrize("base", [0, 1024, 2048 - M])
def test_widths_visibility_is_causal_and_clamped(base):
    w, heads = 2048, 2
    vis = global_visible(base, w, heads)
    want = np.tile(np.clip(base + np.arange(M) + 1, 1, w), heads)
    assert vis.shape == (heads * M, w)
    assert np.array_equal(vis.sum(axis=1), want)
    # a prefix: the running count climbs by one per column until the row's width, then stops
    assert (vis.cumsum(axis=1) == np.minimum(np.arange(1, w + 1)[None, :], want[:, None])).all()


@pytest.mark.parametrize("base", [0, 512, 1536, 2048])
def test_ring_visibility_matches_the_ring_tests_own_mask(base):
    rows = gen.ring_mask_rows(base, M, RING)
    assert np.array_equal(visible_from_ring_rows(rows, RING + M), expand_mask(rows, RING + M))


@pytest.mark.parametrize("base", [0, 1024, 2048 - M])
def test_global_flash_matches_full_attention(base):
    w, hd, heads = 2048, 512, 16
    q, k, v = qkv(base, heads * M, w, hd)
    vis = global_visible(base, w, heads)
    scale = 1.0 / np.sqrt(hd)
    out = flash_attention(q, k, v, vis, scale, b_kv=B_KV)
    assert np.isfinite(out).all()
    assert rel_l2(out, full_attention(q, k, v, vis, scale)) < TOL


@pytest.mark.parametrize("base", [0, 512, 1536, 2048])
def test_sliding_flash_matches_full_attention(base):
    hd, grp = 256, 2
    q, k, v = qkv(100 + base, grp * M, RING + M, hd)
    vis = sliding_visible(base, grp)
    scale = 1.0 / np.sqrt(hd)
    out = flash_attention(q, k, v, vis, scale, b_kv=B_KV)
    assert np.isfinite(out).all()
    assert rel_l2(out, full_attention(q, k, v, vis, scale)) < TOL


def _holed_case():
    q, k, v = qkv(9, 2 * M, RING + M, 256)
    vis = sliding_visible(1536, 2)
    hidden = ~vis.any(axis=0)
    assert hidden[512] and hidden.sum() == 1   # slot 512: hidden for every row, inside block 8
    return q, k, v, vis, hidden


def test_nan_keys_and_huge_values_in_hidden_slots_change_nothing():
    q, k, v, vis, hidden = _holed_case()
    kp, vp = k.copy(), v.copy()
    kp[hidden], vp[hidden] = np.nan, 1.0e4
    clean = flash_attention(q, k, v, vis, 1 / 16, b_kv=B_KV)
    assert np.array_equal(clean, flash_attention(q, kp, vp, vis, 1 / 16, b_kv=B_KV))


def test_a_nan_value_in_a_hidden_slot_is_not_survivable():
    # 0 * NaN is NaN in the x V product, so correct masking cannot protect against a NaN V row:
    # the KV cache tail must hold finite values (it is zero-filled today).
    q, k, v, vis, hidden = _holed_case()
    vp = v.copy()
    vp[hidden] = np.nan
    assert np.isnan(flash_attention(q, k, vp, vis, 1 / 16, b_kv=B_KV)).any()


def test_negative_unguarded_empty_block_is_nan():
    q, k, v = qkv(7, 2 * M, RING + M, 256)
    out = flash_attention(q, k, v, sliding_visible(0, 2), 1 / 16, b_kv=B_KV, guard_empty=False)
    assert np.isnan(out).all()


def test_negative_no_rescale_misses_tolerance():
    w, hd = 1024, 512
    q, k, v = qkv(3, M, w, hd)
    k *= np.linspace(0.1, 3.0, w, dtype=np.float32)[:, None]   # later blocks raise the max
    vis = global_visible(w - M, w, 1)
    ref = full_attention(q, k, v, vis, 1 / np.sqrt(hd))
    out = flash_attention(q, k, v, vis, 1 / np.sqrt(hd), b_kv=B_KV, rescale=False)
    assert rel_l2(out, ref) > 100 * TOL


def test_negative_additive_mask_lets_a_nan_key_through():
    q, k, v, vis, hidden = _holed_case()
    kp = k.copy()
    kp[hidden] = np.nan
    assert np.isnan(flash_attention(q, kp, v, vis, 1 / 16, b_kv=B_KV, additive_mask=True)).any()
