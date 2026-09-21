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
