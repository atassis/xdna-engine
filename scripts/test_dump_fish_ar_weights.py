# SPDX-License-Identifier: Apache-2.0
"""Tests for dump_fish_ar_weights.py -- Fish dual-AR checkpoint -> .npy, HF-style keys.

Run:
  .venv-iron/bin/python -m pytest scripts/test_dump_fish_ar_weights.py -v
"""
import glob
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dump_fish_ar_weights import (  # noqa: E402
    Dims, count_layers, derive_dims, dump, dump_ar_layer, dump_slow_ar_layer, permute_heads,
    rope_row_permutation, Source, STACKS,
)

S2_GGUF = os.environ.get("S2_GGUF_PATH", os.path.join(_HERE, "..", "s2.cpp", "models", "s2-pro-q6_k.gguf"))
_S1 = glob.glob("/mnt/data/cache/huggingface/hub/models--fishaudio--openaudio-s1-mini/snapshots/*/model.pth")
S1_PTH = os.environ.get("S1_MINI_PTH", _S1[0] if _S1 else "")

needs_s2 = pytest.mark.skipif(not os.path.exists(S2_GGUF), reason=f"no S2-Pro GGUF at {S2_GGUF}")
needs_s1 = pytest.mark.skipif(not os.path.exists(S1_PTH), reason="no s1-mini model.pth")


class TestRopePermutation:
    """The permutation is the whole reason this module rewrites weights, so it is asserted against
    the two rotations directly rather than against a golden array."""

    @staticmethod
    def _rotations(head_dim, pos=7.0, base=1e6):
        half = head_dim // 2
        inv = (base ** (-2.0 / head_dim)) ** np.arange(half)
        th = pos * inv
        cos, sin = np.cos(th), np.sin(th)

        def adjacent(x):                       # ggml GGML_ROPE_TYPE_NORMAL
            xr = x.reshape(half, 2)
            return np.stack([xr[:, 0] * cos - xr[:, 1] * sin,
                             xr[:, 0] * sin + xr[:, 1] * cos], axis=-1).reshape(head_dim)

        def neox(x):                           # the rail's rope operator
            x1, x2 = x[:half], x[half:]
            return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin])

        return adjacent, neox

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_permuting_makes_neox_compute_the_adjacent_pair_result(self, head_dim):
        adjacent, neox = self._rotations(head_dim)
        v = np.random.default_rng(0).standard_normal(head_dim)
        perm = rope_row_permutation(head_dim)
        assert np.allclose(neox(v[perm]), adjacent(v)[perm], atol=1e-12)

    def test_the_two_conventions_genuinely_differ(self):
        """Without this the test above would pass on a no-op permutation."""
        adjacent, neox = self._rotations(128)
        v = np.random.default_rng(0).standard_normal(128)
        rel = np.linalg.norm(neox(v) - adjacent(v)) / np.linalg.norm(adjacent(v))
        assert rel > 0.5, rel

    def test_permute_heads_is_per_head_not_whole_matrix(self):
        w = np.arange(2 * 4 * 3).reshape(2 * 4, 3)      # 2 heads, head_dim 4, 3 columns
        got = permute_heads(w, n_heads=2, head_dim=4)
        assert np.array_equal(got[:4], w[[0, 2, 1, 3]])
        assert np.array_equal(got[4:], w[[4, 6, 5, 7]])

    def test_permute_heads_is_an_involution_here(self):
        """head_dim 4 makes [0,2,1,3] self-inverse; 128 does not, so this pins the SHAPE of the
        operation and not a coincidence -- the round trip below is the general claim."""
        w = np.random.default_rng(1).standard_normal((3 * 128, 5))
        perm = rope_row_permutation(128)
        back = permute_heads(w, 3, 128).reshape(3, 128, 5)[:, np.argsort(perm), :].reshape(w.shape)
        assert np.array_equal(back, w)


@needs_s1
class TestS1MiniPth:
    def test_dims_are_derived_from_the_tensors(self):
        d = derive_dims(Source(S1_PTH))
        assert d == Dims(d_model=1024, head_dim=128, n_q_heads=16, n_kv_heads=8, ffn=3072,
                         n_layers=28, vocab=155776, tied_embeddings=False)

    def test_layer0_keys_and_shapes(self):
        src = Source(S1_PTH)
        d = derive_dims(src)
        out = dump_slow_ar_layer(src, d, 0)
        assert out["model.layers.0.self_attn.q_proj.weight"].shape == (2048, 1024)
        assert out["model.layers.0.self_attn.k_proj.weight"].shape == (1024, 1024)
        assert out["model.layers.0.self_attn.v_proj.weight"].shape == (1024, 1024)
        assert out["model.layers.0.mlp.gate_proj.weight"].shape == (3072, 1024)
        assert out["model.layers.0.mlp.down_proj.weight"].shape == (1024, 3072)

    def test_v_and_o_are_untouched_by_the_permutation(self):
        """v is never rotated and o consumes ctx, so permuting either would be a real bug."""
        src = Source(S1_PTH)
        d = derive_dims(src)
        on = dump_slow_ar_layer(src, d, 0, permute_rope=True)
        off = dump_slow_ar_layer(src, d, 0, permute_rope=False)
        for k in ("self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight"):
            assert np.array_equal(on["model.layers.0." + k], off["model.layers.0." + k]), k
        for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.q_norm.weight"):
            assert not np.array_equal(on["model.layers.0." + k], off["model.layers.0." + k]), k

    def test_dump_writes_the_untied_head_and_a_manifest(self, tmp_path):
        d = dump(S1_PTH, str(tmp_path), layers=1)
        assert d.tied_embeddings is False
        assert (tmp_path / "model.lm_head.weight.npy").exists()
        assert (tmp_path / "model.embed_tokens.weight.npy").exists()
        m = json.loads((tmp_path / "dump_manifest.json").read_text())
        assert m["rope_rows_permuted"] is True and m["layers_written"] == 1
        assert m["dims"]["n_layers"] == 28


@needs_s2
class TestS2ProGguf:
    def test_native_naming_is_what_we_bridge_from(self):
        names = Source(S2_GGUF).names()
        assert "layers.0.attention.wqkv.weight" in names
        assert "model.layers.0.self_attn.q_proj.weight" not in names

    def test_dims_are_derived_from_the_tensors(self):
        d = derive_dims(Source(S2_GGUF))
        assert (d.d_model, d.head_dim, d.n_q_heads, d.n_kv_heads, d.ffn) == (2560, 128, 32, 8, 9728)
        assert d.n_layers == 36 and d.tied_embeddings is True

    def test_count_layers_auto_detects_the_full_stack(self):
        assert count_layers(Source(S2_GGUF)) == 36

    def test_a_tied_model_writes_no_separate_head(self, tmp_path):
        dump(S2_GGUF, str(tmp_path), layers=1)
        assert not (tmp_path / "model.lm_head.weight.npy").exists()
        assert (tmp_path / "model.embed_tokens.weight.npy").exists()


@needs_s1
class TestS1MiniFastStack:
    """The Fast AR is the same file's OTHER stack: 4 layers at head_dim 64, no per-head norms."""

    def test_head_dim_cannot_be_derived_and_says_so(self):
        """Two equations, three unknowns. The failure has to name the fix, because the sibling
        stack's 128 is a plausible-looking wrong answer sitting right there."""
        with pytest.raises(ValueError, match="fast_head_dim"):
            derive_dims(Source(S1_PTH), stack="fast")

    def test_dims_with_the_head_dim_supplied(self):
        d = derive_dims(Source(S1_PTH), stack="fast", head_dim=64)
        assert d == Dims(d_model=1024, head_dim=64, n_q_heads=16, n_kv_heads=8, ffn=3072,
                         n_layers=4, vocab=4096, tied_embeddings=False, stack="fast")

    def test_count_layers_does_not_confuse_the_two_stacks(self):
        src = Source(S1_PTH)
        assert count_layers(src, "slow") == 28
        assert count_layers(src, "fast") == 4

    def test_the_fast_layer_emits_no_qk_norm(self):
        src = Source(S1_PTH)
        d = derive_dims(src, stack="fast", head_dim=64)
        out = dump_ar_layer(src, d, 0)
        assert "model.layers.0.self_attn.q_norm.weight" not in out
        assert out["model.layers.0.self_attn.q_proj.weight"].shape == (1024, 1024)
        assert out["model.layers.0.self_attn.k_proj.weight"].shape == (512, 1024)

    def test_dump_reads_head_dim_from_the_checkpoints_config(self, tmp_path):
        """No --head-dim passed: config.json beside model.pth is the authority."""
        d = dump(S1_PTH, str(tmp_path), layers=1, stack="fast")
        assert d.head_dim == 64 and d.stack == "fast"
        m = json.loads((tmp_path / "dump_manifest.json").read_text())
        assert m["stack"] == "fast" and m["dims"]["vocab"] == 4096
        assert (tmp_path / "model.lm_head.weight.npy").exists()
