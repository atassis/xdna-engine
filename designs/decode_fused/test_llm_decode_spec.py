# SPDX-License-Identifier: Apache-2.0
"""Tests for LlmSpec.check_prefill / check_prefill_seq / legal_prefill_batches.

Run:
  PYTHONPATH=designs/decode_fused .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_llm_decode_spec.py -v
"""
import pytest

from llm_decode_spec import GEMMA3_270M, QWEN3_0_6B


class TestCheckPrefillQwen:
    """qwen3-0.6b: d_model=1024, q_dim=2048, kv_dim=1024, ffn=3072, head_dim=128 -- every
    batch-only projection (qkv/o/gate/up/down) already divides the default tile_n=64/cols=8
    config, so only the batch modulus and the ctx op need anything special."""

    def test_legal_at_256(self):
        QWEN3_0_6B.check_prefill(256)

    def test_legal_at_512(self):
        QWEN3_0_6B.check_prefill(512)

    @pytest.mark.parametrize("batch", [1, 32, 64, 128])
    def test_illegal_batch_not_multiple_of_256(self, batch):
        with pytest.raises(ValueError) as exc:
            QWEN3_0_6B.check_prefill(batch)
        msg = str(exc.value)
        assert "batch" in msg
        assert str(batch) in msg
        assert "256" in msg  # tile_m(64)*n_aie_rows(4)

    def test_ctx_needs_tile_n_override(self):
        """ctx: K=S, Nout=head_dim=128. At tile_n=64/cols=8 (512), 128 % 512 != 0."""
        with pytest.raises(ValueError) as exc:
            QWEN3_0_6B.check_prefill_seq(256, 1024)
        msg = str(exc.value)
        assert "ctx" in msg
        assert "128" in msg
        assert "16" in msg  # the tile_n the error suggests

    def test_ctx_passes_with_tile_n_16(self):
        QWEN3_0_6B.check_prefill_seq(256, 1024, tile_n_overrides={"ctx": 16})

    def test_scores_needs_no_override(self):
        """scores: K=head_dim=128, Nout=S=1024. Both divide the default config."""
        QWEN3_0_6B.check_prefill_seq(256, 1024, tile_n_overrides={"ctx": 16})

    def test_legal_prefill_batches(self):
        assert QWEN3_0_6B.legal_prefill_batches(600) == [256, 512]


class TestCheckPrefillGemma:
    """gemma3-270m: d_model=640, ffn=2048, head_dim=256. d_model=640=2^7*5 does not divide
    tile_n(64)*cols(8)=512, so the default config genuinely fails for this spec on the `o` and
    `down` ops (Nout=d_model for both) -- this is the real finding: the rails are NOT uniformly
    generic across checkpoints without a per-op tile_n."""

    def test_default_config_fails_on_o(self):
        with pytest.raises(ValueError) as exc:
            GEMMA3_270M.check_prefill(256)
        msg = str(exc.value)
        assert msg.startswith("gemma3-270m: prefill o ")
        assert "640" in msg

    def test_default_config_fails_on_down_once_o_is_fixed(self):
        with pytest.raises(ValueError) as exc:
            GEMMA3_270M.check_prefill(256, tile_n_overrides={"o": 80})
        msg = str(exc.value)
        assert msg.startswith("gemma3-270m: prefill down ")

    def test_passes_with_both_overrides(self):
        GEMMA3_270M.check_prefill(256, tile_n_overrides={"o": 80, "down": 80})

    def test_legal_prefill_batches_empty_without_overrides(self):
        """The API surfaces the same finding through legal_prefill_batches: with the plain
        default tile_n it reports NO legal batch, even though batch itself is fine at 256."""
        assert GEMMA3_270M.legal_prefill_batches(600) == []

    def test_legal_prefill_batches_with_overrides(self):
        assert GEMMA3_270M.legal_prefill_batches(
            600, tile_n_overrides={"o": 80, "down": 80}) == [256, 512]

    def test_ctx_needs_tile_n_32_not_16(self):
        """head_dim=256 here (not Qwen3's 128), so the required ctx tile_n differs: 32, not 16."""
        with pytest.raises(ValueError) as exc:
            GEMMA3_270M.check_prefill_seq(256, 1024)
        assert "32" in str(exc.value)
        GEMMA3_270M.check_prefill_seq(256, 1024, tile_n_overrides={"ctx": 32})


class TestCheckPrefillTileRules:
    def test_tile_m_must_be_multiple_of_16_under_bfp16(self):
        with pytest.raises(ValueError) as exc:
            QWEN3_0_6B.check_prefill(256, tile_m=8)
        assert "tile_m" in str(exc.value)

    def test_tile_m_multiple_of_8_ok_under_plain_bf16(self):
        # tile_m=8 fails op.py's own min at bfp16 (>=8, so 8 passes there) but here we exercise
        # the plain-bf16 kernel rule instead: bfp16=False only needs tile_m % 8 == 0.
        QWEN3_0_6B.check_prefill(256, tile_m=8, bfp16=False)

    def test_l1_overflow_is_reported(self):
        """tile_n=512 still divides qkv's Nout=4096 at cols=8 (4096/512=8), so the Nout%...
        check passes and the L1 budget is what actually fails -- nothing else in the toolchain
        checks this for GEMM."""
        with pytest.raises(ValueError) as exc:
            QWEN3_0_6B.check_prefill(256, tile_m=64, tile_k=64, tile_n=512)
        msg = str(exc.value)
        assert "L1" in msg
        assert "65536" in msg or "64KB" in msg


class TestCheckPrefillProjections:
    """The explicit-op form, which gen_llm_prefill.py uses because it does NOT build the fused
    `qkv` projection check_prefill assumes -- at M>1 a token's v rows sit between its k rows and
    the next token's q rows, so q and k are projected separately."""

    QWEN_OPS = (("q", 1024, 2048), ("k", 1024, 1024), ("v", 1024, 1024), ("o", 2048, 1024),
                ("gate", 1024, 3072), ("up", 1024, 3072), ("down", 3072, 1024))

    def test_split_qkv_is_legal_at_256(self):
        QWEN3_0_6B.check_prefill_projections(256, self.QWEN_OPS)

    def test_batch_modulus_still_applies(self):
        with pytest.raises(ValueError) as exc:
            QWEN3_0_6B.check_prefill_projections(128, self.QWEN_OPS)
        assert "128" in str(exc.value) and "256" in str(exc.value)

    def test_attention_ops_need_the_ctx_override(self):
        ops = (("scores", 128, 2048), ("ctx", 2048, 128))
        with pytest.raises(ValueError) as exc:
            QWEN3_0_6B.check_prefill_projections(256, ops)
        assert "ctx" in str(exc.value)
        QWEN3_0_6B.check_prefill_projections(256, ops, tile_n_overrides={"ctx": 16})

    def test_an_unsatisfiable_nout_names_no_working_tile(self):
        """head_dim=128 at 16 columns: no multiple of 16 divides 128//16 = 8."""
        with pytest.raises(ValueError) as exc:
            QWEN3_0_6B.check_prefill_projections(256, (("ctx", 2048, 128),), cols=16)
        assert "no tile_n multiple of 16 divides" in str(exc.value)
