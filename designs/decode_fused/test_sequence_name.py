# SPDX-License-Identifier: Apache-2.0
"""Device-free gate: the design name must describe the graph that was built.

IRON keys its artifact cache on this name, and every measurement is attributed by it, so a name
that claims an arm the build declined mis-attributes whatever that ELF measures. It has happened:
Gemma-4 shipped as `mlpdp4_mlpo` with zero `swiglu` in its ELF, and half a -12.96% was credited to
a fusion that was never built.

Run inside the IRON env:
  PYTHONPATH=designs/decode_fused:$IRON .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_sequence_name.py -v
"""
import pytest

gen = pytest.importorskip("gen_llm_decode")

from llm_decode_spec import QWEN3_0_6B, S2_PRO_SLOW_AR  # noqa: E402


class TestMlpArmSuffixes:
    """`sp.mlp_dp_reason()` returns None for every spec -- the refusals live in gen_llm_decode's
    `mlp_dp_why` (operator_rejects, FF % D, MLP_TILE_ROWS) and in `fuse_o`. The name must read
    those, not re-derive a weaker predicate."""

    def test_declined_arms_are_absent_from_the_name(self):
        n = gen.sequence_name(S2_PRO_SLOW_AR, 1, 2048, "", mlp_dp_active=False, mlp_o_active=False)
        assert "mlpdp" not in n, n
        assert "mlpo" not in n, n

    def test_built_arms_are_present(self):
        n = gen.sequence_name(QWEN3_0_6B, 1, 2048, "", mlp_dp_active=True, mlp_o_active=True)
        assert f"mlpdp{gen.MLP_DP_COLS}" in n, n
        assert "mlpo" in n, n

    def test_fuse_o_alone_declining_drops_only_mlpo(self):
        """FUSE_MLP_O's own refusal (sandwich norms, QD % D) is independent of swiglu_mlp_dp's."""
        n = gen.sequence_name(QWEN3_0_6B, 1, 2048, "", mlp_dp_active=True, mlp_o_active=False)
        assert f"mlpdp{gen.MLP_DP_COLS}" in n, n
        assert "mlpo" not in n, n
