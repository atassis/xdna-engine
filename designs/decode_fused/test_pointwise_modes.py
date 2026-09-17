# SPDX-License-Identifier: Apache-2.0
"""Device-free gate for POINTWISE_MODES: which ops become Pointwise modes, and GELU's operands.

Run inside the IRON env (an IRON with iron.operators.pointwise):
  PYTHONPATH=designs/decode_fused:$IRON .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_pointwise_modes.py -v
"""
import pytest

gen = pytest.importorskip("gen_llm_decode")
pytest.importorskip("iron.operators.pointwise.op")


@pytest.fixture
def ctx():
    import aie.utils as aie_utils
    from aie.iron.device import from_name
    from iron.common import AIEContext

    aie_utils.set_current_device(from_name("npu2", n_cols=None))
    return AIEContext()


def test_same_width_ops_become_modes(ctx):
    add = gen.ElementwiseAdd(size=3840, tile_size=480, num_aie_columns=8, context=ctx)
    mul = gen.ElementwiseMul(size=3840, tile_size=480, num_aie_columns=8, context=ctx)
    gelu = gen.GELU(size=3840, num_aie_columns=8, num_channels=1, tile_size=480, context=ctx)
    wide = gen.ElementwiseMul(size=15360, tile_size=1920, num_aie_columns=8, context=ctx)
    rl = [(gelu, "g", "g"), (mul, "g", "u", "gh"), (add, "x", "d", "x1"), (mul, "x1", "s", "x2"),
          (wide, "a", "b", "c")]
    out, report = gen.pointwise_modes(rl)

    assert report == [(3840, 480, ["add", "gelu", "mul"])]
    assert [op.mode for op, *_ in out[:4]] == ["gelu", "mul", "add", "mul"]
    assert out[0][1:] == ("g", "g", "g"), "GELU must pass its input as both operands"
    assert out[1][0] is out[3][0], "one Mul operator became two"
    assert out[4][0] is wide, "a width with a single op was rewritten"
