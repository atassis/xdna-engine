# SPDX-License-Identifier: Apache-2.0
"""Device-free gate for MERGE_WEIGHT_GEMVS: which GEMVs are retiled, and to what.

A family member left out keeps its own device, and a wrong common tiling does not divide some
member's per-core rows -- both cost only a design on paper, so they are checked here rather than
found as a design count on a 48-layer build.

Run inside the IRON env:
  PYTHONPATH=designs/decode_fused:$IRON .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_unify_weight_gemvs.py -v
"""
import pytest

gen = pytest.importorskip("gen_llm_decode")


@pytest.fixture
def ctx(monkeypatch):
    import aie.utils as aie_utils
    from aie.iron.device import from_name
    from iron.common import AIEContext

    aie_utils.set_current_device(from_name("npu2", n_cols=None))
    monkeypatch.setitem(gen._BUILD_STATE, "layout", "row_group_planar")
    monkeypatch.setitem(gen._BUILD_STATE, "scale_dtype", "bf16")
    return AIEContext()


def _q(M, K, ctx):
    return gen.gemv(M, K, ctx, weight_dtype="int4", group_size=32, layout="row_group_planar",
                    scale_dtype="bf16")


def test_gemma4_weight_family_is_one_tiling(ctx):
    """Gemma-4-12B's served K=3840 weight shapes: per-core rows 480/1024/1088/1920/32768, gcd 32."""
    fam = {M: _q(M, 3840, ctx) for M in (3840, 8192, 8704, 15360, 262144)}
    o_proj = _q(3840, 4096, ctx)
    scores = gen.gemv(1024, 256, ctx)
    rl = [(fam[3840], "Wd", "x", "y"), (o_proj, "Wo", "x", "y"), (fam[8192], "Wq", "x", "y"),
          (fam[3840], "Wd2", "x", "y"), (scores, "kc", "q", "sc"), (fam[8704], "Wq1", "x", "y"),
          (fam[15360], "Wg", "x", "y"), (fam[262144], "W_head", "x", "y")]
    out, report = gen.unify_weight_gemvs(rl)

    assert report == [(3840, 4, 32, [3840, 8192, 8704, 15360, 262144])]
    assert out[1][0] is o_proj and out[4][0] is scores, "a single-M or bf16 GEMV was retiled"
    assert out[0][0] is out[3][0], "one operator became two"
    for op, *_ in (out[i] for i in (0, 2, 5, 6, 7)):
        assert op.tiles_rtp and (op.tile_size_input, op.tile_size_output) == (4, 32)
        assert (op.M // op.num_aie_columns) % op.tile_size_output == 0
    assert [e[1:] for e in out] == [e[1:] for e in rl]


def test_a_single_shape_is_left_alone(ctx):
    op = _q(3840, 3840, ctx)
    rl = [(op, "Wd", "x", "y"), (op, "Wd2", "x", "y")]
    out, report = gen.unify_weight_gemvs(rl)
    assert report == [] and all(e[0] is op for e in out)
