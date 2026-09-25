# SPDX-License-Identifier: Apache-2.0
"""Device-free gate keeping `gen_llm_decode.wo_rows_padded` equal to the operator it mirrors.

`SwiGLUMLPDataParallel._wo_rows_padded` (iron/operators/swiglu_mlp_dp/op.py) is the authority;
`wo_rows_padded` is a pure re-derivation so `gen_llm_prefill.py`'s `dq_group` -- which has no
operator instance to ask -- can compute the same number. `test_it_mirrors_the_operator` is what
keeps that honest (the same trade `precision.py`'s `wire_row_units` fallback makes against the
packer it mirrors): a real instance is built and compared, for a padded case (qwen3-0.6b's own
D=1024, QD=2048, N=4 -> 1026 rows) and an unpadded one (D=3840, a multiple of TSI_O -> no pad).

Run inside the IRON env:
  PYTHONPATH=designs/decode_fused:$IRON .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_wo_rows_padded.py -v
"""
import pytest

gld = pytest.importorskip("gen_llm_decode")


def _built(D, QD, N):
    import newstack_compat  # noqa: F401 -- MUST precede iron imports
    from iron.common import AIEContext
    from iron.operators.swiglu_mlp_dp.op import SwiGLUMLPDataParallel
    op = SwiGLUMLPDataParallel(D=D, FF=D * 3, num_aie_columns=N, epsilon=1e-6, QD=QD,
                               fuse_o=True, act="silu", post_norm=False, context=AIEContext(),
                               weight_depth=1, tile_rows_gu=1)
    return op._wo_rows_padded


@pytest.mark.parametrize("D,QD,N", [
    (1024, 2048, 4),   # qwen3-0.6b's own shape: D_PER_CORE=256, TSI_O=3, pads to 258 -> +2 rows
    (3840, 7680, 4),   # D_PER_CORE=960, TSI_O=3, already a multiple -> no pad
    (3840, 3840, 4),   # D_PER_CORE=960, TSI_O=6, already a multiple -> no pad
])
def test_it_mirrors_the_operator(D, QD, N):
    assert gld.wo_rows_padded(D, QD, N) == _built(D, QD, N)


def test_the_qwen3_0_6b_pad_is_two_rows():
    assert gld.wo_rows_padded(1024, 2048, 4) - 1024 == 2
