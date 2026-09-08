# SPDX-License-Identifier: Apache-2.0
"""Device-free gates for the batched-prefill generator.

The one that matters is the RoPE convention. `rope/design.py::core_body` acquires ONE angle row
and applies it to `rows/angle_rows` CONSECUTIVE tensor rows (block); `rope/reference.py` tiles the
angle table with `torch.Tensor.repeat` (interleaved). They agree only at `angle_rows == 1` or
`angle_rows == rows`, which is why nothing on this rail has ever noticed -- decode only ever runs
angle_rows=1. A batched-prefill golden written against the reference is wrong, silently, for every
M in between, so both conventions are written out here and the disagreement is asserted rather
than described.

Run inside the IRON env:
  PYTHONPATH=designs/decode_fused:$IRON .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_llm_prefill.py -v
"""
import numpy as np
import pytest

iron_gen = pytest.importorskip("gen_llm_prefill")


def _apply(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _block(x, ang):
    """design.py: row r takes angle row r // (rows/angle_rows)."""
    rows, ar = x.shape[0], ang.shape[0]
    per = rows // ar
    return np.stack([_apply(x[r], ang[r // per, 0::2], ang[r // per, 1::2]) for r in range(rows)])


def _tile(x, ang):
    """reference.py: cos.repeat(rep, 1) tiles, so row r takes angle row r % angle_rows."""
    rows, ar = x.shape[0], ang.shape[0]
    return np.stack([_apply(x[r], ang[r % ar, 0::2], ang[r % ar, 1::2]) for r in range(rows)])


def _fixture(rows, angle_rows, cols=64, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, cols)).astype(np.float32),
            rng.standard_normal((angle_rows, cols)).astype(np.float32))


@pytest.mark.parametrize("rows,angle_rows", [(8, 1), (8, 8)])
def test_the_two_conventions_coincide_only_at_the_endpoints(rows, angle_rows):
    x, ang = _fixture(rows, angle_rows)
    assert np.allclose(_block(x, ang), _tile(x, ang))


@pytest.mark.parametrize("rows,angle_rows", [(8, 2), (8, 4), (12, 3), (32, 8)])
def test_the_two_conventions_disagree_in_between(rows, angle_rows):
    x, ang = _fixture(rows, angle_rows)
    assert not np.allclose(_block(x, ang), _tile(x, ang))


@pytest.mark.parametrize("M,heads", [(4, 2), (8, 4), (16, 8)])
def test_generator_golden_uses_the_block_convention(M, heads):
    """The prefill golden's vectorised `_rope_block` must equal the row-by-row block rule, over a
    TOKEN-major [M, heads, HD] tensor -- where the block quotient is exactly the head count, which
    is what makes design.py's semantics correct for a batched QKV projection's natural output."""
    HD = 64
    rng = np.random.default_rng(5)
    x = rng.standard_normal((M, heads, HD)).astype(np.float32)
    ang = rng.standard_normal((M, HD)).astype(np.float32)
    got = iron_gen._rope_block(x, ang, heads)
    want = iron_gen.bf16(_block(x.reshape(M * heads, HD), ang).reshape(M, heads, HD))
    # Exact, not approximate: both round the same f32 expression to bf16, so a difference is a
    # different convention and never a rounding artifact.
    assert np.array_equal(np.asarray(got), np.asarray(want)), \
        "golden RoPE is not design.py's block convention"


def test_rope_table_is_interleaved_cos_sin():
    """rope/reference.py reads cos at [0::2] and sin at [1::2]; the table must match, and row t
    must carry ABSOLUTE position base+t, not t."""
    tab = np.asarray(iron_gen.rope_table(5, 3, 64, 1e6), np.float32)
    assert tab.shape == (3, 64)
    inv = 1.0 / (1e6 ** (np.arange(0, 64, 2, dtype=np.float64)[:32] / 64))
    for t in range(3):
        assert np.allclose(tab[t, 0::2], np.cos((5 + t) * inv), atol=8e-3)
        assert np.allclose(tab[t, 1::2], np.sin((5 + t) * inv), atol=8e-3)


def test_transfer_size_divides_and_fits():
    for total in (256 * 2048, 256 * 8 * 128):
        t = iron_gen.pick_transfer(total)
        assert total % t == 0 and t <= iron_gen.XFER_ELEMS


def test_decode_arena_plan_fills_gaps(tmp_path):
    """A layout that names only the weights must come back as a packed order whose fillers cover
    the unnamed intermediates -- otherwise the shared weights slide."""
    meta = tmp_path / "meta.json"
    meta.write_text('{"layout": {"a": {"type": "scratch", "offset": 0, "len": 16},'
                    ' "b": {"type": "scratch", "offset": 48, "len": 8},'
                    ' "x": {"type": "input", "offset": 0, "len": 4}}}')
    _, order, sizes, reserved = iron_gen.decode_arena_plan(str(meta))
    assert order == ["a", "__decode_gap0", "b"]
    assert sizes["__decode_gap0"] == 32 and reserved == 56
