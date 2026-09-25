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
from types import SimpleNamespace

import numpy as np
import pytest

iron_gen = pytest.importorskip("gen_llm_prefill")
BF16 = iron_gen.BF16


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


# ---------------------------------------------------------------------------------------------
# The causal mask. It is a per-row width vector, so the whole contract is (a) which width each row
# gets and (b) that the softmax honours it -- there is no triangle buffer to check.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("base", [0, 256, 512])
def test_widths_are_the_causal_rule_repeated_per_head(base):
    M, S, heads = 8, 2048, 3
    w = iron_gen.causal_widths(base, M, S, heads)
    assert w.dtype == np.int32 and w.shape == (heads * M,)
    for h in range(heads):
        assert list(w[h * M:(h + 1) * M]) == [base + i + 1 for i in range(M)], \
            "row h*M+i must attend base+i+1 positions, the same count under every head"


def test_widths_clamp_at_the_window_and_never_reach_zero():
    """`mask_bf16` loops `for (j = width; j < cols; j++)` over the raw i32: a width past S masks
    nothing and a width of 0 leaves the row all -inf, whose softmax is NaN."""
    M, S = 8, 16
    w = iron_gen.causal_widths(S - 4, M, S, 1)          # runs off the end of the window
    assert list(w) == [13, 14, 15, 16, 16, 16, 16, 16]
    assert w.min() >= 1 and w.max() <= S


def test_the_first_row_of_the_first_chunk_attends_exactly_one_position():
    """The sharpest consequence of causality, and the one a scalar width cannot produce."""
    M, S = 4, 8
    s = np.arange(M * S, dtype=np.float32).reshape(M, S)
    p = np.asarray(iron_gen._softmax_rows(s, iron_gen.causal_widths(0, M, S, 1)), np.float32)
    assert p[0, 0] == 1.0 and np.all(p[0, 1:] == 0.0)
    for i in range(M):
        assert np.count_nonzero(p[i]) == i + 1, f"row {i} must see exactly {i + 1} positions"
        assert abs(p[i].sum() - 1.0) < 1e-2


def test_no_widths_is_the_unmasked_softmax():
    """The `--causal none` arm must stay the exact control it was."""
    s = np.random.default_rng(7).standard_normal((4, 8)).astype(np.float32)
    p = np.asarray(iron_gen._softmax_rows(s, None), np.float32)
    assert np.all(p > 0) and np.allclose(p.sum(-1), 1.0, atol=1e-2)


def test_a_full_width_vector_is_the_unmasked_softmax():
    """A width of S on every row must reduce to the no-mask arm exactly, not approximately --
    both round the same f32 expression to bf16, so any difference is a different computation."""
    M, S = 4, 16
    s = np.random.default_rng(9).standard_normal((M, S)).astype(np.float32)
    full = iron_gen._softmax_rows(s, np.full(M, S, np.int32))
    assert np.array_equal(np.asarray(full), np.asarray(iron_gen._softmax_rows(s, None)))


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


def _scores_entry(n_cols):
    """One head's scores GEMM over the served Gemma-4 sliding geometry, as the runlist writes it.

    kv head 7's slab is the last of eight in an `L0_kc` decode sized to 8 heads x 1024 positions x
    head_dim 256 x bf16 = 4194304 B, so it starts at 3670016 and holds 524288. `n_cols` is the B
    operand's N -- the geometry's own window 1024, or the build-wide S=6912 that reads past it.
    """
    op = SimpleNamespace(get_arg_spec=lambda: [
        SimpleNamespace(shape=(256, 256), dtype=BF16),
        SimpleNamespace(shape=(n_cols, 256), dtype=BF16),
        SimpleNamespace(shape=(256, n_cols), dtype=BF16),
    ])
    fused = SimpleNamespace(subbuffer_layout={
        "q": ("scratch", 0, 256 * 4096 * 2),
        "L0_kc": ("scratch", 0, 4194304),
        "sc": ("scratch", 0, 16 * 256 * n_cols * 2),
    })
    rl = [(op, "q[0:131072]", "L0_kc[3670016:4194304]",
           f"sc[0:{256 * n_cols * 2}]")]
    return rl, fused


def test_operand_bounds_accepts_the_geometrys_own_window():
    iron_gen.check_operand_bounds(*_scores_entry(1024))


def test_operand_bounds_rejects_an_s_wide_read_of_a_narrowed_kv_slab():
    """The defect a LAYOUT_ONLY build reported SUCCESS for on 2026-09-16: every shared buffer's
    own (offset, len) agrees with decode's meta and the descriptor addressing into one does not."""
    with pytest.raises(ValueError) as e:
        iron_gen.check_operand_bounds(*_scores_entry(6912))
    assert "L0_kc" in str(e.value) and "7208960" in str(e.value)


precision = pytest.importorskip("precision")


class TestDqGroupForLength:
    """FORMAT comes from decode's own `weight_quant.plan[site]`, never inferred from the byte
    length -- `dq_group_for_length` used to try every int4 group size and accept any that fit
    `n` "at least" `rows` rows, which is exactly how qwen3.5-4b's `L0_Wg` broke: 14745600 B is
    the EXACT int4/g32 size for [9216, 2560] rows, but also satisfies "int4/g64, >= 9216 rows"
    with 10240 rows (1024 unexplained extra) -- an accidental fit the old ambiguity check
    resolved wrong. `fuse_o` (decode default since 2026-09-08) separately zero-pads Wo past D to
    a whole `TSI_O=3`-row tile (`swiglu_mlp_dp`'s own tiling); D=1024 (qwen3-0.6b) is not a
    multiple of 3, so decode's arena holds 1026 rows, not 1024 -- the ONLY site allowed that
    slack. D=3840 (gemma4-12b) IS a multiple of 3, so it never pads."""

    MLP_G32 = {"mlp": precision.parse_spec("int4/g32/clip", "mlp")}
    MLP_G64 = {"mlp": precision.parse_spec("int4/g64/clip", "mlp")}

    def test_the_qwen35_wg_ambiguity_resolves_to_the_plans_own_group(self):
        """The regression this replaced: reading the format off `n` alone matched BOTH g32 (the
        real one) and g64 (an accidental 1024-row-of-slack fit); reading it off the plan matches
        only the one decode actually built."""
        n, rows, K = 14745600, 9216, 2560
        assert iron_gen.dq_group_for_length(n, rows, K, site="mlp", plan=self.MLP_G32) == 32

    def test_the_same_length_at_the_wrong_plan_site_raises(self):
        """g64's row stride (1440 B) DOES divide 14745600 -- 10240 rows, 1024 more than the 9216
        needed. That accidental fit is exactly the old bug; "mlp" is not "attn_o", so an
        unexplained extra now fails instead of silently passing."""
        n, rows, K = 14745600, 9216, 2560
        with pytest.raises(ValueError, match="9216, 2560"):
            iron_gen.dq_group_for_length(n, rows, K, site="mlp", plan=self.MLP_G64)

    def test_a_fuse_o_padded_bf16_wo_is_still_read_as_bf16(self):
        D, K = 1024, 2048
        assert D % 3 != 0
        padded_rows = -(-D // 3) * 3          # decode's own TSI_O ceiling, reproduced ONLY to
        assert padded_rows == 1026            # build the fixture, not inside the function under test
        n = padded_rows * K * 2
        assert iron_gen.dq_group_for_length(n, D, K, site="attn_o") is None

    def test_an_unpadded_bf16_wo_is_unaffected(self):
        D, K = 1024, 2048
        assert iron_gen.dq_group_for_length(D * K * 2, D, K, site="attn_o") is None

    def test_a_multiple_of_3_d_needs_no_pad_to_pass(self):
        """gemma4-12b's own o-projection shape -- unchanged by this fix: bf16, no plan entry, no
        pad needed or tolerated."""
        D, K = 3840, 8192
        assert D % 3 == 0
        assert iron_gen.dq_group_for_length(D * K * 2, D, K, site="attn_o") is None

    def test_a_short_buffer_still_raises(self):
        """The loud-fail half of K007: fewer bytes than D rows need is a real defect, not padding."""
        D, K = 1024, 2048
        with pytest.raises(ValueError, match="1024, 2048"):
            iron_gen.dq_group_for_length(D * K * 2 - 2, D, K, site="attn_o")

    def test_a_partial_row_still_raises(self):
        """More bytes than D rows need, but not a whole number of rows -- padding is whole rows,
        never a fraction of one."""
        D, K = 1024, 2048
        with pytest.raises(ValueError, match="1024, 2048"):
            iron_gen.dq_group_for_length(D * K * 2 + 1, D, K, site="attn_o")

    def test_an_int4_group_still_resolves_with_padding(self):
        D, K, g = 1024, 2048, 64
        spec = precision.parse_spec(f"int4/g{g}/clip", "attn_o")
        stride = precision.wire_row_units(spec, K)
        assert iron_gen.dq_group_for_length((D + 2) * stride, D, K,
                                            site="attn_o", plan={"attn_o": spec}) == g

    def test_a_pad_outside_attn_o_is_not_tolerated(self):
        """Only Wo's fuse_o pad is a KNOWN pad; an mlp/qkv/head buffer with extra rows is a real
        mismatch, not slack to wave through."""
        D, K = 1024, 2048
        with pytest.raises(ValueError, match="1024, 2048"):
            iron_gen.dq_group_for_length((D + 2) * K * 2, D, K, site="mlp")

    def test_a_pad_past_the_bound_still_raises(self):
        """A bound, not an open door: even at `attn_o`, a pad bigger than any real TSI_O tiling
        produces is a defect (this IS the shape of the qwen3.5 regression, restated at attn_o)."""
        D, K = 1024, 2048
        with pytest.raises(ValueError, match="1024, 2048"):
            iron_gen.dq_group_for_length((D + iron_gen.MAX_FUSE_O_PAD_ROWS + 1) * K * 2, D, K,
                                         site="attn_o")


class TestPlanSpecForSite:
    """gemma4-12b's own `weight_quant.plan` says `"int4/g32/none"` -- `"none"` is not one of
    `precision.SCALE_KINDS["int4"]`, so `precision.parse_spec` alone refuses it (P008). Byte
    length never reads scale_kind, so `plan_spec_for_site` drops it rather than failing the
    whole build over a decode-meta spelling this module's own legality check does not accept."""

    def test_a_scale_kind_the_precision_module_rejects_still_parses(self):
        spec = iron_gen.plan_spec_for_site("int4/g32/none", "attn_o")
        assert (spec.dtype, spec.group_size) == ("int4", 32)

    def test_dtype_and_group_size_are_preserved(self):
        """scale_kind is dropped unconditionally (byte length never reads it, see the class
        docstring) -- dtype and group_size, the two that do, are exactly what was asked for."""
        spec = iron_gen.plan_spec_for_site("int4/g32/clip", "mlp")
        assert (spec.dtype, spec.group_size) == ("int4", 32)

    def test_bf16_is_unaffected(self):
        assert iron_gen.plan_spec_for_site("bf16", "kv").dtype == "bf16"
