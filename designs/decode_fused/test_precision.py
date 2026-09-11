# SPDX-License-Identifier: Apache-2.0
"""Tests for the decode precision plane -- one per rule it refuses on, plus the byte model.

Run:
  PYTHONPATH=designs/decode_fused .venv-iron/bin/python -m pytest \
      designs/decode_fused/test_precision.py -v

Device-free and IRON-free: `packer_dtypes` is set explicitly in every context so the suite
tests the RULES, not whichever checkout is on the path.
"""
from dataclasses import replace

import pytest

import precision as P

FULL = replace(P.QWEN3_06B, packer_dtypes=P.DTYPES, packer_takes_scale_kind=True)
# Every fused arm off: Wqkv lands on a plain GEMV, which is the one carrier with the axis.
UNFUSED = replace(FULL, fused_layer=False, fuse_o=False, fused_qkv_dp=False)


def plan(**kw):
    out = {k: P.BF16_SPEC for k in P.SITES}
    out.update({k: P.parse_spec(v, k) for k, v in kw.items()})
    return out


def refusal(rule, fn, *a, **kw):
    with pytest.raises(P.PrecisionRefusal) as exc:
        fn(*a, **kw)
    assert exc.value.rule.id == rule, f"expected {rule}, got {exc.value.rule.id}: {exc.value}"
    return exc.value


class TestParsing:
    def test_bare_dtype_takes_the_group_default(self):
        assert P.parse_spec("int8", "mlp") == P.Spec("int8", 128, "absmax")

    def test_scale_kind_defaults_per_class(self):
        assert P.parse_spec("int4a", "mlp").scale_kind == "zero_grid"
        assert P.parse_spec("int4a", "qkv").scale_kind == "free_min"

    def test_unknown_dtype_is_P001(self):
        refusal("P001", P.parse_spec, "fp8", "mlp")

    def test_scale_kind_must_match_the_family(self):
        refusal("P008", P.parse_spec, "int8/g128/zero_grid", "mlp")
        refusal("P008", P.parse_spec, "int8a/g128/clip", "mlp")

    def test_unknown_site_is_refused(self):
        refusal("P008", P.parse_plan, '{"wqkv": "int8"}')

    def test_plan_and_legacy_env_together_are_refused(self):
        refusal("P008", P.plan_from_env,
                {"PRECISION": "bf16", "QUANT_MLP_DTYPE": "int4"})

    def test_legacy_env_still_resolves_to_a_plan(self):
        got, prov = P.plan_from_env({"QUANT_MLP_DTYPE": "int4", "QUANT_MLP_GROUP": "64",
                                     "QUANT_CLIP_SEARCH": "1"})
        assert got["mlp"] == P.Spec("int4", 64, "clip")
        assert got["head"] == P.BF16_SPEC
        assert "legacy" in prov


class TestCapability:
    def test_a_packer_without_the_dtype_is_P001(self):
        ctx = replace(FULL, packer_dtypes=(P.BF16, "int4", "int8"))
        refusal("P001", P.check, plan(mlp="int8a"), ctx)

    def test_a_packer_without_scale_selection_refuses_clip(self):
        ctx = replace(FULL, packer_takes_scale_kind=False)
        refusal("P001", P.check, plan(mlp="int4/g128/clip"), ctx)
        P.check(plan(mlp="int4/g128/absmax", attn_o="int4/g128/absmax"), ctx)


class TestOneFifoOneDtype:
    def test_qkv_cannot_differ_from_the_cache_in_the_fused_layer(self):
        exc = refusal("P002", P.check, plan(qkv="int8a", kv="bf16"), FULL)
        assert "dedicated input" in str(exc)

    def test_the_message_carries_the_escape_route_price(self):
        exc = refusal("P002", P.check, plan(qkv="int8a"), FULL)
        # 8 attention columns against a margin of 2 in / 4 out.
        assert "over by 6 and 4" in str(exc)

    def test_wo_cannot_differ_from_the_mlp_under_fuse_o(self):
        refusal("P002", P.check, plan(mlp="int8a", attn_o="bf16"), FULL)

    def test_wo_is_free_when_it_has_its_own_channel(self):
        P.check(plan(mlp="int8a", attn_o="bf16"), UNFUSED)


class TestDeclaringOperator:
    def test_the_fused_attention_half_has_no_axis(self):
        exc = refusal("P003", P.check, plan(qkv="int8a", kv="int8a"), FULL)
        assert "attn_block_dp" in str(exc)

    def test_qkv_is_reachable_only_on_a_plain_gemv(self):
        P.check(plan(qkv="int8a"), UNFUSED)

    def test_qkv_head_dp_has_no_axis_either(self):
        """The carrier that is neither the fused layer nor a plain GEMV. Missing it built an
        artifact whose Wqkv buffer was packed to 4325376 B against a declared 8388608."""
        exc = refusal("P003", P.check, plan(qkv="int8a"),
                      replace(UNFUSED, fused_qkv_dp=True))
        assert "qkv_head_dp" in str(exc)

    def test_split_qkv_gemvs_cannot_take_the_axis(self):
        refusal("P003", P.check, plan(qkv="int8a"), replace(UNFUSED, fused_qkv_gemv=False))


class TestChannelBudget:
    def test_dedicated_kv_channels_do_not_fit(self):
        exc = refusal("P004", P.check, plan(kv="int8a", qkv="int8a"),
                      replace(FULL, kv_dedicated_channels=True))
        assert "over by 6 / 4" in str(exc)

    def test_the_cost_is_derived_from_the_column_count(self):
        c = P.dedicated_channel_cost(replace(FULL, attn_cols=2))
        assert c == {"in_need": 2, "out_need": 2, "in_margin": 2, "out_margin": 4,
                     "in_over": 0, "out_over": 0}


class TestCacheRules:
    def test_four_bit_cache_is_refused_on_the_scale_axis(self):
        refusal("P006", P.check, plan(kv="int4a", qkv="int4a"),
                replace(UNFUSED, fused_qkv_gemv=True))

    def test_address_granule_follows_the_cache_dtype(self):
        assert P.kv_addr_gran_elems(plan()) == 2
        assert P.kv_addr_gran_elems(plan(kv="int8a")) == 4

    def test_the_granule_default_is_the_bf16_answer(self):
        """kv_layout.derive_block_size defaults addr_gran_elems to 2, which is only right for a
        bf16 cache. The plane is what stops that default from being reached silently."""
        from iron.common.kv_layout import derive_block_size
        import inspect
        assert inspect.signature(derive_block_size).parameters["addr_gran_elems"].default == 2
        assert derive_block_size(HD=128, Hkv=8, addr_gran_elems=2) != \
            derive_block_size(HD=128, Hkv=8, addr_gran_elems=4)


class TestWireArithmetic:
    def test_group_must_divide_K(self):
        refusal("P007", P.wire_row_units, P.parse_spec("int4/g96"), 1024)

    def test_an_unaligned_row_stride_is_refused(self):
        refusal("P007", P.wire_row_units, P.parse_spec("int4/g2"), 6)

    def test_int4_and_affine_int4_are_the_same_bytes_at_the_same_group(self):
        assert P.wire_bytes_per_element(P.parse_spec("int4/g128"), 1024) == \
            P.wire_bytes_per_element(P.parse_spec("int4a/g128"), 1024)

    def test_int4_g128_is_the_measured_3_765x(self):
        """int4's header needs no pad at this shape -- 32 B of scales against a 32 B load -- so
        the recorded ratio is untouched by the alignment fix. int8's is not."""
        got = 2.0 / P.wire_bytes_per_element(P.parse_spec("int4/g128"), 1024)
        assert abs(got - 3.765) < 0.001

    def test_the_plane_agrees_with_the_packer(self):
        """The plane prices an arm the packer ships. A divergence is this test, not a wrong
        census."""
        quant = pytest.importorskip("iron.common.quant")
        for dtype in ("int4", "int8"):
            for k, g in ((1024, 128), (3072, 128), (2048, 64), (1024, 32)):
                assert P.wire_row_units(P.parse_spec(f"{dtype}/g{g}"), k) == \
                    quant.row_stride_bytes(k, g, dtype), (dtype, k, g)


class TestPackerContract:
    """The plan's scale_kind has to reach the packer under the name the packer uses. A wrong
    keyword is a TypeError three frames into the build, which is the failure P001 exists for."""

    def test_every_scale_kind_reaches_the_packer(self):
        quant = pytest.importorskip("iron.common.quant")
        import inspect
        import numpy as np
        params = inspect.signature(quant.quantize_weight).parameters
        for kind, kw in (("clip", "clip_search"), ("zero_grid", "affine_zero_on_grid"),
                         ("free_min", "affine_zero_on_grid")):
            assert kw in params, f"scale_kind {kind!r} has no packer argument {kw!r}"
        W = np.random.default_rng(0).standard_normal((8, 1024), dtype=np.float32)
        for dtype in P.SYMMETRIC + P.AFFINE:
            if dtype not in P.packer_capability()[0]:
                pytest.skip(f"packer on this path has no {dtype}")
            spec = P.parse_spec(f"{dtype}/g128")
            packed = quant.quantize_weight(
                W, 128, dtype,
                **({"clip_search": False} if dtype in P.SYMMETRIC
                   else {"affine_zero_on_grid": spec.scale_kind == "zero_grid"}))
            assert packed.nbytes == 8 * P.wire_row_units(spec, 1024)


class TestByteModel:
    def test_bf16_reproduces_the_census(self):
        assert abs(P.token_mb(plan())["total"] - P.CENSUS_TOKEN_MB) < 0.01

    def test_the_sites_and_the_remainder_close_on_the_census(self):
        assert abs(sum(s.mb_per_token for s in P.SITES.values())
                   + P.CENSUS_UNSITED_MB - P.CENSUS_TOKEN_MB) < 0.01

    def test_bf16_rows_are_counted_in_elements_not_bytes(self):
        """The unit is what the array is indexed in. Returning bytes here reads correct and
        halves the row count of every bf16 reshape -- it corrupted Wqkv's head-major reorder."""
        assert P.wire_row_units(P.BF16_SPEC, 1024) == 1024

    def test_the_rate_is_K_independent_for_the_shipped_layout(self):
        """header+payload with n_groups scaling in K, so the per-element rate does not move with
        K. It stopped being true the afternoon the header was padded to the load width, which is
        why the byte model asks the packer rather than assuming."""
        for dt in ("int4/g128", "int8/g128"):
            spec = P.parse_spec(dt)
            assert P.wire_bytes_per_element(spec, 1024) == P.wire_bytes_per_element(spec, 3072)
        assert P.wire_bytes_per_element(P.parse_spec("int4/g32"), 1024) > \
            P.wire_bytes_per_element(P.parse_spec("int4/g128"), 1024)

    def test_int8_weights_land_on_the_recorded_615_mb(self):
        """int8 at every weight site takes the token's weight bytes 1193 -> 615 MB, measured,
        for exactly this plan. Alignment is bought with the LOAD WIDTH, not with header padding,
        so the bytes are unchanged -- a pad would have cost 17 MB here."""
        p = plan(mlp="int8a", attn_o="int8a", qkv="int8a", head="int8a")
        mb = P.token_mb(p)
        weights = mb["mlp"] + mb["attn_o"] + mb["qkv"] + mb["head"]
        assert 610 < weights < 620, weights

    def test_the_transport_prediction_is_the_byte_delta_at_the_fitted_rate(self):
        p = plan(mlp="int8a")
        delta_mb = P.token_mb(p)["total"] - P.CENSUS_TOKEN_MB
        assert abs(P.predicted_ms_delta(p) - delta_mb * P.MARGINAL_US_PER_MB / 1e3) < 1e-9


class TestCensusPerSpec:
    """The defect this fixes: gemma4-12b built and printed qwen3-0.6b's 1664.09 MB/token because
    the census had no spec axis. CENSUS gives every function here a spec_name; qwen3-0.6b's
    default must stay byte-identical to what it printed before CENSUS existed."""

    def test_default_spec_name_is_qwen3(self):
        assert P.DEFAULT_CENSUS_SPEC == "qwen3-0.6b"

    def test_qwen3_token_mb_is_unchanged_by_the_spec_axis(self):
        assert P.token_mb(plan()) == P.token_mb(plan(), "qwen3-0.6b")
        assert abs(P.token_mb(plan())["total"] - P.CENSUS_TOKEN_MB) < 0.01

    def test_qwen3_describe_is_byte_identical_to_the_pre_census_output(self):
        """Reconstructs exactly what describe() returned before CENSUS existed, straight from
        the module's own SITES/CENSUS_TOKEN_MB constants -- not a frozen string copy, so this
        cannot drift out of sync with a legitimate qwen3 re-measurement."""
        p = plan(mlp="int8a")
        mb = P.token_mb(p)
        lines = [f"precision plan (census {P.CENSUS_DATE}, window {P.CENSUS_WINDOW}, "
                 f"{P.CENSUS_TOKEN_MB:.2f} MB/token bf16)"]
        for key, site in sorted(P.SITES.items(), key=lambda kv: -kv[1].mb_per_token):
            spec = p.get(key, P.BF16_SPEC)
            lines.append(f"  {key:7} {str(spec):22} {site.mb_per_token:8.2f} -> "
                         f"{mb[key]:8.2f} MB  ({100 * mb[key] / site.mb_per_token:5.1f}%)")
        lines.append(f"  {'TOTAL':7} {'':22} {P.CENSUS_TOKEN_MB:8.2f} -> {mb['total']:8.2f} MB")
        lines.append(f"  transport prediction: {P.predicted_ms_delta(p):+.2f} ms/token "
                     "(byte term only; the dequant's core cost is not in this number)")
        assert P.describe(p) == "\n".join(lines)

    def test_gemma4_reproduces_its_own_census_at_the_measured_plan(self):
        """The shipped gemma4-12b build quantized mlp/qkv/attn_o int8/g64
        (scenarios/generate-gemma4-12b.toml); querying that exact plan must reproduce the
        measured total, not re-shrink an already-quantized figure a second time."""
        p = plan(mlp="int8a/g64", qkv="int8a/g64", attn_o="int8a/g64")
        assert abs(P.token_mb(p, "gemma4-12b")["total"] - 17108.56) < 0.01

    def test_gemma4_bf16_plan_prices_above_the_measured_int8_total(self):
        """17108.56 is what the build measured with mlp/qkv/attn_o ALREADY at int8/g64 -- a
        bf16-everywhere plan undoes that quantization and must cost strictly more, not
        reproduce the same figure (the double-discount Census.baseline exists to prevent)."""
        assert P.token_mb(plan(), "gemma4-12b")["total"] > 17108.56

    def test_gemma4_census_sites_match_the_measured_ground_truth(self):
        c = P.CENSUS["gemma4-12b"]
        assert c.site_mb["mlp"] == 9046.42
        assert c.site_mb["kv"] == 1280.74
        assert c.site_mb["head"] == 2013.27
        assert c.site_mb["qkv"] == 2859.94
        assert c.site_mb["attn_o"] == 0.0

    def test_every_census_entry_carries_its_provenance(self):
        """A measured constant with no traceable source is the exact failure class the census
        constants had before this fix -- CENSUS must not reintroduce it under a new name."""
        for name, c in P.CENSUS.items():
            assert c.method, name
            assert c.date, name

    def test_gemma4_describe_does_not_crash_on_the_unsplit_attn_o_site(self):
        """attn_o's baseline is 0 (qkv carries the combined Wqkv+Wo figure) -- describe() must
        not divide by it."""
        out = P.describe(plan(), "gemma4-12b")
        assert "attn_o" in out
        assert "n/a" in out

    def test_unknown_spec_is_refused_not_silently_priced_as_qwen3(self):
        with pytest.raises(ValueError, match="no precision census"):
            P.token_mb(plan(), "gemma3-270m")
        with pytest.raises(ValueError, match="no precision census"):
            P.predicted_ms_delta(plan(), "gemma3-270m")

    def test_unknown_spec_describe_prints_unpriced_rather_than_a_borrowed_number(self):
        out = P.describe(plan(), "gemma3-270m")
        assert "no census" in out
        assert "1664.09" not in out and "17108.56" not in out
        # gen_llm_decode.py prints describe(...).splitlines()[1:] -- a one-line message here
        # would be silently swallowed there, which is the exact failure this refuses.
        assert len(out.splitlines()) >= 2


class TestNaming:
    def test_two_plans_that_differ_do_not_share_an_artifact_name(self):
        names = {P.suffix(parse) for parse in (
            plan(), plan(mlp="int8a"), plan(mlp="int4a"), plan(mlp="int4a/g64"),
            plan(mlp="int4a/g64/free_min"), plan(head="int8a"))}
        assert len(names) == 6

    def test_bf16_contributes_no_suffix(self):
        assert P.suffix(plan()) == ""


class TestPresets:
    @pytest.mark.parametrize("name", sorted(P.PRESETS))
    def test_every_preset_builds_on_the_arm_it_claims(self, name):
        raw, reach = P.PRESETS[name]
        import json
        P.check(P.parse_plan(json.dumps(raw)), FULL if reach == "fused" else UNFUSED)

    def test_the_unfused_preset_is_refused_on_the_fused_arm(self):
        """`all-int8` is the measured quality optimum and the fused layer cannot carry it. The
        table says so; this is the check that the table is not lying."""
        import json
        refusal("P002", P.check, P.parse_plan(json.dumps(P.PRESETS["all-int8"][0])), FULL)
