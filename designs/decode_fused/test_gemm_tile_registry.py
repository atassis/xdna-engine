# SPDX-License-Identifier: Apache-2.0
"""Device-free gates for the measured GEMM tile registry and the rules the sweep filters on.

The one that matters is `test_seeds_are_the_pre_registry_constants`. Replacing a hardcoded triple
with a lookup is only safe if the lookup starts out returning the SAME triple; otherwise the change
that was supposed to be inert silently rebuilds every prefill ELF with a different tiling. Once the
sweep lands and moves an entry, that test's `source` assert flips to "sweep" and the equality is
expected to fail -- update it then, deliberately.

Run without IRON:
  PYTHONPATH=designs/decode_fused python -m pytest designs/decode_fused/test_gemm_tile_registry.py
"""
import json

import pytest

from gemm_tile_registry import (
    IllegalRegistryEntry, Registry, UnsweptGemmShape, key_of, DEFAULT_REGISTRY,
)
from llm_decode_spec import (
    QWEN3_0_6B, gemm_l1_bytes, gemm_memtile_bytes, gemm_tile_census, gemm_tile_grid,
    gemm_tiling_rejection, L1_BYTES, MEMTILE_BYTES,
)

# The seven GEMMs gen_llm_prefill.py builds for qwen3-0.6b at M=256, S=2048.
SHAPES = {
    "q": (256, 1024, 2048, True),
    "kv": (256, 1024, 1024, True),
    "o": (256, 2048, 1024, True),
    "gate_up": (256, 1024, 3072, True),
    "down": (256, 3072, 1024, True),
    "scores": (256, 128, 2048, True),
    "ctx": (256, 2048, 128, False),
}


@pytest.fixture
def reg():
    return Registry.load(DEFAULT_REGISTRY, overrides={})


class TestRegistry:
    def test_seeds_are_the_pre_registry_constants(self, reg):
        """64/64/64 at 8 columns for every GEMM, with ctx's tile_n at 16 -- exactly what the
        module constants were. This is the whole inertness claim, asserted rather than described."""
        for label, (M, K, N, bcm) in SHAPES.items():
            ch = reg.lookup(M, K, N, b_col_maj=bcm, label=label)
            want_n = 16 if label == "ctx" else 64
            assert (ch.tile_m, ch.tile_k, ch.tile_n, ch.cols) == (64, 64, want_n, 8), label
            assert ch.source == "seed", f"{label} has moved off the seed; update this test"
            assert ch.measured is None

    def test_every_numerics_arm_is_seeded(self, reg):
        """PREFILL_BFP16 / PREFILL_ACC change the KEY, so an unseeded arm would start raising the
        moment someone runs the numerics A/B."""
        for M, K, N, _ in SHAPES.values():
            for emulate in (True, False):
                for prio in (False, True):
                    reg.lookup(M, K, N, emulate=emulate, prio_accuracy=prio)

    def test_an_unswept_shape_raises_and_names_the_sweep(self, reg):
        with pytest.raises(UnsweptGemmShape) as exc:
            reg.lookup(512, 1024, 2048, label="q")
        msg = str(exc.value)
        assert "sweep_gemm_tiles.sh --shape 512x1024x2048" in msg
        assert "64" not in msg.split("--tile")[0], "a miss must not name a default tile"

    def test_a_near_miss_does_not_fall_back_to_a_neighbour(self, reg):
        """M=256 K=1024 N=2048 is seeded; N=1536 is a different shape and must NOT inherit it --
        even though the seeded tiling would be perfectly legal for it."""
        reg.lookup(256, 1024, 2048)
        assert gemm_tiling_rejection(256, 1024, 1536, 64, 64, 64, 8) is None
        with pytest.raises(UnsweptGemmShape):
            reg.lookup(256, 1024, 1536)

    def test_orientation_mismatch_raises(self, reg):
        """ctx is recorded b_col_maj=False; B's dims_to_stream differs, so the measurement does
        not carry across the two orientations."""
        with pytest.raises(IllegalRegistryEntry):
            reg.lookup(256, 2048, 128, b_col_maj=True, label="ctx")

    def test_an_illegal_hand_edit_is_caught_on_lookup(self, tmp_path):
        doc = json.loads(DEFAULT_REGISTRY.read_text())
        doc["entries"][key_of(256, 1024, 2048)]["tile_n"] = 48   # 2048 % (48*8) != 0
        p = tmp_path / "gemm_tiles.json"
        p.write_text(json.dumps(doc))
        with pytest.raises(IllegalRegistryEntry) as exc:
            Registry.load(p, overrides={}).lookup(256, 1024, 2048, label="q")
        assert "48" in str(exc.value)

    def test_record_refuses_an_illegal_tiling(self):
        r = Registry({})
        with pytest.raises(IllegalRegistryEntry):
            r.record(256, 1024, 2048, 64, 64, 48, 8)

    def test_override_by_label(self):
        r = Registry.load(DEFAULT_REGISTRY, overrides={"scores": {"tile_n": 32}})
        ch = r.lookup(256, 128, 2048, label="scores")
        assert (ch.tile_n, ch.source) == (32, "override")
        assert r.lookup(256, 1024, 2048, label="q").tile_n == 64, "override must not leak"

    def test_override_on_an_unswept_shape_needs_the_whole_tiling(self):
        r = Registry.load(DEFAULT_REGISTRY, overrides={"q": {"tile_n": 32}})
        with pytest.raises(ValueError, match="whole tiling"):
            r.lookup(512, 1024, 2048, label="q")

    def test_override_rejects_an_unknown_field(self):
        r = Registry.load(DEFAULT_REGISTRY, overrides={"q": {"tile_z": 32}})
        with pytest.raises(ValueError, match="unknown field"):
            r.lookup(256, 1024, 2048, label="q")


class TestTilingRules:
    def test_prio_accuracy_costs_two_more_bytes_per_c_element(self):
        """design.py drops the C fifo to depth 1 under prio_accuracy but adds an f32 acc_buffer:
        6 bytes/element, not 4. The formula this file replaced had only the 4."""
        plain = gemm_l1_bytes(64, 64, 64)
        acc = gemm_l1_bytes(64, 64, 64, prio_accuracy=True)
        assert acc - plain == 2 * 64 * 64

    def test_the_seeded_tiles_fit_l1_under_both_accumulators(self):
        for label, (_, _, _, _) in SHAPES.items():
            tn = 16 if label == "ctx" else 64
            for prio in (False, True):
                assert gemm_l1_bytes(64, 64, tn, prio_accuracy=prio) <= L1_BYTES, (label, prio)

    def test_the_memtile_budget_never_binds_before_l1(self):
        """Over the whole derived grid: L1 = 4mk+4kn+4mn+0xD00 <= 64K implies MemTile <= 512K for
        every legal column count, so the MemTile filter cannot reject a candidate L1 admits. It is
        kept because it is a real budget that a different dataflow could hit -- but it is inert
        here, and a census that ever reports a `memtile` rejection means this changed."""
        offenders = [c for c in gemm_tile_grid()
                     if gemm_memtile_bytes(*c) > MEMTILE_BYTES
                     and gemm_l1_bytes(*c[:3]) <= L1_BYTES]
        assert offenders == []

    def test_ctx_is_the_shape_that_forces_a_narrow_tile_n(self):
        """Nout=128 at 8 columns admits exactly one tile_n. This is why the generators carried a
        hand-written override for this op and only this op."""
        legal, _ = gemm_tile_census(256, 2048, 128)
        assert sorted({c[2] for c in legal if c[3] == 8}) == [16]

    def test_the_census_partitions_the_grid(self):
        legal, rejected = gemm_tile_census(256, 1024, 2048)
        assert len(legal) + sum(len(v) for v in rejected.values()) == len(gemm_tile_grid())
        assert set(rejected) <= {"cols", "tile_m", "tile_k", "tile_n", "batch", "K", "N",
                                 "l1", "memtile"}

    def test_the_bfp16_path_forbids_a_tile_m_the_plain_path_allows(self):
        """(r,s,t) is (8,8,8) with the emulation on and (4,8,8) without, so tile_m=8 is legal on
        one path and not the other -- which is why `emulate` is part of the registry key."""
        assert gemm_tiling_rejection(256, 1024, 2048, 8, 64, 64, 8, bfp16=True).code == "tile_m"
        assert gemm_tiling_rejection(256, 1024, 2048, 8, 64, 64, 8, bfp16=False) is None

    def test_the_rules_agree_with_the_spec_checks(self):
        """`check_prefill_projections` and `gemm_tiling_rejection` must never disagree: the sweep
        would build what the generator then refuses, or worse the other way round."""
        for M, K, N, _ in SHAPES.values():
            for cand in gemm_tile_grid(tile_ms=[16, 64], tile_ks=[8, 64], tile_ns=[16, 64],
                                       colss=[4, 8]):
                tm, tk, tn, c = cand
                rej = gemm_tiling_rejection(M, K, N, tm, tk, tn, c)
                try:
                    QWEN3_0_6B.check_prefill_projections(M, (("op", K, N),), tile_m=tm, tile_k=tk,
                                                         tile_n=tn, cols=c)
                    raised = None
                except ValueError as e:
                    raised = str(e)
                assert (rej is None) == (raised is None), (M, K, N, cand, rej, raised)
