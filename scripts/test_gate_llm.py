# SPDX-License-Identifier: Apache-2.0
"""Device-free gates for the two-tier correctness gate itself.

Everything here runs on numpy alone -- no NPU, no IRON, no toolchain instance. The three properties
worth pinning are the ones a gate can get wrong while still printing PASS:

  1. it FAILS on a defect, including a single wrong element in a quarter of a million,
  2. it FAILS LOUD when the reference is missing, rather than passing an unchecked artifact,
  3. the token-set rule judges the FIRST divergence and stops there.

  .venv-iron/bin/python -m pytest scripts/test_gate_llm.py -v
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import gate_numeric as gn  # noqa: E402
from gate_token_set import judge  # noqa: E402

ml_dtypes = pytest.importorskip("ml_dtypes")
BF16 = ml_dtypes.bfloat16


# ------------------------------------------------------------------------------------------------
# TIER 1: the check itself.
# ------------------------------------------------------------------------------------------------
def test_rtol_is_the_bf16_tolerance_and_is_not_per_artifact():
    """A build must not be able to lower its own bar. rtol is a property of the dtype."""
    assert gn.RTOL == 1.6e-2


def test_identical_tensors_pass_at_zero_atol():
    ref = np.linspace(-3, 3, 1000).astype(np.float32)
    r = gn.check(ref, ref, atol=0.0)
    assert r["pass"] and r["n_bad"] == 0 and r["mean_rel_L1"] == 0.0


def test_one_wrong_element_in_a_quarter_million_fails():
    """The reason the check is element-wise over the FULL output rather than a summary: a single
    structurally wrong element moves no aggregate metric far enough to be seen."""
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(262144).astype(np.float32)
    got = ref.copy()
    got[12345] = 9.0
    r = gn.check(got, ref, atol=gn.atol_for(ref, ref))
    assert not r["pass"] and r["n_bad"] == 1 and r["worst_index"] == 12345
    # ... and the summary metric it would have hidden behind is essentially unmoved.
    assert r["mean_rel_L1"] < 1e-4


def test_a_relative_error_inside_the_bf16_tolerance_passes():
    """rtol is a tolerance, not a target: a datapath 1% off is inside the canonical bf16 band and
    must not be failed for it. The `x bf16 floor` note is what surfaces that, not the gate."""
    ref = np.full(4096, 2.0, np.float32)
    assert gn.check(ref * 1.005, ref, atol=0.0)["pass"]
    assert not gn.check(ref * 1.05, ref, atol=0.0)["pass"]


def test_atol_is_sized_off_the_bf16_floor_not_the_output_magnitude():
    """Two tensors of the same RMS but different cancellation must get different atol.

    This is the whole reason atol is derived from a measured floor rather than from rms: an output
    that is a difference of large terms is accurate to the scale of THOSE terms, and an atol read
    off its own magnitude would fail a perfect implementation.
    """
    rng = np.random.default_rng(1)
    clean = rng.standard_normal(65536).astype(np.float32)
    floor_clean = np.asarray(np.asarray(clean, BF16), np.float32)
    # Same rms, but each element is a small difference of two large numbers.
    big = rng.standard_normal(65536).astype(np.float32) * 100.0
    cancel = ((big + clean) - big).astype(np.float32)
    floor_cancel = (np.asarray(np.asarray(big + clean, BF16), np.float32)
                    - np.asarray(np.asarray(big, BF16), np.float32))
    assert gn.atol_for(cancel, floor_cancel) > 10 * gn.atol_for(clean, floor_clean)


def test_the_bf16_floor_always_passes_its_own_atol():
    """A perfect bf16 datapath must pass. If it did not, the gate would be unsatisfiable in exactly
    the way the identity gate it replaces was."""
    rng = np.random.default_rng(2)
    ref = (rng.standard_normal((256, 512)).astype(np.float32) @
           rng.standard_normal((512, 128)).astype(np.float32))
    floor = np.asarray(np.asarray(ref, BF16), np.float32)
    assert gn.check(floor, ref, atol=gn.atol_for(ref, floor))["pass"]


def test_shape_mismatch_is_an_error_not_a_verdict():
    with pytest.raises(ValueError):
        gn.check(np.zeros(10, np.float32), np.zeros(11, np.float32), atol=1.0)


# ------------------------------------------------------------------------------------------------
# TIER 1: the artifact plumbing. A missing reference must never read as a pass.
# ------------------------------------------------------------------------------------------------
def _artifact(tmp_path, refs, floors, name="out"):
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "designs", "decode_fused"))
    from prefill_ref import gate_block
    art = str(tmp_path)
    gate = gate_block(art, refs, floors)
    json.dump({"output": name, "gate": gate}, open(os.path.join(art, "meta.json"), "w"))
    return art


def test_an_artifact_without_a_gate_block_refuses_to_run(tmp_path):
    json.dump({"output": "out"}, open(tmp_path / "meta.json", "w"))
    with pytest.raises(SystemExit) as e:
        gn.load_gate(str(tmp_path))
    assert "--refresh-goldens" in str(e.value), "the failure must name the command that fixes it"


def test_a_missing_reference_file_is_an_error(tmp_path):
    rng = np.random.default_rng(3)
    ref = rng.standard_normal(1024).astype(np.float32)
    art = _artifact(tmp_path, {"out": ref}, {"out": np.asarray(np.asarray(ref, BF16), np.float32)})
    os.remove(os.path.join(art, "buffers", "golden_f32", "out.bin"))
    os.makedirs(os.path.join(art, "dump"), exist_ok=True)
    np.asarray(ref, BF16).tofile(os.path.join(art, "dump", "out.bin"))
    with pytest.raises(SystemExit) as e:
        gn.gate_artifact(art, os.path.join(art, "dump"))
    assert "missing" in str(e.value)


def test_a_stale_rtol_in_the_artifact_is_rejected(tmp_path):
    rng = np.random.default_rng(4)
    ref = rng.standard_normal(64).astype(np.float32)
    art = _artifact(tmp_path, {"out": ref}, {"out": np.asarray(np.asarray(ref, BF16), np.float32)})
    mp = os.path.join(art, "meta.json")
    meta = json.load(open(mp))
    meta["gate"]["rtol"] = 0.1
    json.dump(meta, open(mp, "w"))
    with pytest.raises(SystemExit) as e:
        gn.load_gate(art)
    assert "regenerate" in str(e.value)


def test_a_dump_of_the_wrong_length_is_an_error_not_a_comparison(tmp_path):
    """A truncated or stale dump must not be silently compared against a prefix of the reference."""
    rng = np.random.default_rng(5)
    ref = rng.standard_normal(1024).astype(np.float32)
    art = _artifact(tmp_path, {"out": ref}, {"out": np.asarray(np.asarray(ref, BF16), np.float32)})
    dump = os.path.join(art, "dump")
    os.makedirs(dump, exist_ok=True)
    np.asarray(ref[:512], BF16).tofile(os.path.join(dump, "out.bin"))
    with pytest.raises(SystemExit) as e:
        gn.gate_artifact(art, dump)
    assert "not the same build" in str(e.value)


def test_a_round_trip_through_a_real_artifact_passes_and_a_perturbed_one_fails(tmp_path):
    rng = np.random.default_rng(6)
    ref = (rng.standard_normal((64, 128)).astype(np.float32) @
           rng.standard_normal((128, 64)).astype(np.float32))
    floor = np.asarray(np.asarray(ref, BF16), np.float32)
    art = _artifact(tmp_path, {"out": ref}, {"out": floor})
    dump = os.path.join(art, "dump")
    os.makedirs(dump, exist_ok=True)
    np.asarray(floor, BF16).tofile(os.path.join(dump, "out.bin"))
    assert all(r["pass"] for _, r in gn.gate_artifact(art, dump))
    bad = floor.copy()
    bad[7] = bad[6]                                   # one stale row
    np.asarray(bad, BF16).tofile(os.path.join(dump, "out.bin"))
    assert not any(r["pass"] for _, r in gn.gate_artifact(art, dump))


# ------------------------------------------------------------------------------------------------
# TIER 2: the token-set rule.
# ------------------------------------------------------------------------------------------------
def _pair(ref_ids, npu_ids, tops):
    return ({"prompt_ids": [1], "gen_ids": ref_ids},
            {"prompt_ids": [1], "gen_ids": npu_ids, "topk_ids": tops, "k": 5})


def test_exact_parity_passes():
    ok, why, first = judge(*_pair([1, 2, 3], [1, 2, 3], [[1], [2], [3]]), k=5)
    assert ok and first is None and "parity" in why


def test_first_divergence_inside_topk_passes():
    ok, why, first = judge(*_pair([1, 2, 3], [1, 9, 3], [[1, 0, 0, 0, 0],
                                                         [9, 4, 2, 7, 8],
                                                         [3, 0, 0, 0, 0]]), k=5)
    assert ok and first == 1 and "rank-2" in why


def test_first_divergence_outside_topk_fails():
    ok, _, first = judge(*_pair([1, 2, 3], [1, 9, 3], [[1, 0, 0, 0, 0],
                                                       [9, 4, 5, 7, 8],
                                                       [3, 0, 0, 0, 0]]), k=5)
    assert not ok and first == 1


def test_only_the_first_divergence_is_judged():
    """Free-running decode is not independent between steps: after one different token every later
    step is on a different trajectory. A later step outside top-k cannot fail a run whose first
    divergence was inside it, and a later step inside top-k cannot rescue one that was outside."""
    tops = [[1, 0, 0, 0, 0], [9, 2, 0, 0, 0], [7, 0, 0, 0, 0]]
    ok, _, first = judge(*_pair([1, 2, 3], [1, 9, 7], tops), k=5)
    assert ok and first == 1


def test_k_narrows_the_set():
    tops = [[9, 4, 2, 7, 8]]
    assert judge(*_pair([2], [9], tops), k=5)[0]
    assert not judge(*_pair([2], [9], tops), k=2)[0]


def test_a_capture_without_topk_cannot_be_judged():
    """Silence is not a pass: a device file with no candidates recorded for the divergent step must
    fail and say why, not fall through to the exact-parity branch."""
    ok, why, _ = judge({"prompt_ids": [1], "gen_ids": [1, 2]},
                       {"prompt_ids": [1], "gen_ids": [1, 9]}, k=5)
    assert not ok and "--topk" in why


def test_the_gate_constants_are_the_documented_ones():
    import gate_llm_reference as glr
    assert (glr.GATE_N_TOKENS, glr.GATE_K) == (32, 5)


# ------------------------------------------------------------------------------------------------
# The shipped reference, as data. Cheap, and it catches a regenerated file that lost a field.
# ------------------------------------------------------------------------------------------------
REF_PATH = os.path.join(os.path.dirname(_HERE), "tests", "refs", "qwen3-0.6b", "gate_ref_n32.json")


@pytest.mark.skipif(not os.path.isfile(REF_PATH), reason="reference not generated")
def test_the_shipped_reference_is_complete_and_self_consistent():
    d = json.load(open(REF_PATH))
    assert d["n_tokens"] == 32 and d["k"] == 5
    assert len(d["gen_ids"]) == len(d["topk_ids"]) == len(d["margins"]) == 32
    for i, (g, t, m) in enumerate(zip(d["gen_ids"], d["topk_ids"], d["margins"])):
        assert t[0] == g, f"step {i}: gen_id must be the top-1 of its own top-k"
        assert len(t) == 5 and len(set(t)) == 5
        assert m >= 0, f"step {i}: top1-top2 margin cannot be negative"
    # The knife-edge step this gate exists for. If it ever stops being knife-edge, the argument in
    # tests/refs/qwen3-0.6b/README.md needs re-checking rather than the test being relaxed.
    assert min(d["margins"]) < 0.05


@pytest.mark.skipif(not os.path.isfile(REF_PATH), reason="reference not generated")
def test_the_comparator_runs_as_a_command_and_reports_pass():
    r = subprocess.run([sys.executable, os.path.join(_HERE, "gate_token_set.py"),
                        "--ref", REF_PATH, "--npu", REF_PATH], capture_output=True, text=True)
    assert r.returncode == 0 and "*** PASS ***" in r.stdout


# ------------------------------------------------------------------------------------------------
# attention_k_eq_v ablation: the numpy oracle's v_mode axis. The upcoming V-cache removal moves
# V's gainless RMSNorm from the point K is written to the point V is consumed -- same math, a
# different WHEN. Prove the two orderings are bit-identical before anything downstream depends on it.
# ------------------------------------------------------------------------------------------------
def _tiny_attention_k_eq_v_spec(qk_norm=False):
    """A one-layer, all-global spec exercising has_v_proj()==False (attention_k_eq_v): V has no
    projection of its own and must be derived from the raw k_proj output via a gainless RMSNorm --
    exactly the case the upcoming V-cache removal touches. Dims are the smallest that satisfy the
    reshape/GQA arithmetic in run_numpy(), not gemma4-12b's real ones.

    `qk_norm=True` is the real Gemma-4-12B axis combination (attention_k_eq_v + qk_norm both on) --
    needed to exercise `recompute_from_kc`'s gain-division step at all: with qk_norm off, kc holds
    RoPE(raw) with no gain and no per-position scalar, so a wrong gain formula would be invisible."""
    from llm_decode_spec import LlmSpec
    return LlmSpec(
        name="tiny-k-eq-v-qkn" if qk_norm else "tiny-k-eq-v",
        d_model=8, n_layers=1, n_q_heads=2, n_kv_heads=1, head_dim=4,
        ffn=8, vocab=16, eps=1e-6, act="silu", norm_gain="w",
        sandwich_norms=False, qk_norm=qk_norm, embed_scale="none",
        rope_theta_global=10_000.0, rope_theta_local=None,
        sliding_window=None, sw_pattern=None, query_pre_attn_scalar=None,
        attn_scale_fixed=None, v_from_k_on_global=True, v_norm=True,
        weight_prefix="model.",
    )


def _write_tiny_weights(weights_dir, sp):
    """Random (but deterministic) .npy weight tensors matching sp's shapes, for run_numpy() to load."""
    rng = np.random.default_rng(42)

    def save(name, shape):
        np.save(os.path.join(weights_dir, f"{name}.npy"),
                rng.standard_normal(shape).astype(np.float32))

    save(f"{sp.weight_prefix}embed_tokens.weight", (sp.vocab, sp.d_model))
    save(f"{sp.weight_prefix}norm.weight", (sp.d_model,))
    for l in range(sp.n_layers):
        p = f"{sp.weight_prefix}layers.{l}."
        save(p + "input_layernorm.weight", (sp.d_model,))
        save(p + "post_attention_layernorm.weight", (sp.d_model,))
        save(p + "self_attn.q_proj.weight", (sp.q_dim, sp.d_model))
        save(p + "self_attn.k_proj.weight", (sp.kv_dim, sp.d_model))
        assert not sp.has_v_proj(l), "this fixture only covers the no-v_proj (attention_k_eq_v) case"
        if sp.qk_norm:
            save(p + "self_attn.q_norm.weight", (sp.head_dim_for(l),))
            save(p + "self_attn.k_norm.weight", (sp.head_dim_for(l),))
        save(p + "self_attn.o_proj.weight", (sp.d_model, sp.q_dim))
        save(p + "mlp.gate_proj.weight", (sp.ffn, sp.d_model))
        save(p + "mlp.up_proj.weight", (sp.ffn, sp.d_model))
        save(p + "mlp.down_proj.weight", (sp.d_model, sp.ffn))


def test_v_skip_oracle_matches_stored_v_oracle_on_global_layer(tmp_path):
    """attention_k_eq_v: computing V's gainless RMSNorm at the point V is CONSUMED (the attention
    weighted sum) must give the bit-identical numpy result to today's compute-once-at-K-write
    oracle -- it's the same RMSNorm call over the same raw k projection, only relocated."""
    import gate_llm_reference as glr

    sp = _tiny_attention_k_eq_v_spec()
    _write_tiny_weights(str(tmp_path), sp)
    prompt_ids, n_tokens, k = [0, 1], 3, 2

    stored = glr.run_numpy(sp, str(tmp_path), prompt_ids, n_tokens, k, v_mode="store_at_write")
    recomputed = glr.run_numpy(sp, str(tmp_path), prompt_ids, n_tokens, k, v_mode="recompute_at_read")

    gen_s, tops_s, margins_s = stored
    gen_r, tops_r, margins_r = recomputed
    assert gen_s == gen_r
    assert margins_s == margins_r
    for (ids_s, vals_s), (ids_r, vals_r) in zip(tops_s, tops_r):
        assert ids_s == ids_r
        np.testing.assert_array_equal(vals_s, vals_r)


def test_v_skip_zero_storage_recompute_matches_the_stored_raw_k_reference(tmp_path):
    """v_mode='recompute_from_kc' derives V from the already-cached ROTATED kc (no new
    per-position array -- this is what the real kernels do), via RoPE-inversion and a divide by
    qk-norm's gain. It must match v_mode='recompute_at_read' (which stores true raw K separately
    and is correct by construction) within a small eps-order tolerance, not bit-exactly."""
    import gate_llm_reference as glr

    sp = _tiny_attention_k_eq_v_spec(qk_norm=True)
    _write_tiny_weights(str(tmp_path), sp)
    prompt_ids, n_tokens, k = [0, 1], 3, 2

    ref = glr.run_numpy(sp, str(tmp_path), prompt_ids, n_tokens, k, v_mode="recompute_at_read")
    got = glr.run_numpy(sp, str(tmp_path), prompt_ids, n_tokens, k, v_mode="recompute_from_kc")

    gen_ref, tops_ref, margins_ref = ref
    gen_got, tops_got, margins_got = got
    # The two paths take eps through a different denominator (s^2*mean(raw^2) here vs
    # mean(raw^2) directly in the true path -- see the module's derivation), so this is not
    # bit-exact. Measured residual on this fixture is ~1e-6-1e-7 (sp.eps order); atol is 100x
    # that ceiling, well below the ~0.87 a wrong gain formula produces (see this test's own
    # sabotage check in the commit history) -- tight enough to catch a real defect, loose
    # enough not to flake on eps-order noise.
    assert sp.eps == 1e-6, "atol below assumes this fixture's eps; re-check if eps changes"
    np.testing.assert_allclose(margins_got, margins_ref, rtol=0, atol=1e-4)
    assert gen_got == gen_ref


# ------------------------------------------------------------------------------------------------
# kv_dtype="int8": does int8-quantizing kc poison the RoPE-inversion V derivation? A sibling task
# is adding int8 K-cache quantization; nobody had numerically checked int8 -> RoPE-inversion ->
# gain-divide -> RMSNorm -> V, a different noise path than a plain stored-int8-V read.
# ------------------------------------------------------------------------------------------------
def _rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_int8_kc_recompute_from_kc_matches_bf16_within_bounds_over_free_running_generation(tmp_path):
    """kv_dtype="int8" round-trips kc through the real int8 packer (one group per row at
    group_size=head_dim) the moment K is written, so every later RoPE-inversion V-derivation reads
    the quantized value -- including across several FREE-RUNNING steps, where an early position's
    quantization affects every later step's attention over it, not just its own step."""
    import gate_llm_reference as glr

    sp = _tiny_attention_k_eq_v_spec(qk_norm=True)
    _write_tiny_weights(str(tmp_path), sp)
    prompt_ids, n_tokens, k = [0, 1], 6, 2

    cap_bf16, cap_int8 = {}, {}
    gen_bf16, _, margins_bf16 = glr.run_numpy(sp, str(tmp_path), prompt_ids, n_tokens, k,
                                              v_mode="recompute_from_kc", kv_dtype="bf16",
                                              capture_derived_v=cap_bf16)
    gen_int8, _, margins_int8 = glr.run_numpy(sp, str(tmp_path), prompt_ids, n_tokens, k,
                                              v_mode="recompute_from_kc", kv_dtype="int8",
                                              capture_derived_v=cap_int8)

    # Every layer/position derived under the bf16 run must also have been derived under int8, over
    # more than one free-running step -- otherwise this fixture would only be exercising the
    # prompt prefix, not the multi-token generation the sibling task's caution is about.
    assert cap_bf16.keys() == cap_int8.keys()
    positions = sorted(cap_bf16[0].keys())
    assert len(positions) >= prompt_ids.__len__() + 2, "fixture must exercise >1 free-running step"

    rel_l2s = [_rel_l2(cap_int8[l][p], cap_bf16[l][p]) for l in cap_bf16 for p in cap_bf16[l]]
    worst = max(rel_l2s)
    print(f"[int8-kc] derived-V rel-L2 over {len(rel_l2s)} (layer, position) points: "
         f"max={worst:.4f} mean={sum(rel_l2s) / len(rel_l2s):.4f}")

    # int8/group_size=head_dim round-trips at ~0.7-0.9% rel-L2 on raw data (sibling measurement).
    # CORRECTED after review: "RoPE-inversion is norm-preserving + RMSNorm is scale-invariant" is
    # NOT sufficient on its own -- the gain-divide sits between them, and dividing by a near-zero
    # PER-CHANNEL gain would distort direction in exactly that channel, which RMSNorm's overall
    # scale-invariance cannot undo (verified numerically: synthetic per-channel gain with
    # min|gain|~0.05 pushes rel-L2 to ~19%, past this bound). This tiny fixture's own random
    # k_norm.weight (seed 42) avoided that regime by luck, not by a general property of the math.
    # What actually makes this safe: Gemma-4-12B's REAL k_norm weights are UNIFORM per layer, not
    # per-channel-varying -- checked all 48 layers' real checkpoint tensors
    # (/mnt/data/xdna/artifacts/gemma4-12b/weights_int4g32qat_hf/*.self_attn.k_norm.weight.npy):
    # every layer's min/max/mean |w| agree to within ~1e-4 (global-layer min|w|=0.0605, layer 17,
    # uniform across all 512 channels). A UNIFORM gain is an overall rescale, which RoPE-inversion
    # and RMSNorm's scale-invariance genuinely do absorb -- confirmed by re-running this same
    # composition with that real gain vector: max rel-L2 1.13% over 200 trials, matching this test's
    # own measured range, not the danger-zone number above. If a future model variant's k_norm ever
    # becomes meaningfully non-uniform, THIS reasoning (not the disproven "always safe" claim) is
    # what must be re-checked before trusting int8 KV under kv_skip_v again.
    assert worst < 0.10, (
        f"int8 kc -> RoPE-inversion -> gain-divide -> RMSNorm amplified the raw ~0.8% round-trip "
        f"floor to {worst:.4f} rel-L2 -- report this, do not loosen the bound to hide it")

    # The strongest signal this project uses: does greedy decoding still land on the same tokens.
    assert gen_int8 == gen_bf16, (
        f"int8 kc flips a greedy token over free-running generation: bf16={gen_bf16} "
        f"int8={gen_int8} (margins bf16={margins_bf16} int8={margins_int8})")


def test_int8_kc_amplifies_under_a_non_uniform_near_zero_gain(tmp_path):
    """The test above passes because Gemma-4-12B's REAL k_norm weights are UNIFORM per layer
    (verified against the real checkpoint -- see that test's comment). This test proves the 0.10
    bound is a real detector, not dead code: a synthetic PER-CHANNEL-varying gain with one
    near-zero channel pushes the SAME composition (int8 kc -> RoPE-inversion -> gain-divide ->
    RMSNorm) well past the bound, confirming the danger scenario is real and would be caught if a
    future checkpoint's k_norm ever stopped being uniform."""
    import gate_llm_reference as glr

    sp = _tiny_attention_k_eq_v_spec(qk_norm=True)
    _write_tiny_weights(str(tmp_path), sp)
    # Overwrite the random k_norm.weight with a deliberately non-uniform vector, one channel
    # near zero -- the exact shape of gain that makes gain-divide direction-distorting instead of
    # an overall rescale.
    hd = sp.head_dim_for(0)
    bad_gain = np.full(hd, 0.25, dtype=np.float32)
    bad_gain[0] = 0.001
    gain_path = f"{sp.weight_prefix}layers.0.self_attn.k_norm.weight.npy"
    np.save(os.path.join(str(tmp_path), gain_path), bad_gain)

    cap_bf16, cap_int8 = {}, {}
    glr.run_numpy(sp, str(tmp_path), [0, 1], 6, 2, v_mode="recompute_from_kc", kv_dtype="bf16",
                  capture_derived_v=cap_bf16)
    glr.run_numpy(sp, str(tmp_path), [0, 1], 6, 2, v_mode="recompute_from_kc", kv_dtype="int8",
                  capture_derived_v=cap_int8)
    rel_l2s = [_rel_l2(cap_int8[l][p], cap_bf16[l][p]) for l in cap_bf16 for p in cap_bf16[l]]
    assert max(rel_l2s) > 0.10, (
        "expected the non-uniform near-zero-gain case to exceed the 0.10 bound the real-weight "
        "test relies on -- if it doesn't, that bound is not actually detecting this failure mode")


# ------------------------------------------------------------------------------------------------
# rope_impl="poly" (Task 2 of the kv_skip_v plan): the device forms V by inverting K's RoPE
# rotation on-chip, and the chosen scheme is the int_phase polynomial (measure_rope_poly.py) --
# not host float64 cos/sin. `rope_impl` swaps that in ONLY for recompute_from_kc's inverse-rotation
# call (see gate_llm_reference.run_numpy's docstring); forward Q/K RoPE stays exact, matching the
# device (which always uses the host LUT there). Two tests: the tiny fixture's small positions
# cannot exercise the large-p regime the poly's error bound was measured against, so that regime
# gets its own unit test of phase_cs() directly, at Gemma-4's real geometries and p up to 262144.
# ------------------------------------------------------------------------------------------------
def test_int_phase_matches_exact_cos_sin_at_gemma4_geometries_and_large_positions():
    """Unit-level check of the chosen on-chip scheme itself (measure_rope_poly.phase_cs), at the
    two Gemma-4-12B RoPE geometries and positions up to 262144 -- the regime
    measure_rope_poly.py measured max|dcos|=1.908e-4, max|dsin|=1.907e-4 over. Bound is 3e-4, ~1.6x
    that measurement (headroom for a handful of positions this test picks rather than the
    exhaustive sweep, not a re-tuned pass) and still ~20x tighter than the 5.86e-3 bf16 gate."""
    import measure_rope_poly as rp

    BOUND = 3e-4
    geoms = [
        ("global", 512, 1_000_000.0, 0.25),
        ("local", 256, 10_000.0, None),
    ]
    positions = np.array([0, 1, 17, 4096, 131_072, 262_144 - 1, 262_144], dtype=np.int64)
    for label, hd, theta, partial in geoms:
        inv = rp.rope_inv_freq(hd, theta, partial)
        inv = inv[inv != 0.0]
        F = rp.inv_freq_to_turns_u32(inv)
        for p in positions:
            pos_u64 = np.array(p, dtype=np.int64).view(np.uint64)
            ph = (pos_u64 * F.astype(np.uint64)) & rp.U32_MASK
            s, c = rp.phase_cs(ph)
            ref_c, ref_s = np.cos(p * inv), np.sin(p * inv)
            max_c, max_s = np.max(np.abs(c - ref_c)), np.max(np.abs(s - ref_s))
            assert max_c <= BOUND and max_s <= BOUND, (
                f"{label} geometry, p={p}: max|dcos|={max_c:.4e} max|dsin|={max_s:.4e} "
                f"exceed {BOUND:.1e}")


def test_kv_skip_v_int_phase_rope_matches_exact_within_bound_over_free_running_generation(tmp_path):
    """rope_impl="poly" must reproduce rope_impl="exact"'s derived V (recompute_from_kc) and the
    resulting greedy tokens, over several free-running steps. Two independent checks:

    1. STRUCTURAL (capture_forward_k): forward Q/K rope must be BIT-IDENTICAL between the two
       runs -- proves the flag reached only the inverse-rotation call, not a numeric coincidence.
       At this fixture's tiny positions (<=7) int_phase's own error is ~1e-8, so a rel-L2 bound
       alone cannot tell "forward rope untouched" apart from "forward rope also uses poly, but
       poly is accurate here too" -- confirmed by sabotage: routing poly into the forward calls
       too still passes every rel-L2/token assertion below, and only this bit-identity check
       catches it (see the commit message for the sabotage run).
    2. NUMERIC (derived-V rel-L2 + greedy tokens): the inverse-rotation swap itself doesn't move
       the answer at this fixture's scale (see the unit test above for the large-position error
       this fixture is too small to exercise) and doesn't flip a sign on the -pos inversion.
    """
    import gate_llm_reference as glr

    sp = _tiny_attention_k_eq_v_spec(qk_norm=True)
    _write_tiny_weights(str(tmp_path), sp)
    prompt_ids, n_tokens, k = [0, 1], 6, 2

    cap_exact, cap_poly = {}, {}
    fwd_exact, fwd_poly = {}, {}
    gen_exact, _, margins_exact = glr.run_numpy(sp, str(tmp_path), prompt_ids, n_tokens, k,
                                                v_mode="recompute_from_kc", rope_impl="exact",
                                                capture_derived_v=cap_exact,
                                                capture_forward_k=fwd_exact)
    gen_poly, _, margins_poly = glr.run_numpy(sp, str(tmp_path), prompt_ids, n_tokens, k,
                                              v_mode="recompute_from_kc", rope_impl="poly",
                                              capture_derived_v=cap_poly,
                                              capture_forward_k=fwd_poly)

    assert fwd_exact.keys() == fwd_poly.keys()
    for l in fwd_exact:
        for p in fwd_exact[l]:
            assert np.array_equal(fwd_exact[l][p], fwd_poly[l][p]), (
                f"layer {l} position {p}: forward-RoPE'd K differs between rope_impl='exact' and "
                f"'poly' -- the flag is affecting forward Q/K rope, not just the inverse rotation")

    assert cap_exact.keys() == cap_poly.keys()
    positions = sorted(cap_exact[0].keys())
    assert len(positions) >= len(prompt_ids) + 2, "fixture must exercise >1 free-running step"

    rel_l2s = [_rel_l2(cap_poly[l][p], cap_exact[l][p]) for l in cap_exact for p in cap_exact[l]]
    worst, mean = max(rel_l2s), sum(rel_l2s) / len(rel_l2s)
    print(f"[rope-poly] derived-V rel-L2 over {len(rel_l2s)} (layer, position) points: "
         f"max={worst:.2e} mean={mean:.2e}")

    # At this fixture's positions (<=7) int_phase's phase error is ~1e-8 rad (see the module
    # docstring's p*2**-33-turns bound) -- orders below bf16 noise. 1e-3 is a real bound, not a
    # rubber stamp: it would catch a sign flip on the -pos inversion at this fixture's scale (the
    # forward-leak wiring bug is caught by the bit-identity check above, not by this one).
    assert worst < 1e-3, (
        f"rope_impl='poly' moved derived V by {worst:.4f} rel-L2 at these small positions -- "
        f"report this, do not loosen the bound to hide it")
    assert gen_poly == gen_exact, (
        f"rope_impl='poly' flips a greedy token over free-running generation: "
        f"exact={gen_exact} poly={gen_poly} (margins exact={margins_exact} poly={margins_poly})")
