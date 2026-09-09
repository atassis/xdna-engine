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
