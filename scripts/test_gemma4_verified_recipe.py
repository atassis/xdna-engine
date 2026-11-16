#!/usr/bin/env python3
"""Construction-only contract for the unregistered Gemma4 verified-data recipe."""
import hashlib
import json
import os
import shutil
from contextlib import ExitStack
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_artifact_dispatch as dispatch
from test_verified_gemma4_data import (Gemma4PublishedDataTests as _Gemma4PublishedFixture,
                                       Gemma4WitnessTests as _Gemma4WitnessFixture)


def witness_fixture(case, fixture_class=_Gemma4WitnessFixture):
    fixture = fixture_class('test_positive_build_load_and_verify')
    fixture.setUp()
    case.addCleanup(fixture.doCleanups)
    return fixture


del _Gemma4WitnessFixture

REPO = Path(__file__).resolve().parents[1]
SCENARIO = "scenarios/generate-gemma4-12b-resident-256k.toml"
PIN = "a" * 64


class Gemma4VerifiedRecipeTests(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("TMPDIR", "/tmp"))
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="g4-recipe-", dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.owner = self.root / "accepted"
        self.source = self.owner / "gemma4-12b"
        self.witness = self.root / "review" / "witness.json"
        self.artifacts, self.inputs, self.build = (self.root / name for name in ("artifacts", "inputs", "build"))

    def recipe(self, **kwargs):
        values = {"witness": self.witness, "witness_sha256": PIN, "data_source": self.source,
                  "accepted_data_root": self.owner}
        values.update(kwargs)
        return dispatch.gemma4_verified_recipe(
            SCENARIO, REPO, self.artifacts, self.inputs, self.build, **values)

    def test_constructs_exact_three_step_pinned_closure(self):
        recipe = self.recipe()
        self.assertEqual(recipe.model, "gemma4-12b")
        self.assertEqual(recipe.inputs, (str(self.source / "store"), str(self.source / "rf_stack"),
                                         str(self.source / "hf_config" / "config.json"),
                                         str(self.source / "tokenizer"), str(self.witness)))
        self.assertEqual(len(recipe.outputs), 3)
        self.assertEqual(tuple(item.step for item in recipe.commands), ("verified-data", "build", "package"))
        data, build, package = recipe.commands
        proof = self.build / "gemma4-12b" / "verified-data"
        self.assertEqual(data.argv, (dispatch.sys.executable, str(REPO / "scripts/gemma4_verified_entry.py"), "proof",
                                     "--source", str(self.source), "--accepted-data-root", str(self.owner),
                                     "--witness", str(self.witness), "--expected-witness-sha256", PIN,
                                     "--out", str(proof)))
        self.assertEqual(data.stage.dependencies, ())
        self.assertEqual(data.stage.outputs[0].patterns, (".verified-data-witness.json",))
        self.assertTrue(data.stage.outputs[0].require_each)
        self.assertFalse(data.stage.toolchain)
        self.assertEqual(build.stage.dependencies, ("verified-data",))
        self.assertEqual(build.env["RF_STORE"], str(self.source / "store"))
        self.assertEqual(build.env["RF_HF_CONFIG"], str(self.source / "hf_config" / "config.json"))
        self.assertEqual(build.env["KEEP_WORK"], build.env["RF_BUILD"])
        self.assertEqual(package.stage.dependencies, ("verified-data", "build"))
        self.assertEqual(tuple(view.root for view in package.stage.outputs), recipe.outputs)
        self.assertEqual(package.argv[2], "package")
        self.assertEqual(package.env, {"RF_MODEL": "gemma4-12b",
                                       "RF_STORE": str(self.source / "store"),
                                       "RF_HF_CONFIG": str(self.source / "hf_config" / "config.json")})
        self.assertTrue(all(str(self.inputs) not in value for value in package.env.values()))
        command_declarations = " ".join(" ".join(item.argv) + " " + item.step for item in recipe.commands)
        self.assertNotIn("gemma4_data.sh", command_declarations)
        self.assertNotIn("dump_llm_weights.py", command_declarations)
        self.assertFalse(any(item.step.startswith("data-") for item in recipe.commands))
        self.assertIn("scripts/gemma4_verified_entry.py", package.stage.sources)
        self.assertIn("scripts/verified_gemma4_data.py", package.stage.sources)
        self.assertIn("scripts/verified_gemma_data.py", package.stage.sources)
        self.assertIn("designs/resident_forward/artifact_meta.py", package.stage.sources)
        self.assertIn("designs/resident_forward/package_artifact.py", package.stage.sources)
        self.assertIn("third_party/iron/iron/common/quant.py", package.stage.sources)
        self.assertEqual(package.stage.symbols, ("package_gemma4", "copy_tokenizer"))

    def test_rebase_changes_only_candidate_and_build_paths(self):
        recipe = self.recipe()
        staged = dispatch.rebase_recipe(recipe, self.artifacts, self.root / "stage-artifacts",
                                        self.build, self.root / "stage-build")
        self.assertEqual(staged.inputs, recipe.inputs)
        self.assertEqual(staged.commands[0].argv[4], str(self.source))
        self.assertEqual(staged.commands[0].argv[8], str(self.witness))
        self.assertEqual(staged.commands[0].argv[-1], str(self.root / "stage-build/gemma4-12b/verified-data"))
        self.assertEqual(staged.commands[1].env["RF_BUILD"], str(self.root / "stage-build/gemma4-12b/rf48C"))
        self.assertEqual(staged.commands[1].env["KEEP_WORK"], staged.commands[1].env["RF_BUILD"])
        self.assertEqual(staged.commands[2].argv[staged.commands[2].argv.index("--artifacts-root") + 1],
                         str(self.root / "stage-artifacts"))
        self.assertEqual(staged.commands[2].env["RF_STORE"], str(self.source / "store"))
        self.assertEqual(staged.commands[2].env["RF_HF_CONFIG"], str(self.source / "hf_config" / "config.json"))

    def test_validator_accepts_actual_whitespace_witness_without_writing(self):
        fixture = witness_fixture(self)
        self.owner, self.source = fixture.owner, fixture.root
        document = fixture.witness()
        raw = json.dumps(document, indent=2).encode() + b'\n'
        self.witness.parent.mkdir(parents=True)
        self.witness.write_bytes(raw)
        pin = hashlib.sha256(raw).hexdigest()
        recipe = self.recipe(witness_sha256=pin)
        before = sorted(self.root.rglob('*'))
        with mock.patch.object(dispatch, 'build_recipe', side_effect=AssertionError('must not build')):
            dispatch.validate_gemma4_verified_recipe(recipe, REPO, self.artifacts, self.build)
        self.assertEqual(before, sorted(self.root.rglob('*')))

    def test_validator_reconstructs_the_complete_closure_and_ignores_template_inputs(self):
        fixture = witness_fixture(self)
        self.owner, self.source = fixture.owner, fixture.root
        raw = json.dumps(fixture.witness(), separators=(',', ':')).encode()
        self.witness.parent.mkdir(parents=True)
        self.witness.write_bytes(raw)
        recipe = self.recipe(witness_sha256=hashlib.sha256(raw).hexdigest())
        alternate = dispatch.gemma4_verified_recipe(SCENARIO, REPO, self.artifacts, self.root / 'inputs-A', self.build,
                                                    witness=self.witness, witness_sha256=hashlib.sha256(raw).hexdigest(),
                                                    data_source=self.source, accepted_data_root=self.owner)
        other = dispatch.gemma4_verified_recipe(SCENARIO, REPO, self.artifacts, self.root / 'inputs-B', self.build,
                                                witness=self.witness, witness_sha256=hashlib.sha256(raw).hexdigest(),
                                                data_source=self.source, accepted_data_root=self.owner)
        self.assertEqual(dispatch.stage_digest(dispatch.asdict(alternate)), dispatch.stage_digest(dispatch.asdict(other)))
        mutations = (
            replace(recipe, commands=(recipe.commands[1], recipe.commands[0], recipe.commands[2])),
            replace(recipe, commands=(recipe.commands[0], recipe.commands[1], recipe.commands[1])),
            replace(recipe, commands=recipe.commands[:2]),
            replace(recipe, commands=(replace(recipe.commands[0], argv=("/foreign/python", *recipe.commands[0].argv[1:])),
                                      *recipe.commands[1:])),
            replace(recipe, commands=(replace(recipe.commands[0], argv=(*recipe.commands[0].argv[:3], "--witness", str(self.witness), *recipe.commands[0].argv[3:])),
                                      *recipe.commands[1:])),
            replace(recipe, commands=(replace(recipe.commands[0], stage=replace(recipe.commands[0].stage,
                                                                                  inputs=recipe.commands[0].stage.inputs[:-1])),
                                      *recipe.commands[1:])),
            replace(recipe, outputs=(str(self.root / 'foreign-output'), *recipe.outputs[1:])),
            replace(recipe, commands=(recipe.commands[0], replace(recipe.commands[1], env=recipe.commands[1].env | {'RF_STORE': 'foreign'}), recipe.commands[2])),
        )
        for mutated in mutations:
            with self.subTest(mutated=mutated.commands[0].step):
                with self.assertRaises((ValueError, dispatch.verified_gemma4.VerifiedDataError)):
                    dispatch.validate_gemma4_verified_recipe(mutated, REPO, self.artifacts, self.build)
        with mock.patch.dict(os.environ, {'RF_STORE': 'foreign', 'RF_HF_CONFIG': 'foreign'}, clear=False):
            # The environment is part of the reconstructed graph even though this recipe has no raw inputs.
            with self.assertRaises(ValueError):
                dispatch.validate_gemma4_verified_recipe(recipe, REPO, self.artifacts, self.build)

    def test_validator_refuses_bad_pin_and_source_before_any_writer(self):
        fixture = witness_fixture(self)
        self.owner, self.source = fixture.owner, fixture.root
        raw = json.dumps(fixture.witness()).encode()
        self.witness.parent.mkdir(parents=True)
        self.witness.write_bytes(raw)
        recipe = self.recipe(witness_sha256='0' * 64)
        with mock.patch.object(dispatch, 'build_recipe', side_effect=AssertionError('must not build')):
            with self.assertRaises(dispatch.verified_gemma4.VerifiedDataError):
                dispatch.validate_gemma4_verified_recipe(recipe, REPO, self.artifacts, self.build)
        valid = self.recipe(witness_sha256=hashlib.sha256(raw).hexdigest())
        blob = self.source / 'store' / 'blobs' / ('%064x.bin' % 0)
        blob.write_bytes(b'changed')
        with self.assertRaises(dispatch.verified_gemma4.VerifiedDataError):
            dispatch.validate_gemma4_verified_recipe(valid, REPO, self.artifacts, self.build)
        alias = self.root / 'alias-source'
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(Exception):
            dispatch.validate_gemma4_verified_recipe(self.recipe(data_source=alias), REPO, self.artifacts, self.build)
        bad_source = replace(valid.commands[0], argv=(*valid.commands[0].argv[:4],
                                                       str(self.source / '..' / 'gemma4-12b'),
                                                       *valid.commands[0].argv[5:]))
        with self.assertRaises(Exception):
            dispatch.validate_gemma4_verified_recipe(replace(valid, commands=(bad_source, *valid.commands[1:])),
                                                     REPO, self.artifacts, self.build)

    def test_verified_data_factory_registers_the_pinned_gemma4_closure(self):
        recipe = dispatch.verified_data_recipe(
            "gemma4-12b", SCENARIO, REPO, self.artifacts, self.inputs, self.build,
            witness=self.witness, witness_sha256=PIN, data_source=self.source,
            accepted_data_root=self.owner)
        standalone = self.recipe()
        self.assertEqual(dispatch.stage_digest(dispatch.asdict(recipe)),
                         dispatch.stage_digest(dispatch.asdict(standalone)))
        for pin in (None, PIN.upper()):
            with self.subTest(pin=pin):
                with self.assertRaises(ValueError):
                    dispatch.verified_data_recipe(
                        "gemma4-12b", SCENARIO, REPO, self.artifacts, self.inputs, self.build,
                        witness=self.witness, witness_sha256=pin, data_source=self.source,
                        accepted_data_root=self.owner)

    def test_input_adapter_requires_explicit_context_before_raw_member_checks(self):
        fixture = witness_fixture(self)
        self.owner, self.source = fixture.owner, fixture.root
        raw = json.dumps(fixture.witness(), separators=(",", ":")).encode()
        self.witness.parent.mkdir(parents=True)
        self.witness.write_bytes(raw)
        recipe = self.recipe(witness_sha256=hashlib.sha256(raw).hexdigest())
        with mock.patch.object(dispatch, "require_members", side_effect=AssertionError("raw member check")):
            with self.assertRaisesRegex(ValueError, "explicit pin context"):
                dispatch.validate_recipe_inputs([recipe])
        with mock.patch.object(dispatch, "require_members", side_effect=AssertionError("raw member check")):
            dispatch.validate_recipe_inputs([recipe], repo=REPO, artifacts=self.artifacts, build=self.build)

    def test_output_adapter_verifies_published_data_without_the_old_source(self):
        fixture = _Gemma4PublishedFixture('test_published_data_is_source_independent')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        raw = (fixture.resident / dispatch.verified_gemma4.SIDECAR).read_bytes()
        recipe = self.recipe(witness_sha256=hashlib.sha256(raw).hexdigest())
        (fixture.resident / 'meta.json').write_text('{}')
        recipe = replace(recipe, outputs=(str(fixture.resident), str(fixture.tokenizer), str(fixture.checkpoint)))
        shutil.rmtree(fixture.fixture.root)
        # Output validation must not declare/probe the historical proof path either.
        self.build.symlink_to(self.root / "foreign-inaccessible", target_is_directory=True)
        dispatch.validate_recipe_outputs(recipe)
        dispatch.validate_stage_outputs(recipe, "package")
        with self.assertRaises(ValueError):
            dispatch.validate_recipe_outputs(replace(recipe, outputs=recipe.outputs[:2]))

    def test_stage_proof_adapter_reads_only_the_pinned_sidecar(self):
        fixture = witness_fixture(self)
        self.owner, self.source = fixture.owner, fixture.root
        raw = json.dumps(fixture.witness(), indent=2).encode() + b"\n"
        self.witness.parent.mkdir(parents=True)
        self.witness.write_bytes(raw)
        pin = hashlib.sha256(raw).hexdigest()
        recipe = self.recipe(witness_sha256=pin)
        proof = Path(recipe.commands[0].argv[-1])
        proof.mkdir(parents=True)
        (proof / dispatch.verified_gemma4.SIDECAR).write_bytes(raw)
        shutil.rmtree(self.source)
        dispatch.validate_stage_outputs(recipe, dispatch.VERIFIED_DATA_STEP)
        (proof / dispatch.verified_gemma4.SIDECAR).write_bytes(raw + b" ")
        with self.assertRaises(dispatch.verified_gemma4.VerifiedDataError):
            dispatch.validate_stage_outputs(recipe, dispatch.VERIFIED_DATA_STEP)

    def test_ordinary_gemma4_recipe_remains_an_eight_root_source_recipe(self):
        ordinary = dispatch.recipe_for("gemma4-12b", SCENARIO, REPO, self.artifacts,
                                       self.inputs, self.build)
        self.assertEqual(len(ordinary.outputs), 8)
        self.assertNotIn(dispatch.VERIFIED_DATA_STEP, [item.step for item in ordinary.commands])

    def test_build_recipe_automatically_gates_gemma4_before_any_effect(self):
        fixture = witness_fixture(self)
        self.owner, self.source = fixture.owner, fixture.root
        raw = json.dumps(fixture.witness(), separators=(",", ":")).encode()
        self.witness.parent.mkdir(parents=True)
        self.witness.write_bytes(raw)
        recipe = self.recipe(witness_sha256=hashlib.sha256(raw).hexdigest())

        def effects_forbidden(*_args, **_kwargs):
            raise AssertionError("pre-gate effect reached")

        malformed = (
            replace(recipe, commands=(replace(recipe.commands[0], argv=(
                *recipe.commands[0].argv[:10], "0" * 64, *recipe.commands[0].argv[11:])),
                *recipe.commands[1:])),
            replace(recipe, commands=(replace(recipe.commands[0], stage=None), *recipe.commands[1:])),
            replace(recipe, outputs=(str(self.root / "foreign-output"), *recipe.outputs[1:])),
            replace(recipe, commands=(replace(recipe.commands[0], argv=(
                "/foreign/python", *recipe.commands[0].argv[1:])), *recipe.commands[1:])),
            replace(recipe, commands=(recipe.commands[0], replace(recipe.commands[1], env={}),
                                      *recipe.commands[2:])),
        )
        for candidate in malformed:
            with self.subTest(candidate=candidate.commands[0].argv[0]):
                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(dispatch, "authority_digest",
                                                          side_effect=effects_forbidden))
                    stack.enter_context(mock.patch.object(dispatch, "resource_locks",
                                                          side_effect=effects_forbidden))
                    stack.enter_context(mock.patch.object(dispatch, "run_commands",
                                                          side_effect=effects_forbidden))
                    stack.enter_context(mock.patch.object(dispatch.subprocess, "run",
                                                          side_effect=effects_forbidden))
                    with self.assertRaises((ValueError, dispatch.verified_gemma4.VerifiedDataError)):
                        dispatch.build_recipe(candidate, REPO, REPO / SCENARIO,
                                              self.artifacts, self.build)
        with mock.patch.object(dispatch, "authority_digest", side_effect=RuntimeError("gate reached")) as authority:
            with self.assertRaisesRegex(RuntimeError, "gate reached"):
                dispatch.build_recipe(recipe, REPO, REPO / SCENARIO, self.artifacts, self.build)
        authority.assert_called_once()
        (self.source / "store" / "blobs" / ("%064x.bin" % 0)).write_bytes(b"changed")
        with mock.patch.object(dispatch, "authority_digest", side_effect=effects_forbidden):
            with self.assertRaises(dispatch.verified_gemma4.VerifiedDataError):
                dispatch.build_recipe(recipe, REPO, REPO / SCENARIO, self.artifacts, self.build)

    def test_refuses_wrong_scenario_pin_alias_and_rebased_source_or_witness(self):
        with self.assertRaises(ValueError):
            dispatch.gemma4_verified_recipe("scenarios/wrong.toml", REPO, self.artifacts, self.inputs, self.build,
                                             witness=self.witness, witness_sha256=PIN, data_source=self.source,
                                             accepted_data_root=self.owner)
        with self.assertRaises(ValueError):
            self.recipe(witness_sha256="")
        with self.assertRaises(Exception):
            self.recipe(data_source=self.source / ".." / "gemma4-12b")
        with self.assertRaises(RuntimeError):
            self.recipe(data_source=self.artifacts / "gemma4-12b", accepted_data_root=self.artifacts)
        with self.assertRaises(RuntimeError):
            self.recipe(witness=self.build / "witness.json")


if __name__ == "__main__":
    unittest.main()
