import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_artifact_dispatch import (
    Command,
    Recipe,
    authority_digest,
    copy_tree,
    copy_tokenizer,
    materialize_input,
    build_recipe,
    configured_pyannote_sdks,
    package_gemma4,
    ensure_outputs_ready,
    plan as build_plan,
    publish_recipe,
    provision_inputs,
    rebase_recipe,
    recipe_key,
    recipe_for,
    source_manifest_path,
    run_commands,
    validate_recipe_inputs,
    validate_recipe_outputs,
    verified_existing_gemma4,
    validate_command_executables,
)

REPO = Path(__file__).resolve().parents[1]
DISPATCH = REPO / "scripts/model_artifact_dispatch.py"
CONFIG = REPO / "scripts/tests/fixtures/model-artifact-all-models.engine.toml"


def concurrent_build(root, recipe, artifacts, build, entered, release, result,
                     fail_publish=False, attempting=None, forced_key=None, contended=None):
    import model_artifact_dispatch as dispatch

    def toy_builder(staged, repo):
        for raw in staged.outputs:
            target = Path(raw)
            target.mkdir(parents=True, exist_ok=True)
            (target / "value").write_text("new")
        entered.set()
        if not release.wait(10):
            raise RuntimeError("fixture release timed out")

    original_hit = dispatch.cache_hit
    original_flock = fcntl.flock

    def observed_flock(handle, mode):
        try:
            original_flock(handle, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            if contended is not None:
                contended.set()
            original_flock(handle, mode)

    def checked_hit(candidate, key):
        if fail_publish and candidate.outputs == recipe.outputs:
            return False
        return original_hit(candidate, key)

    with patch.object(dispatch, "LOCK_ROOT", root / "locks", create=True), \
            patch.object(dispatch, "run_commands", toy_builder), \
            patch.object(dispatch, "cache_hit", checked_hit), \
            patch.object(fcntl, "flock", observed_flock):
        if forced_key is not None:
            dispatch.recipe_key = lambda *args, **kwargs: forced_key
        if attempting is not None:
            attempting.set()
        try:
            result.put(("ok", build_recipe(recipe, root, root / "engine.toml", artifacts, build)))
        except Exception as error:
            result.put(("error", str(error)))


class ModelArtifactConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "scenario.toml").write_text("fixture")
        (self.root / "engine.toml").write_text("fixture")
        self.context = multiprocessing.get_context("spawn")

    def recipe(self, outputs=("model",), model="fixture", changed=False):
        return Recipe(model, "scenario.toml", (),
                      tuple(str(self.root / "artifacts" / name) for name in outputs),
                      (Command("toy", ("toy",), {"VERSION": "two"}),) if changed else ())

    def start(self, recipe, *, artifacts=None, build=None, fail_publish=False, forced_key=None):
        entered, release, attempting, contended = (self.context.Event() for _ in range(4))
        result = self.context.Queue()
        process = self.context.Process(target=concurrent_build, args=(
            self.root, recipe, artifacts or self.root / "artifacts",
            build or self.root / "build", entered, release, result, fail_publish, attempting,
            forced_key, contended))
        process.start()
        self.addCleanup(self.stop, process, release)
        self.assertTrue(attempting.wait(5))
        return process, entered, release, result, contended

    @staticmethod
    def stop(process, release):
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)

    def finish(self, worker):
        process, entered, release, result = worker[:4]
        release.set()
        process.join(10)
        self.assertFalse(process.is_alive())
        self.assertEqual(process.exitcode, 0)
        return result.get(timeout=2)

    def serialized(self, first_recipe, second_recipe, **second_options):
        first = self.start(first_recipe)
        self.assertTrue(first[1].wait(5))
        second = self.start(second_recipe, **second_options)
        self.assertTrue(second[4].wait(5), "second writer did not contend on the shared resource")
        self.assertFalse(second[1].is_set())
        inodes = {path.name: path.stat().st_ino for path in (self.root / "locks").iterdir()}
        self.assertEqual(self.finish(first), ("ok", "BUILT"))
        outcome = self.finish(second)
        self.assertEqual({name: (self.root / "locks" / name).stat().st_ino for name in inodes}, inodes)
        return outcome

    def test_same_output_same_key_waits_then_reuses(self):
        recipe = self.recipe()
        self.assertEqual(self.serialized(recipe, recipe), ("ok", "HIT"))
        self.assertEqual(list((self.root / "build/model-artifact-staging").iterdir()), [])

    def test_changed_key_same_output_with_different_build_root_waits(self):
        self.assertEqual(self.serialized(self.recipe(), self.recipe(changed=True),
                                        build=self.root / "other-build"), ("ok", "BUILT"))

    def test_shared_staging_across_artifact_roots_waits(self):
        first = self.start(self.recipe(), forced_key="shared-stage-key")
        self.assertTrue(first[1].wait(5))
        artifacts = self.root / "other-artifacts"
        recipe = Recipe("fixture", "scenario.toml", (), (str(artifacts / "model"),), ())
        second = self.start(recipe, artifacts=artifacts, forced_key="shared-stage-key")
        self.assertTrue(second[4].wait(5), "shared stage did not contend through a different output root")
        self.assertFalse(second[1].is_set())
        self.assertEqual(self.finish(first), ("ok", "BUILT"))
        self.assertEqual(self.finish(second), ("ok", "BUILT"))

    def test_overlapping_output_sets_in_reverse_order_wait(self):
        self.assertEqual(self.serialized(self.recipe(("a", "b")),
                                        self.recipe(("b", "a"), changed=True)), ("ok", "BUILT"))

    def test_independent_targets_can_build_together(self):
        first = self.start(self.recipe())
        self.assertTrue(first[1].wait(5))
        second = self.start(self.recipe(("other",), model="other"))
        self.assertTrue(second[1].wait(5), "independent target blocked")
        self.assertEqual(self.finish(second), ("ok", "BUILT"))
        self.assertEqual(self.finish(first), ("ok", "BUILT"))

    def test_failed_publication_recovery_is_not_overwritten_on_retry(self):
        recipe = self.recipe()
        live = Path(recipe.outputs[0])
        live.mkdir(parents=True)
        (live / "value").write_text("old")
        first = self.start(recipe, fail_publish=True)
        self.assertTrue(first[1].wait(5))
        retry = self.start(recipe, fail_publish=True)
        self.assertTrue(retry[4].wait(5))
        self.assertFalse(retry[1].is_set())
        outcome = self.finish(first)
        self.assertEqual(outcome[0], "error")
        self.assertIn("published receipt did not verify", outcome[1])
        backup = next((self.root / "artifacts/.model-artifact-rollback").rglob("value"))
        self.assertEqual(backup.read_text(), "old")
        outcome = self.finish(retry)
        self.assertEqual(outcome[0], "error")
        self.assertIn("publication recovery requires inspection", outcome[1])
        self.assertFalse(retry[1].is_set())
        self.assertEqual(backup.read_text(), "old")

    def test_parent_output_excludes_a_nested_writer(self):
        self.assertEqual(self.serialized(self.recipe(), self.recipe(("model/child",), changed=True),
                                        build=self.root / "other-build"), ("ok", "BUILT"))


class ModelArtifactStageLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "scenario.toml").write_text("fixture")
        self.config = self.root / "engine.toml"
        self.config.write_text("fixture")
        self.artifacts, self.build = self.root / "artifacts", self.root / "build"
        self.recipe = Recipe("fixture", "scenario.toml", (),
                             (str(self.artifacts / "model"),), ())
        self.shared = self.artifacts / "shared-blobs/value.bin"
        self.shared.parent.mkdir(parents=True)
        self.shared.write_bytes(b"packed")
        self.external = self.root / "outside-stage"
        self.external.mkdir()
        (self.external / "source").write_bytes(b"unique")
        self.stage = self.build / "not-started"
        self.fail_builder = False
        self.calls = 0

    def builder(self, staged, repo):
        self.calls += 1
        output = Path(staged.outputs[0])
        output.mkdir(parents=True)
        (output / "value").write_bytes(b"published")
        self.stage = output.parents[1]
        pool = self.stage / "artifacts/shared-blobs"
        pool.mkdir()
        os.link(self.shared, pool / "value.bin")
        (self.stage / "build").mkdir()
        (self.stage / "build/intermediate").write_bytes(b"disposable")
        (self.stage / "external").symlink_to(self.external, target_is_directory=True)
        if self.fail_builder:
            raise RuntimeError("builder failed")

    def run_build(self):
        import model_artifact_dispatch as dispatch
        with patch.object(dispatch, "LOCK_ROOT", self.root / "locks"), \
                patch.object(dispatch, "run_commands", self.builder):
            return build_recipe(self.recipe, self.root, self.config, self.artifacts, self.build)

    def test_verified_publication_expires_all_transaction_residue(self):
        self.assertEqual(self.run_build(), "BUILT")
        self.assertFalse(self.stage.exists())
        self.assertEqual((self.artifacts / "model/value").read_bytes(), b"published")
        self.assertEqual(self.shared.read_bytes(), b"packed")
        self.assertEqual(self.shared.stat().st_nlink, 1)
        self.assertEqual((self.external / "source").read_bytes(), b"unique")

    def test_failed_builder_retains_its_stage_and_live_output(self):
        live = self.artifacts / "model/value"
        live.parent.mkdir()
        live.write_bytes(b"previous")
        self.fail_builder = True
        with self.assertRaisesRegex(RuntimeError, "builder failed"):
            self.run_build()
        self.assertEqual((self.stage / "build/intermediate").read_bytes(), b"disposable")
        self.assertEqual(live.read_bytes(), b"previous")

    def test_published_receipt_uses_canonical_command_paths(self):
        output = str(self.artifacts / "model")
        input_path = str(self.build / "input")
        self.recipe = Recipe("fixture", "scenario.toml", (), (output,),
                             (Command("builder", ("fixture", output, input_path),
                                      {"WEIGHTS": input_path}),))
        self.assertEqual(self.run_build(), "BUILT")
        receipt = Path(output) / ".model-recipe.json"
        recorded = json.loads(receipt.read_text())["commands"][0]
        self.assertEqual(recorded["argv"], ["fixture", output, input_path])
        self.assertEqual(recorded["env"], {"WEIGHTS": input_path})
        self.assertNotIn("model-artifact-staging", receipt.read_text())
        self.assertFalse(self.stage.exists())

    def test_publication_rejection_retains_stage_and_rollback(self):
        import model_artifact_dispatch as dispatch
        live = self.artifacts / "model/value"
        live.parent.mkdir()
        live.write_bytes(b"previous")
        original_hit = dispatch.cache_hit

        def reject(recipe, key, **kwargs):
            return False if recipe.outputs == self.recipe.outputs else original_hit(recipe, key, **kwargs)

        with patch.object(dispatch, "cache_hit", reject):
            with self.assertRaisesRegex(RuntimeError, "published receipt did not verify"):
                self.run_build()
        self.assertTrue(self.stage.is_dir())
        backup = next((self.artifacts / ".model-artifact-rollback").rglob("value"))
        self.assertEqual(backup.read_bytes(), b"previous")

    def test_repeat_hit_does_not_build_or_collect_other_transaction(self):
        self.assertEqual(self.run_build(), "BUILT")
        other = self.build / "model-artifact-staging/fixture-other-key/input"
        other.parent.mkdir(parents=True)
        other.write_bytes(b"pending")
        self.assertEqual(self.run_build(), "HIT")
        self.assertEqual(self.calls, 1)
        self.assertEqual(other.read_bytes(), b"pending")
        self.assertFalse(self.stage.exists())


class Qwen35StoragePilotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.repo = self.root / "repo"
        (self.repo / "scenarios").mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        self.scenario = "scenarios/generate-qwen3.5-4b.toml"
        (self.repo / self.scenario).write_text("fixture")
        self.config = self.repo / "engine.toml"
        self.config.write_text("fixture")
        self.artifacts, self.build = self.root / "artifacts", self.root / "build"
        self.recipe = recipe_for("qwen3.5-4b", self.scenario, self.repo,
                                 self.artifacts, self.root / "inputs", self.build)
        source = Path(self.recipe.inputs[0])
        source.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
            (source / name).write_text("{}")
        self.dump = self.build / "not-started"
        self.calls = []
        self.missing = None
        self.metadata_update = lambda step, meta: None
        lock_patch = patch("model_artifact_dispatch.LOCK_ROOT", self.root / "locks")
        lock_patch.start()
        self.addCleanup(lock_patch.stop)

    def subprocess_output(self, argv, *, check, cwd, env):
        self.assertEqual(cwd, self.repo)
        self.calls.append(argv)
        step = next(c.step for c in self.recipe.commands if Path(c.argv[1]).name == Path(argv[1]).name
                    and (c.step != "tokenizer" or "--copy-tokenizer" in argv))
        if step == "weights":
            self.dump = Path(argv[argv.index("--out") + 1])
            self.dump.mkdir(parents=True)
            (self.dump / "tensor.npy").write_bytes(b"raw-fixture")
        elif step in {"decode", "prefill"}:
            self.assertEqual(Path(env["WEIGHTS"]), self.dump)
            self.assertTrue((self.dump / "tensor.npy").is_file())
            out = Path(argv[-1])
            (out / "buffers").mkdir(parents=True)
            meta = {"elf": f"{step}.elf", "weights": ["W"],
                    "layout": {"W": {"len": 6}}}
            (out / f"{step}.elf.zst").write_bytes(b"controlled-elf")
            if step == "decode":
                meta["embed_blob"] = "W_embed"
                (out / "buffers/W.bin").write_bytes(b"packed")
                (out / "buffers/W_embed.bin").write_bytes(b"embed")
            else:
                decode = Path(env["DECODE_META"])
                meta.update(weights_from=str(decode.parent / "buffers"),
                            decode_artifact={"meta": str(decode)}, arena_shared=True)
                meta["weights"].append("P")
                meta["layout"]["P"] = {"len": 5}
                (out / "buffers/P.bin").write_bytes(b"local")
            self.metadata_update(step, meta)
            (out / "meta.json").write_text(json.dumps(meta))
            if self.missing and step == self.missing[0]:
                (out / self.missing[1]).unlink()
        else:
            copy_tokenizer(Path(argv[-2]), Path(argv[-1]))
            if self.missing and self.missing[0] == "tokenizer":
                (Path(argv[-1]) / self.missing[1]).unlink()
        return subprocess.CompletedProcess(argv, 0)

    def run_build(self):
        with patch("model_artifact_dispatch.subprocess.run", self.subprocess_output):
            return build_recipe(self.recipe, self.repo, self.config, self.artifacts, self.build)

    def test_missing_final_blob_refuses_publication_and_retains_dump(self):
        self.missing = ("decode", "buffers/W.bin")
        live = Path(self.recipe.outputs[0])
        live.mkdir(parents=True)
        (live / "old.bin").write_bytes(b"old")
        with self.assertRaisesRegex(FileNotFoundError, "W.bin"):
            self.run_build()
        self.assertEqual((live / "old.bin").read_bytes(), b"old")
        self.assertTrue(self.dump.is_dir())

    def test_missing_required_companions_fail_before_publication(self):
        for step, filename in (("decode", "decode.elf.zst"), ("decode", "buffers/W_embed.bin"),
                               ("prefill", "prefill.elf.zst"), ("prefill", "buffers/P.bin"),
                               ("prefill", "meta.json"), ("tokenizer", "tokenizer.json"),
                               ("tokenizer", "tokenizer_config.json"), ("tokenizer", "config.json")):
            with self.subTest(step=step, filename=filename):
                self.missing = (step, filename)
                with self.assertRaises(FileNotFoundError):
                    self.run_build()
                self.assertTrue(self.dump.is_dir())
                self.assertFalse(Path(self.recipe.outputs[0]).exists())

    def test_invalid_blob_length_is_not_accepted(self):
        self.metadata_update = lambda step, meta: meta["layout"]["W"].update(len=7)
        with self.assertRaisesRegex(ValueError, "blob length"):
            self.run_build()
        self.assertTrue(self.dump.exists())

    def test_shared_prefill_weight_must_be_uploaded_by_decode(self):
        def missing_upload(step, meta):
            if step == "decode":
                meta["layout"]["P"] = {"len": 5}
        self.metadata_update = missing_upload
        with self.assertRaisesRegex(ValueError, "not uploaded by decode"):
            self.run_build()
        self.assertTrue(self.dump.exists())

    def test_existing_persistent_dump_and_checkpoint_are_not_collected(self):
        retained = self.artifacts / "qwen3.5-4b/weights/tensor.npy"
        retained.parent.mkdir(parents=True)
        retained.write_bytes(b"retained")
        checkpoint = Path(self.recipe.inputs[0], "model.safetensors")
        original = checkpoint.read_bytes()
        self.assertEqual(self.run_build(), "BUILT")
        self.assertEqual(retained.read_bytes(), b"retained")
        self.assertEqual(checkpoint.read_bytes(), original)

    def test_partial_publication_preserves_recovery_and_raw_dump(self):
        import model_artifact_dispatch as dispatch
        for raw in self.recipe.outputs:
            live = Path(raw)
            live.mkdir(parents=True)
            (live / "old.bin").write_bytes(b"old")
        original_replace = dispatch.os.replace

        def fail_prefill(source, target):
            if Path(target) == Path(self.recipe.outputs[1]) and "model-artifact-staging" in str(source):
                raise OSError("fixture publication failure")
            return original_replace(source, target)

        with patch.object(dispatch.os, "replace", fail_prefill):
            with self.assertRaisesRegex(OSError, "fixture publication failure"):
                self.run_build()
        recovery = next((self.artifacts / ".model-artifact-rollback").rglob("old.bin"))
        self.assertEqual(recovery.read_bytes(), b"old")
        self.assertEqual(Path(self.recipe.outputs[1], "old.bin").read_bytes(), b"old")
        self.assertEqual(Path(self.recipe.outputs[2], "old.bin").read_bytes(), b"old")
        self.assertTrue(self.dump.exists())

    def test_prefill_cannot_require_dump_as_its_weight_source(self):
        def require_dump(step, meta):
            if step == "prefill":
                meta["weights_from"] = str(self.dump)
        self.metadata_update = require_dump
        with self.assertRaisesRegex(ValueError, "published decode"):
            self.run_build()
        self.assertTrue(self.dump.exists())

    def test_unknown_dump_reference_is_rejected_not_stripped(self):
        self.metadata_update = lambda step, meta: meta.update(unknown_dependency=str(self.dump))
        with self.assertRaisesRegex(RuntimeError, "staged path leaked"):
            self.run_build()
        metadata = next(self.dump.parents[2].glob("artifacts/**/meta.json"))
        self.assertEqual(json.loads(metadata.read_text())["unknown_dependency"], str(self.dump))
        self.assertTrue(self.dump.exists())

    def test_command_failure_preserves_old_output_and_raw_recovery(self):
        live = Path(self.recipe.outputs[0])
        live.mkdir(parents=True)
        (live / "old.bin").write_bytes(b"old")
        original = self.subprocess_output

        def fail_prefill(argv, **kwargs):
            result = original(argv, **kwargs)
            if Path(argv[1]).name == "build_prefill.sh":
                raise subprocess.CalledProcessError(17, argv)
            return result

        with patch("model_artifact_dispatch.subprocess.run", fail_prefill):
            with self.assertRaises(subprocess.CalledProcessError):
                build_recipe(self.recipe, self.repo, self.config, self.artifacts, self.build)
        self.assertEqual((live / "old.bin").read_bytes(), b"old")
        self.assertTrue(self.dump.is_dir())

    def test_publication_rejection_retains_dump_and_previous_outputs(self):
        import model_artifact_dispatch as dispatch
        live = Path(self.recipe.outputs[0])
        live.mkdir(parents=True)
        (live / "old.bin").write_bytes(b"old")
        original_hit = dispatch.cache_hit

        def reject(candidate, key, **kwargs):
            if candidate.outputs == self.recipe.outputs:
                return False
            return original_hit(candidate, key, **kwargs)

        with patch.object(dispatch, "cache_hit", reject):
            with self.assertRaisesRegex(RuntimeError, "published receipt did not verify"):
                self.run_build()
        self.assertTrue(self.dump.is_dir())
        recovery = next((self.artifacts / ".model-artifact-rollback").rglob("old.bin"))
        self.assertEqual(recovery.read_bytes(), b"old")
        calls = len(self.calls)
        with self.assertRaisesRegex(RuntimeError, "recovery requires inspection"):
            self.run_build()
        self.assertEqual(len(self.calls), calls)
        self.assertTrue(self.dump.is_dir())

    def test_changed_checkpoint_and_quant_authority_rebuild(self):
        self.assertEqual(self.run_build(), "BUILT")
        first_calls = len(self.calls)
        Path(self.recipe.inputs[0], "model.safetensors").write_bytes(b"changed")
        self.assertEqual(self.run_build(), "BUILT")
        self.assertEqual(len(self.calls), first_calls * 2)
        quant = self.repo / "designs/decode_fused/quant.py"
        quant.parent.mkdir(parents=True)
        quant.write_text("quant = 2\n")
        self.assertEqual(self.run_build(), "BUILT")
        self.assertEqual(len(self.calls), first_calls * 3)
        self.assertEqual(self.run_build(), "HIT")
        self.assertEqual(len(self.calls), first_calls * 3)

    def test_missing_published_blob_invalidates_receipt(self):
        self.assertEqual(self.run_build(), "BUILT")
        Path(self.recipe.outputs[0], "buffers/W.bin").unlink()
        self.assertEqual(self.run_build(), "BUILT")
        self.assertEqual(len(self.calls), 8)

    def test_different_data_roots_build_only_in_their_own_stage(self):
        self.assertEqual(self.run_build(), "BUILT")
        first_dump = self.dump
        self.artifacts, self.build = self.root / "other/artifacts", self.root / "other/build"
        self.recipe = recipe_for("qwen3.5-4b", self.scenario, self.repo,
                                 self.artifacts, self.root / "inputs", self.build)
        self.assertEqual(self.run_build(), "BUILT")
        self.assertNotEqual(first_dump, self.dump)
        self.assertTrue(self.dump.is_relative_to(self.build / "model-artifact-staging"))
        self.assertFalse(self.dump.exists())

    def test_pilot_keeps_checkpoint_and_quantization_contract(self):
        commands = {command.step: command for command in self.recipe.commands}
        weights = commands["weights"]
        self.assertEqual(weights.argv[2:-1], ("--spec", "qwen3.5-4b", "--checkpoint-dir",
                                            self.recipe.inputs[0], "--out"))
        self.assertEqual(json.loads(commands["decode"].env["PRECISION"]),
                         {site: "int4/g32/clip" for site in ("mlp", "attn_o", "qkv")})
        self.assertEqual(commands["decode"].env["WEIGHTS"], weights.argv[-1])
        self.assertEqual(commands["prefill"].env["WEIGHTS"], weights.argv[-1])

    def test_verified_publication_expires_transaction_local_stage(self):
        import model_artifact_dispatch as dispatch
        accepted = []
        original_hit = dispatch.cache_hit

        def observe(candidate, key, **kwargs):
            hit = original_hit(candidate, key, **kwargs)
            if candidate.outputs == self.recipe.outputs and hit:
                self.assertTrue(self.dump.is_dir())
                accepted.append(True)
            return hit

        with patch.object(dispatch, "cache_hit", observe):
            self.assertEqual(self.run_build(), "BUILT")
        self.assertTrue(accepted)
        self.assertTrue(self.dump.is_relative_to(self.build / "model-artifact-staging"))
        self.assertIn("build", self.dump.relative_to(self.build / "model-artifact-staging").parts)
        self.assertNotIn(str(self.artifacts / "qwen3.5-4b/weights"), self.recipe.outputs)
        self.assertFalse(self.dump.exists())
        self.assertFalse(self.dump.parents[2].exists())
        self.assertFalse((self.artifacts / "qwen3.5-4b/weights").exists())
        decode, prefill, tokenizer = map(Path, self.recipe.outputs)
        meta = json.loads((prefill / "meta.json").read_text())
        self.assertEqual(meta["weights_from"], str(decode / "buffers"))
        self.assertEqual(meta["decode_artifact"]["meta"], str(decode / "meta.json"))
        validate_recipe_outputs(self.recipe)
        receipt = (decode / ".model-recipe.json").read_text()
        self.assertNotIn("model-artifact-staging", receipt)
        calls = len(self.calls)
        self.assertEqual(self.run_build(), "HIT")
        self.assertEqual(len(self.calls), calls)


class ModelArtifactDispatchPlanTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        lock_patch = patch("model_artifact_dispatch.LOCK_ROOT", Path(directory.name) / "locks", create=True)
        lock_patch.start()
        self.addCleanup(lock_patch.stop)

    def test_coordination_locks_follow_data_root_without_creating_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "selected-data"
            env = dict(os.environ, XDNA_DATA=str(data), HOME=str(root / "home"))
            result = subprocess.run(
                [sys.executable, "-c", "from model_artifact_dispatch import LOCK_ROOT; print(LOCK_ROOT)"],
                cwd=REPO / "scripts", env=env, capture_output=True, text=True, check=True,
            )
            self.assertEqual(Path(result.stdout.strip()), data / "locks/model-artifacts")
            self.assertFalse(data.exists())

    def test_install_exports_its_selected_onnx_runtime_environment(self) -> None:
        import re
        import shlex
        install = (REPO / "install.sh").read_text()
        assignment = re.search(r"^(?:export )?ONNX_ASR_VENV=.*$", install, re.MULTILINE)
        self.assertIsNotNone(assignment)
        assert assignment is not None
        env = {key: value for key, value in os.environ.items() if key != "ONNX_ASR_VENV"}
        probe = "import os; assert os.environ.get('ONNX_ASR_VENV') == '/repo/.venv-export'"
        script = "REPO=/repo; " + assignment.group(0) + "; " + shlex.quote(sys.executable) + " -c " + shlex.quote(probe)
        result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tokenizer_embeds_its_declared_jinja_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.mkdir()
            (source / "tokenizer_config.json").write_text("{}")
            template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}"
            (source / "chat_template.jinja").write_text(template)
            copy_tokenizer(source, target)
            self.assertEqual(json.loads((target / "tokenizer_config.json").read_text())["chat_template"], template)
            self.assertEqual((target / "chat_template.jinja").read_text(), template)

    def test_copy_tree_dereferences_file_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.mkdir()
            blob = root / "blob.bin"
            blob.write_bytes(b"weights")
            (source / "weights.bin").symlink_to(blob)
            copy_tree(source, target)
            blob.unlink()
            self.assertEqual((target / "weights.bin").read_bytes(), b"weights")

    def test_recipe_keeps_symlinked_outputs_inside_its_staging_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, snapshot = root / "artifacts", root / "snapshot"
            (artifacts / "qwen3-0.6b").mkdir(parents=True)
            snapshot.mkdir()
            (artifacts / "qwen3-0.6b/tokenizer").symlink_to(snapshot, target_is_directory=True)
            recipe = recipe_for("qwen3-0.6b", "scenarios/generate-qwen3-0.6b.toml",
                                REPO, artifacts, root / "inputs", root / "build")
            staged = rebase_recipe(recipe, artifacts, root / "staging", root / "build", root / "staged-build")
            tokenizer = next(item for item in staged.commands if item.step == "tokenizer")
            self.assertEqual(tokenizer.argv[-1], str(root / "staging/qwen3-0.6b/tokenizer"))
            self.assertIn(str(root / "staging/qwen3-0.6b/tokenizer"), staged.outputs)

    def test_gemma4_packages_tower_checkpoint_without_full_text_weights(self) -> None:
        import numpy as np
        from safetensors.numpy import load_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, source = root / "artifacts", root / "source"
            source.mkdir()
            (source / "config.json").write_text("{}")
            (source / "tokenizer.json").write_text("{}")
            (source / "model.safetensors").write_bytes(b"unneeded-text-stack")
            store = artifacts / "gemma4-12b/store"
            (store / "blobs").mkdir(parents=True)
            array = np.array([1.25, 2.5], dtype=np.float32)
            (store / "blobs/tower.bin").write_bytes(array.tobytes())
            (store / "manifest.json").write_text(json.dumps({"towers": {
                "model.embed_audio.embedding_projection.weight": {
                    "blob": "tower", "offset": 0, "length": array.nbytes,
                    "shape": list(array.shape), "dtype": "float32", "layout": "raw_native"}}}))
            with patch("model_artifact_dispatch.subprocess.run"):
                package_gemma4(REPO, artifacts, root / "build", root / "out", source, root / "tokenizer")
            checkpoint = artifacts / "gemma4-12b-qat/checkpoint"
            saved = load_file(checkpoint / "model.safetensors")
            self.assertEqual(list(saved), ["model.embed_audio.embedding_projection.weight"])
            np.testing.assert_array_equal(next(iter(saved.values())), array)

    def test_gemma4_build_consumes_its_staged_store(self) -> None:
        artifacts, build = REPO / "data/artifacts", REPO / "data/build"
        recipe = recipe_for("gemma4-12b", "scenarios/generate-gemma4-12b-resident-256k.toml",
                            REPO, artifacts, REPO / "data/model-inputs", build)
        stage = build / "staging/gemma4"
        staged = rebase_recipe(recipe, artifacts, stage / "artifacts", build, stage / "build")
        command = next(item for item in staged.commands if item.step == "build")
        self.assertEqual(command.env.get("RF_STORE"), str(stage / "artifacts/gemma4-12b/store"))

    def test_published_prefill_metadata_resolves_its_published_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scenario.toml").write_text("[scenario]\nkind = 'generate'\n")
            config = root / "engine.toml"
            config.write_text("[[model]]\nname = 'fixture'\nscenario = 'scenario.toml'\n")
            artifacts, build = root / "artifacts", root / "build"
            decode, prefill = artifacts / "decode", artifacts / "prefill"
            builder = root / "scripts/builder.py"
            builder.write_text(
                "import os,json\nfrom pathlib import Path\n"
                "d=Path(os.environ['DECODE']);p=Path(os.environ['PREFILL'])\n"
                "(d/'buffers').mkdir(parents=True);p.mkdir(parents=True)\n"
                "(d/'meta.json').write_text('{}');(d/'buffers/W.bin').write_bytes(b'weights')\n"
                "(p/'prefill.elf').write_bytes(b'program')\n"
                "(p/'meta.json').write_text(json.dumps({'weights_from':str(d/'buffers'),"
                "'decode_artifact':{'meta':str(d/'meta.json')}}))\n")
            recipe = Recipe("fixture", "scenario.toml", (), (str(decode), str(prefill)),
                            (Command("build", (sys.executable, str(builder)),
                                     {"DECODE": str(decode), "PREFILL": str(prefill)}),))
            self.assertEqual(build_recipe(recipe, root, config, artifacts, build), "BUILT")
            metadata = json.loads((prefill / "meta.json").read_text())
            self.assertEqual(metadata["weights_from"], str(decode / "buffers"))
            self.assertEqual(metadata["decode_artifact"]["meta"], str(decode / "meta.json"))
            self.assertEqual((Path(metadata["weights_from"]) / "W.bin").read_bytes(), b"weights")
            self.assertEqual((prefill / "prefill.elf").read_bytes(), b"program")
            self.assertEqual(build_recipe(recipe, root, config, artifacts, build), "HIT")

    def test_staging_rebase_does_not_rewrite_a_replacement_twice(self) -> None:
        artifacts, build = Path("/data/artifacts"), Path("/data/build")
        stage = build / "model-artifacts/staging/model"
        recipe = Recipe("fixture", "scenario", (), (str(artifacts / "model"),),
                        (Command("build", ("builder", str(build / "work")),
                                 {"META": str(artifacts / "decode/meta.json")}),))
        staged = rebase_recipe(recipe, artifacts, stage / "artifacts", build, stage / "build")
        self.assertEqual(staged.outputs, (str(stage / "artifacts/model"),))
        self.assertEqual(staged.commands[0].argv[1], str(stage / "build/work"))
        self.assertEqual(staged.commands[0].env["META"], str(stage / "artifacts/decode/meta.json"))

    def test_source_identity_ignores_bytecode_and_git_administration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "scripts/helper.py"
            helper.parent.mkdir()
            helper.write_text("value = 1\n")
            (root / "third_party/iron").mkdir(parents=True)
            recipe = Recipe("fixture", "scenario.toml", (), (), ())
            first = authority_digest(recipe, root)
            for rel in ("scripts/__pycache__/helper.cpython-314.pyc",
                        "third_party/iron/.git",
                        "third_party/iron/.pytest_cache/v/cache/nodeids"):
                generated = root / rel
                generated.parent.mkdir(parents=True, exist_ok=True)
                generated.write_bytes(b"generated")
            self.assertEqual(authority_digest(recipe, root), first)
            helper.write_text("value = 2\n")
            self.assertNotEqual(authority_digest(recipe, root), first)

    def test_weight_recipes_bind_the_declared_iron_source(self) -> None:
        for model, scenario, step in (
            ("gemma3-270m", "scenarios/generate-gemma3-270m.toml", "weights"),
            ("qwen3-0.6b", "scenarios/generate-qwen3-0.6b.toml", "weights"),
            ("qwen3.5-4b", "scenarios/generate-qwen3.5-4b.toml", "weights"),
            ("gemma4-12b", "scenarios/generate-gemma4-12b-resident-256k.toml", "data"),
        ):
            with self.subTest(model=model):
                recipe = recipe_for(model, scenario, REPO, REPO / "data/artifacts",
                                    REPO / "data/model-inputs", REPO / "data/build")
                build = next(command for command in recipe.commands if command.step == step)
                self.assertEqual(build.env.get("IRON_DIR"), str(REPO / "third_party/iron"))
                self.assertEqual(build.env.get("PYTHONPATH"), str(REPO / "third_party/iron"))

    def test_install_provisions_weight_dump_dependencies_before_rust_build(self) -> None:
        install = (REPO / "install.sh").read_text()
        self.assertIn("requirements-model-weights.txt", install)
        self.assertLess(install.index("requirements-model-weights.txt"), install.index("cargo build --release"))
        requirements = (REPO / "scripts/requirements-model-weights.txt").read_text().splitlines()
        self.assertIn("torch==2.12.0+cpu", requirements)
        self.assertIn("safetensors==0.8.0", requirements)
        self.assertIn("huggingface-hub==0.36.2", requirements)

    def test_community_sdk_pins_checkpoint_loader_fix(self) -> None:
        requirements = (REPO / "scripts/requirements-pyannote-community.txt").read_text().splitlines()
        self.assertIn("pyannote.audio==4.0.3", requirements)
        self.assertIn("torchcodec==0.7.0", requirements)
        self.assertIn("torch==2.8.0+cpu", requirements)
        self.assertIn("torchaudio==2.8.0+cpu", requirements)

    def test_install_provisions_only_the_pyannote_sdks_selected_by_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "engine.toml"
            config.write_text(
                "[[model]]\nname = 'pyannote-community-1'\nscenario = 'scenarios/diarize-pyannote-community-1.toml'\n"
                "[[model]]\nname = 'pyannote-3.1'\nscenario = 'scenarios/diarize-pyannote-3.1.toml'\n"
            )
            self.assertEqual(configured_pyannote_sdks(config), ("community", "legacy"))

        install = (REPO / "install.sh").read_text()
        self.assertIn("--pyannote-sdks", install)
        self.assertIn("setup_pyannote_community_venv.sh", install)
        self.assertIn("setup_pyannote_venv.sh", install)
        self.assertLess(install.index("--pyannote-sdks"), install.index("cargo build --release"))

    def test_community_recipe_uses_its_own_sdk_and_validates_local_checkpoints_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            community_python = root / "pyannote-community/bin/python"
            legacy_python = root / "pyannote-3.1/bin/python"
            with patch.dict(os.environ, {
                "MODEL_PYANNOTE_COMMUNITY_PY": str(community_python),
                "MODEL_PYANNOTE_PY": str(legacy_python),
            }, clear=False):
                community = recipe_for(
                    "pyannote-community-1",
                    "scenarios/diarize-pyannote-community-1.toml",
                    REPO,
                    root / "artifacts",
                    root / "inputs",
                    root / "build",
                )
                legacy = recipe_for(
                    "pyannote-3.1",
                    "scenarios/diarize-pyannote-3.1.toml",
                    REPO,
                    root / "artifacts",
                    root / "inputs",
                    root / "build",
                )

            self.assertEqual([command.step for command in community.commands], ["validate", "export"])
            self.assertEqual(community.commands[0].argv[0], str(community_python))
            self.assertEqual(legacy.commands[0].argv[0], str(legacy_python))
            self.assertIn("verify_pyannote_local_load.py", community.commands[0].argv[1])
            self.assertEqual(community.commands[0].argv[2], community.inputs[0])

    def test_community_sdk_identity_invalidates_the_recipe_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenarios").mkdir()
            (root / "scripts").mkdir()
            (root / "designs/decode_fused").mkdir(parents=True)
            (root / "designs/resident_forward").mkdir(parents=True)
            (root / "aie_kernels").mkdir()
            (root / "third_party/iron").mkdir(parents=True)
            (root / "toolchain.lock").write_text("PIN=one\n")
            (root / "scenarios/diarize-pyannote-community-1.toml").write_text("[scenario]\nkind = 'diarize'\n")
            config = root / "engine.toml"
            config.write_text("[[model]]\nname = 'pyannote-community-1'\nscenario = 'scenarios/diarize-pyannote-community-1.toml'\n")
            source = root / "input"
            source.mkdir()
            (source / "config.yaml").write_text("pipeline:\n  params:\n    segmentation: $model/segmentation\n    embedding: $model/embedding\n")
            for name in ("segmentation", "embedding"):
                checkpoint = source / name / "pytorch_model.bin"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(name.encode())
            exporter = root / "scripts/export_pyannote.py"
            validator = root / "scripts/verify_pyannote_local_load.py"
            exporter.write_text("export = 1\n")
            validator.write_text("validate = 1\n")
            recipe = Recipe(
                "pyannote-community-1",
                "scenarios/diarize-pyannote-community-1.toml",
                (str(source),),
                (str(root / "output"),),
                (Command("validate", (str(root / "community-python"), str(validator), str(source)), {}),
                 Command("export", (str(root / "community-python"), str(exporter)), {})),
            )

            with patch("model_artifact_dispatch.pyannote_sdk_identity", return_value="sdk-one"):
                first = recipe_key(recipe, root, config)
            with patch("model_artifact_dispatch.pyannote_sdk_identity", return_value="sdk-two"):
                second = recipe_key(recipe, root, config)

            self.assertNotEqual(first, second)

    def test_onnx_external_data_gets_a_private_inode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            payload = source / "encoder-model.onnx.data"
            payload.write_bytes(b"external tensor fixture")
            target = root / "input"
            copy_tree(source, target)
            copied = target / payload.name
            self.assertEqual(copied.read_bytes(), payload.read_bytes())
            self.assertEqual(copied.stat().st_nlink, 1)
            self.assertNotEqual(copied.stat().st_ino, payload.stat().st_ino)

    def test_dotted_model_directory_is_not_treated_as_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "snapshot"
            source.mkdir()
            (source / "config.yaml").write_text("pipeline: fixture\n")
            target = root / "inputs/speaker-diarization-3.1"
            recipe = Recipe("pyannote-3.1", "fixture", (str(target),), (), ())
            self.assertEqual(source_manifest_path(recipe), target / ".model-input-source.json")
            materialize_input(recipe, source, "fixture", "fixture")
            self.assertTrue((target / "config.yaml").is_file())

    def test_recipe_count_follows_the_configured_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "engine.toml"
            config.write_text('[[model]]\nname = "bge-base"\nscenario = "scenarios/bge-base.toml"\n')
            recipes = build_plan(config, REPO, root / "artifacts", root / "inputs", root / "build")
            self.assertEqual(len(recipes), 1)

    def plan(self) -> list[dict]:
        result = subprocess.run(
            [
                sys.executable,
                str(DISPATCH),
                "--plan",
                "--config",
                str(CONFIG),
                "--repo",
                str(REPO),
                "--artifacts-root",
                "/model-artifacts",
                "--model-input-root",
                "/model-inputs",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_selected_config_has_one_recipe_for_each_configured_model(self) -> None:
        plan = self.plan()
        self.assertEqual(len(plan), 10)
        self.assertEqual(
            [entry["model"] for entry in plan],
            [
                "parakeet",
                "bge-base",
                "pyannote-community-1",
                "pyannote-3.1",
                "whisper-turbo",
                "gemma3-270m",
                "qwen3-0.6b",
                "gemma4-12b",
                "qwen3.5-4b",
                "espcn",
            ],
        )
        self.assertTrue(all(entry["scenario"].startswith("scenarios/") for entry in plan))
        self.assertTrue(all(entry["commands"] for entry in plan))

    def test_generate_pairs_build_decode_before_prefill_with_absolute_metadata(self) -> None:
        plan = {entry["model"]: entry for entry in self.plan()}
        for model in ("qwen3-0.6b", "qwen3.5-4b"):
            commands = plan[model]["commands"]
            steps = [command["step"] for command in commands]
            self.assertLess(steps.index("decode"), steps.index("prefill"))
            prefill = commands[steps.index("prefill")]
            self.assertTrue(prefill["env"]["DECODE_META"].startswith("/model-artifacts/"))
            self.assertNotIn("$PWD", prefill["env"]["DECODE_META"])

    def test_resident_recipe_packages_the_declared_artifact(self) -> None:
        gemma4 = next(entry for entry in self.plan() if entry["model"] == "gemma4-12b")
        self.assertEqual([command["step"] for command in gemma4["commands"]], ["data", "build", "package"])
        outputs = set(gemma4["outputs"])
        self.assertTrue({
            "/model-artifacts/gemma4-12b/resident_rf48C_p7148a7",
            "/model-artifacts/gemma4-12b/tokenizer",
            "/model-artifacts/gemma4-12b-qat/checkpoint",
            "/model-artifacts/gemma4-12b/rf_stack",
            "/model-artifacts/gemma4-12b/store",
            "/model-artifacts/gemma4-12b/weights_int4g32sbf16_planar_qat_rg",
        }.issubset(outputs))

    def test_mocked_command_failure_propagates_without_running_a_model(self) -> None:
        recipe = Recipe(
            model="dispatch-fixture",
            scenario="scenarios/asr.toml",
            inputs=(),
            outputs=(),
            commands=(Command("fixture", (sys.executable, "-c", "raise SystemExit(17)"), {}),),
        )
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            run_commands(recipe, REPO)
        self.assertEqual(caught.exception.returncode, 17)

    def test_unmanaged_artifact_output_is_left_for_transactional_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact"
            output.mkdir()
            (output / "legacy.bin").write_bytes(b"old")
            recipe = Recipe("fixture", "scenarios/asr.toml", (), (str(output),), ())
            ensure_outputs_ready(recipe, "different-key")
            self.assertEqual((output / "legacy.bin").read_bytes(), b"old")

    def test_missing_absolute_builder_is_reported_before_any_recipe_runs(self) -> None:
        recipe = Recipe(
            "fixture", "scenarios/asr.toml", (), (),
            (Command("export", ("/no/such/python", "export.py"), {}),),
        )
        with self.assertRaisesRegex(FileNotFoundError, "missing executable"):
            validate_command_executables([recipe])

    def test_recipe_key_tracks_the_declared_builder_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenarios").mkdir()
            (root / "scenarios/asr.toml").write_text("[scenario]\nkind = 'asr'\n")
            (root / "toolchain.lock").write_text("PIN=one\n")
            builder = root / "builder.py"
            builder.write_text("print('one')\n")
            config = root / "engine.toml"
            config.write_text("[[model]]\nname = 'fixture'\nscenario = 'scenarios/asr.toml'\n")
            input_file = root / "input.bin"
            input_file.write_bytes(b"input")
            output = root / "output"
            recipe = Recipe("fixture", "scenarios/asr.toml", (str(input_file),), (str(output),),
                            (Command("build", (sys.executable, str(builder)), {}),))
            first = recipe_key(recipe, root, config)
            builder.write_text("print('two')\n")
            self.assertNotEqual(first, recipe_key(recipe, root, config))

    def test_transitive_decode_and_prefill_generators_invalidate_a_recipe_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenarios").mkdir()
            (root / "designs/decode_fused").mkdir(parents=True)
            (root / "designs/resident_forward").mkdir(parents=True)
            (root / "scripts").mkdir()
            (root / "aie_kernels").mkdir()
            (root / "third_party/iron").mkdir(parents=True)
            (root / "scenarios/asr.toml").write_text("[scenario]\nkind = 'asr'\n")
            (root / "toolchain.lock").write_text("PIN=one\n")
            (root / "designs/decode_fused/gen_llm_decode.py").write_text("decode = 1\n")
            prefill = root / "designs/decode_fused/gen_llm_prefill.py"
            prefill.write_text("prefill = 1\n")
            config = root / "engine.toml"
            config.write_text("[[model]]\nname = 'fixture'\nscenario = 'scenarios/asr.toml'\n")
            source = root / "input.bin"
            source.write_bytes(b"input")
            recipe = Recipe("fixture", "scenarios/asr.toml", (str(source),), (str(root / "output"),), ())
            first = recipe_key(recipe, root, config)
            prefill.write_text("prefill = 2\n")
            self.assertNotEqual(first, recipe_key(recipe, root, config))

    def test_input_validation_fails_for_a_later_recipe_before_commands_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            present = root / "present"
            present.mkdir()
            later = root / "missing"
            first = Recipe("fixture-one", "scenarios/asr.toml", (str(present),), (), ())
            second = Recipe("fixture-two", "scenarios/asr.toml", (str(later),), (), ())
            with self.assertRaisesRegex(FileNotFoundError, "fixture-two"):
                validate_recipe_inputs([first, second])

    def test_cache_snapshot_provisions_an_explicit_input_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "hub/models--BAAI--bge-base-en-v1.5/snapshots/revision"
            cache.mkdir(parents=True)
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (cache / name).write_text(name)
            target = root / "inputs/bge-base"
            recipe = Recipe("bge-base", "scenarios/bge-base.toml", (str(target),), (str(root / "out"),), ())
            prior = os.environ.get("HF_HUB_CACHE")
            os.environ["HF_HUB_CACHE"] = str(root / "hub")
            try:
                provision_inputs([recipe], allow_download=False)
            finally:
                if prior is None:
                    os.environ.pop("HF_HUB_CACHE", None)
                else:
                    os.environ["HF_HUB_CACHE"] = prior
            self.assertTrue((target / ".model-input-source.json").is_file())
            validate_recipe_inputs([recipe])

    def test_explicit_local_espcn_source_is_manifested_without_a_download_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/espcn_x3_dyn.onnx"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"onnx")
            target = root / "inputs/espcn/espcn_x3_dyn.onnx"
            recipe = Recipe("espcn", "scenarios/upscale-espcn.toml", (str(target),), (str(root / "out"),), ())
            os.environ["MODEL_INPUT_SOURCE_ESPCN"] = str(source)
            try:
                provision_inputs([recipe], allow_download=False)
            finally:
                os.environ.pop("MODEL_INPUT_SOURCE_ESPCN", None)
            self.assertEqual(target.read_bytes(), b"onnx")
            self.assertTrue((target.parent / ".model-input-source.json").is_file())

    def test_parakeet_requires_all_served_source_and_artifact_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "parakeet"
            source.mkdir()
            (source / "encoder-model.onnx").write_bytes(b"onnx")
            recipe = Recipe("parakeet", "scenarios/asr.toml", (str(source),), (str(root / "artifact"),), ())
            with self.assertRaisesRegex(FileNotFoundError, "encoder-model.onnx.data"):
                validate_recipe_inputs([recipe])
            artifact = root / "artifact"
            (artifact / "encoder").mkdir(parents=True)
            (artifact / "encoder/manifest.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "preprocessor.onnx"):
                validate_recipe_outputs(recipe)

    def test_pyannote_local_checkpoint_path_is_a_file_and_missing_component_fails_preflight(self) -> None:
        from pyannote_local_inputs import local_checkpoint_path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "speaker-diarization-3.1"
            root.mkdir()
            (root / "config.yaml").write_text(
                "pipeline:\n"
                "  params:\n"
                "    segmentation: example/segmentation\n"
                "    embedding: example/embedding\n"
            )
            checkpoint = root / "segmentation/pytorch_model.bin"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"segmentation")
            recipe = Recipe("pyannote-3.1", "scenarios/diarize-pyannote-3.1.toml", (str(root),), (), ())

            self.assertEqual(local_checkpoint_path(root, "segmentation"), checkpoint)
            with self.assertRaisesRegex(FileNotFoundError, "embedding/pytorch_model.bin"):
                validate_recipe_inputs([recipe])

    def test_pyannote_existing_pipeline_materializes_configured_cached_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "inputs/pyannote/speaker-diarization-3.1"
            target.mkdir(parents=True)
            (target / "config.yaml").write_text(
                "pipeline:\n"
                "  params:\n"
                "    segmentation: fixture/segmentation-model\n"
                "    embedding: fixture/embedding-model\n"
            )
            source_manifest_path(Recipe("pyannote-3.1", "fixture", (str(target),), (), ())).write_text(
                json.dumps({"model": "pyannote-3.1", "source": "pipeline-fixture"}) + "\n"
            )
            cache = root / "hub"
            for repo_id, payload in (("fixture/segmentation-model", b"seg"),
                                     ("fixture/embedding-model", b"emb")):
                snapshot = cache / f"models--{repo_id.replace('/', '--')}/snapshots/revision"
                snapshot.mkdir(parents=True)
                (snapshot / "pytorch_model.bin").write_bytes(payload)
            recipe = Recipe("pyannote-3.1", "scenarios/diarize-pyannote-3.1.toml", (str(target),), (), ())
            prior_cache = os.environ.get("HF_HUB_CACHE")
            prior_token = os.environ.pop("HF_TOKEN", None)
            os.environ["HF_HUB_CACHE"] = str(cache)
            try:
                provision_inputs([recipe], allow_download=False)
                first_manifest = source_manifest_path(recipe).read_bytes()
                provision_inputs([recipe], allow_download=False)
            finally:
                if prior_cache is None:
                    os.environ.pop("HF_HUB_CACHE", None)
                else:
                    os.environ["HF_HUB_CACHE"] = prior_cache
                if prior_token is not None:
                    os.environ["HF_TOKEN"] = prior_token

            self.assertEqual((target / "segmentation/pytorch_model.bin").read_bytes(), b"seg")
            self.assertEqual((target / "embedding/pytorch_model.bin").read_bytes(), b"emb")
            record = json.loads(source_manifest_path(recipe).read_text())
            self.assertEqual(
                [dependency["source"] for dependency in record["dependencies"]],
                ["fixture/segmentation-model", "fixture/embedding-model"],
            )
            self.assertEqual(source_manifest_path(recipe).read_bytes(), first_manifest)

    def test_pyannote_dependency_manifest_records_the_full_configured_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "inputs/pyannote/speaker-diarization-3.1"
            target.mkdir(parents=True)
            (target / "config.yaml").write_text(
                "pipeline:\n"
                "  params:\n"
                "    segmentation: fixture/segmentation-model\n"
                "    embedding: fixture/embedding-model\n"
            )
            (target / "segmentation").mkdir()
            (target / "segmentation/pytorch_model.bin").write_bytes(b"seg")
            recipe = Recipe("pyannote-3.1", "scenarios/diarize-pyannote-3.1.toml", (str(target),), (), ())
            source_manifest_path(recipe).write_text(json.dumps({"model": "pyannote-3.1"}) + "\n")
            cache = root / "hub"
            for repo_id, payload in (("fixture/segmentation-model", b"seg"),
                                     ("fixture/embedding-model", b"emb")):
                snapshot = cache / f"models--{repo_id.replace('/', '--')}/snapshots/revision"
                snapshot.mkdir(parents=True)
                (snapshot / "pytorch_model.bin").write_bytes(payload)
            prior_cache = os.environ.get("HF_HUB_CACHE")
            os.environ["HF_HUB_CACHE"] = str(cache)
            try:
                provision_inputs([recipe], allow_download=False)
            finally:
                if prior_cache is None:
                    os.environ.pop("HF_HUB_CACHE", None)
                else:
                    os.environ["HF_HUB_CACHE"] = prior_cache

            record = json.loads(source_manifest_path(recipe).read_text())
            self.assertEqual(
                [dependency["source"] for dependency in record["dependencies"]],
                ["fixture/segmentation-model", "fixture/embedding-model"],
            )

    def test_verified_publish_removes_its_transaction_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / "artifacts/model"
            staged = root / "staging/model"
            live.mkdir(parents=True)
            staged.mkdir(parents=True)
            (live / "value").write_text("old")
            (staged / "value").write_text("new")
            recipe = Recipe("fixture", "scenarios/asr.toml", (), (str(live),), ())
            staged_recipe = Recipe("fixture", "scenarios/asr.toml", (), (str(staged),), ())
            publish_recipe(recipe, staged_recipe, "key", root / "rollback")
            self.assertEqual((live / "value").read_text(), "new")
            self.assertFalse((root / "rollback/key").exists())
            self.assertTrue((live / ".model-recipe.json").is_file())

    def test_failed_publish_verification_retains_recovery_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live, staged = root / "artifacts/model", root / "staging/model"
            live.mkdir(parents=True)
            staged.mkdir(parents=True)
            (live / "value").write_text("old")
            (staged / "value").write_text("new")
            recipe = Recipe("fixture", "scenarios/asr.toml", (), (str(live),), ())
            staged_recipe = Recipe("fixture", "scenarios/asr.toml", (), (str(staged),), ())
            with patch("model_artifact_dispatch.cache_hit", side_effect=[True, False]):
                with self.assertRaisesRegex(RuntimeError, "published receipt did not verify"):
                    publish_recipe(recipe, staged_recipe, "key", root / "rollback")
            self.assertEqual((root / "rollback/key/0/model/value").read_text(), "old")

    def test_publish_does_not_collect_another_transaction_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live, staged = root / "artifacts/model", root / "staging/model"
            live.mkdir(parents=True)
            staged.mkdir(parents=True)
            (live / "value").write_text("old")
            (staged / "value").write_text("new")
            recovery = root / "rollback/other/value"
            recovery.parent.mkdir(parents=True)
            recovery.write_text("recovery")
            recipe = Recipe("fixture", "scenarios/asr.toml", (), (str(live),), ())
            staged_recipe = Recipe("fixture", "scenarios/asr.toml", (), (str(staged),), ())
            publish_recipe(recipe, staged_recipe, "key", root / "rollback")
            self.assertEqual(recovery.read_text(), "recovery")
            self.assertFalse((root / "rollback/key").exists())

    def test_fresh_gemma4_provenance_is_verified_without_rebuilding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = tuple(str(root / name) for name in (
                "resident", "tokenizer", "checkpoint", "rf_stack", "store", "weights", "hf_config", "towers"))
            for output in outputs:
                Path(output).mkdir(parents=True)
            (Path(outputs[0]) / "meta.json").write_text(json.dumps({"iron": {"commit": "a", "dirty": False, "identity": "pinned:a"}}))
            (Path(outputs[1]) / "tokenizer.json").write_text("{}")
            (Path(outputs[2]) / "config.json").write_text("{}")
            (Path(outputs[3]) / "w_head.npy").write_bytes(b"w")
            (Path(outputs[4]) / "manifest.json").write_text("{}")
            (Path(outputs[5]) / "quant.json").write_text("{}")
            (Path(outputs[6]) / "config.json").write_text("{}")
            (Path(outputs[7]) / "tower.npy").write_bytes(b"w")
            for output in outputs[3:]:
                (Path(output) / ".recipe-manifest.json").write_text("{}")
            recipe = Recipe("gemma4-12b", "scenarios/generate-gemma4-12b-resident-256k.toml", (), outputs, ())
            self.assertTrue(verified_existing_gemma4(recipe, "key"))
            self.assertTrue((Path(outputs[0]) / ".model-recipe.json").is_file())

    def test_failed_staged_command_never_replaces_an_unmanaged_live_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scenarios").mkdir()
            (root / "scripts").mkdir()
            (root / "designs/decode_fused").mkdir(parents=True)
            (root / "designs/resident_forward").mkdir(parents=True)
            (root / "aie_kernels").mkdir()
            (root / "third_party/iron").mkdir(parents=True)
            (root / "scenarios/asr.toml").write_text("[scenario]\nkind = 'asr'\n")
            (root / "toolchain.lock").write_text("PIN=one\n")
            config = root / "engine.toml"
            config.write_text("[[model]]\nname = 'fixture'\nscenario = 'scenarios/asr.toml'\n")
            source = root / "source"
            source.mkdir()
            live = root / "artifacts/fixture"
            live.mkdir(parents=True)
            (live / "value").write_text("old")
            builder = root / "scripts/builder.py"
            builder.write_text(
                "import os\nfrom pathlib import Path\nout = Path(os.environ['OUT'])\nout.mkdir(parents=True, exist_ok=True)\n(out / 'value').write_text('partial')\nraise SystemExit(17)\n")
            recipe = Recipe("fixture", "scenarios/asr.toml", (str(source),), (str(live),),
                            (Command("fixture", (sys.executable, str(builder)), {"OUT": str(live)}),))
            with self.assertRaises(subprocess.CalledProcessError):
                build_recipe(recipe, root, config, root / "artifacts", root / "build")
            self.assertEqual((live / "value").read_text(), "old")
            self.assertFalse((live / ".model-recipe.json").exists())

    def test_espcn_recipe_bakes_the_served_checkpoint_from_its_explicit_input(self) -> None:
        espcn = next(entry for entry in self.plan() if entry["model"] == "espcn")
        bake = next(command for command in espcn["commands"] if command["step"] == "bake")
        self.assertIn("checkpoint", bake["argv"])
        self.assertIn("--source", bake["argv"])
        self.assertTrue(any(value.startswith("path:/model-inputs/espcn/") for value in bake["argv"]))
        self.assertIn("/model-artifacts/espcn/espcn.safetensors", bake["argv"])

    def test_whisper_recipe_passes_the_venv_root_to_its_existing_builder(self) -> None:
        whisper = next(entry for entry in self.plan() if entry["model"] == "whisper-turbo")
        decode = next(command for command in whisper["commands"] if command["step"] == "decode")
        self.assertTrue(decode["env"]["VENV_IRON"].endswith("/.venv-iron"))

    def test_install_uses_the_same_configured_model_dispatch(self) -> None:
        install = (REPO / "install.sh").read_text()
        self.assertIn("model_artifact_dispatch.py", install)
        self.assertIn("--model-input-root", install)
        self.assertIn("--report", install)
        self.assertIn("$XDNA_LOGS/model-install-report.json", install)


if __name__ == "__main__":
    unittest.main()
