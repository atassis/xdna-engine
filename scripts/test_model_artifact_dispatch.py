import json
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


class ModelArtifactDispatchPlanTests(unittest.TestCase):
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

    def test_publish_keeps_the_old_tree_as_rollback_and_writes_verified_receipt(self) -> None:
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
            backups = list((root / "rollback").rglob("value"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), "old")
            self.assertTrue((live / ".model-recipe.json").is_file())

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
