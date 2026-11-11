import copy
import hashlib
import json
import shutil
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_artifact_dispatch as dispatch
from lib.collect_served_artifacts import scenario_artifact_paths

REPO = Path(__file__).resolve().parents[1]
MODEL = "qwen3-embedding-4b"
SCENARIO = f"scenarios/{MODEL}.toml"


class QwenEmbeddingRecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.recipe = dispatch.recipe_for(MODEL, SCENARIO, REPO,
                                          self.root / "artifacts", self.root / "inputs",
                                          self.root / "build")
        self.commands = {item.step: item for item in self.recipe.commands}

    def fixture(self):
        config = {"num_hidden_layers": 36, "hidden_size": 2560, "intermediate_size": 9728,
                  "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
                  "vocab_size": 151665}
        files = {"config.json": json.dumps(config).encode(), "tokenizer.json": b"tokenizer fixture",
                 "tokenizer_config.json": b"{}"}
        pin = {"repo": "Qwen/Qwen3-Embedding-4B", "revision": "a" * 40, "dimension": 2560,
               "files": {name: {"size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                         for name, value in files.items()}}
        source = {"model": pin["repo"], "revision": pin["revision"],
                  "files": {name: entry["sha256"] for name, entry in pin["files"].items()}}
        decode, prefill, tokenizer = map(Path, self.recipe.outputs)
        for directory in (decode, prefill, tokenizer, Path(self.recipe.inputs[0])):
            directory.mkdir(parents=True, exist_ok=True)
        for name, value in files.items():
            (tokenizer / name).write_bytes(value)
            (Path(self.recipe.inputs[0]) / name).write_bytes(value)
        dimensions = dict(layers=36, d_model=2560, ffn=9728, q_heads=32, kv_heads=8,
                          head_dim=128, vocab=151665, S=512, kv_block=64)
        self.decode_meta = {"spec": MODEL, "elf": "decode.elf", "dims": dimensions,
                            "embedding_source": source, "weights": ["W", "W_head"],
                            "layout": {"W": {"len": 2}, "W_head": {"len": 2}},
                            "weight_quant": {"plan": {site: "bf16" for site in
                                             ("mlp", "attn_o", "qkv", "head", "kv")}}}
        self.prefill_meta = {"spec": MODEL, "elf": "prefill.elf", "dims": dict(dimensions, M=256),
                             "embedding_source": source, "weights": ["W", "x"],
                             "layout": {"W": {"len": 2}, "x": {"len": 2}},
                             "weights_from": str(decode / "buffers"),
                             "decode_artifact": {"meta": str(decode / "meta.json")},
                             "causal": True, "causal_mode": "rows", "arena_shared": True}
        for directory, meta in ((decode, self.decode_meta), (prefill, self.prefill_meta)):
            (directory / "buffers").mkdir()
            (directory / "meta.json").write_text(json.dumps(meta))
            (directory / meta["elf"]).write_bytes(b"ELF fixture")
        for directory, name in ((decode, "W"), (decode, "W_head"), (prefill, "x")):
            (directory / "buffers" / f"{name}.bin").write_bytes(b"ab")
        return pin

    def test_validated_pinned_checkpoint_needs_no_input_rewrite_or_download(self):
        pin = self.fixture()
        checkpoint = Path(self.recipe.inputs[0])
        before = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
        with patch("qwen_embedding.model_pin", return_value=pin), \
                patch("model_artifact_dispatch.subprocess.run", side_effect=AssertionError("no download")):
            dispatch.provision_inputs([self.recipe], allow_download=True)
            self.assertEqual({path.name: path.read_bytes() for path in checkpoint.iterdir()}, before)
            (checkpoint / "config.json").write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                dispatch.provision_inputs([self.recipe], allow_download=True)

    def test_artifact_closure_rejects_missing_shared_weight_and_wrong_owner(self):
        pin = self.fixture()
        decode, prefill, _ = map(Path, self.recipe.outputs)
        with patch("qwen_embedding.model_pin", return_value=pin):
            dispatch.validate_recipe_outputs(self.recipe)
            weight = decode / "buffers/W.bin"
            weight.unlink()
            with self.assertRaises(FileNotFoundError):
                dispatch.validate_recipe_outputs(self.recipe)
            weight.write_bytes(b"ab")
            changed = dict(self.prefill_meta, weights_from=str(self.root / "unrelated"))
            (prefill / "meta.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                dispatch.validate_recipe_outputs(self.recipe)

    def test_artifact_closure_rejects_source_depth_precision_and_tokenizer_drift(self):
        pin = self.fixture()
        decode, prefill, tokenizer = map(Path, self.recipe.outputs)
        changes = []
        for key, value in (("revision", "b" * 40), ("files", {})):
            changed = copy.deepcopy(self.decode_meta)
            changed["embedding_source"][key] = value
            changes.append(changed)
        changed = copy.deepcopy(self.decode_meta)
        changed["dims"]["layers"] = 1
        changes.append(changed)
        changed = copy.deepcopy(self.decode_meta)
        changed["weight_quant"]["plan"]["mlp"] = "int4/g64"
        changes.append(changed)
        with patch("qwen_embedding.model_pin", return_value=pin):
            for changed in changes:
                with self.subTest(change=changed):
                    (decode / "meta.json").write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        dispatch.validate_recipe_outputs(self.recipe)
            (decode / "meta.json").write_text(json.dumps(self.decode_meta))
            changed = copy.deepcopy(self.prefill_meta)
            changed["embedding_source"]["revision"] = "b" * 40
            (prefill / "meta.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                dispatch.validate_recipe_outputs(self.recipe)
            (prefill / "meta.json").write_text(json.dumps(self.prefill_meta))
            (tokenizer / "tokenizer.json").write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                dispatch.validate_recipe_outputs(self.recipe)

    def test_prefill_is_causal_shared_m256_and_precision_declarations_agree(self):
        pin = self.fixture()
        decode, prefill, _ = map(Path, self.recipe.outputs)
        with patch("qwen_embedding.model_pin", return_value=pin):
            changed = copy.deepcopy(self.decode_meta)
            changed["weight_quant"]["mlp_dtype"] = "int4"
            (decode / "meta.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                dispatch.validate_recipe_outputs(self.recipe)
            (decode / "meta.json").write_text(json.dumps(self.decode_meta))
            for field, value in (("causal", False), ("causal_mode", "none"), ("arena_shared", False)):
                changed = dict(self.prefill_meta, **{field: value})
                (prefill / "meta.json").write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    dispatch.validate_recipe_outputs(self.recipe)
            for field, value in (("M", 32), ("kv_block", 512)):
                changed = copy.deepcopy(self.prefill_meta)
                changed["dims"][field] = value
                (prefill / "meta.json").write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    dispatch.validate_recipe_outputs(self.recipe)

    def test_staged_pair_publishes_without_transient_dependencies_and_cache_rejects_drift(self):
        pin = self.fixture()
        workspace = self.root / "transaction"
        staged = dispatch.rebase_recipe(self.recipe, self.root / "artifacts", workspace / "artifacts",
                                        self.root / "build", workspace / "build")
        for source, target in zip(self.recipe.outputs, staged.outputs):
            shutil.copytree(source, target)
        decode, prefill, _ = map(Path, staged.outputs)
        changed = dict(self.prefill_meta, weights_from=str(decode / "buffers"),
                       decode_artifact={"meta": str(decode / "meta.json")})
        (prefill / "meta.json").write_text(json.dumps(changed))
        (decode / "buffers/W.bin").write_bytes(b"cd")
        with patch("qwen_embedding.model_pin", return_value=pin), \
                patch("model_artifact_dispatch.LOCK_ROOT", self.root / "locks"):
            for step in ("decode", "prefill", "tokenizer"):
                dispatch.validate_stage_outputs(staged, step)
            dispatch.validate_recipe_outputs(staged)
            dispatch.relocate_recipe_metadata(self.recipe, staged)
            dispatch.validate_recipe_outputs(staged, references=self.recipe)
            dispatch.no_staging_references(staged, workspace)
            dispatch.publish_recipe(self.recipe, staged, "fixture-key", self.root / "rollback")
            self.assertTrue(dispatch.cache_hit(self.recipe, "fixture-key"))
            self.assertEqual((Path(self.recipe.outputs[0]) / "buffers/W.bin").read_bytes(), b"cd")
            changed["weights_from"] = str(self.root / "unrelated")
            (Path(self.recipe.outputs[1]) / "meta.json").write_text(json.dumps(changed))
            self.assertFalse(dispatch.cache_hit(self.recipe, "fixture-key"))

    def test_recipe_selects_full_depth_bf16_and_the_accepted_schedule(self):
        self.assertEqual(tuple(self.commands), ("weights", "decode", "prefill", "tokenizer"))
        weights = self.commands["weights"]
        self.assertEqual(weights.argv[weights.argv.index("--spec") + 1], MODEL)
        self.assertEqual(weights.argv[weights.argv.index("--checkpoint-dir") + 1], self.recipe.inputs[0])
        self.assertEqual(weights.argv[weights.argv.index("--quant") + 1], "bf16")
        decode = self.commands["decode"]
        self.assertEqual(decode.argv[2:4], (MODEL, "36"))
        self.assertEqual(decode.env["GEN_EXTRA"], "--max-seq 512")
        self.assertEqual(decode.env["KV_BLOCK_T"], "64")
        self.assertEqual(decode.env["PRECISION"], "bf16")
        for key in ("FUSE_QKV_DP", "FUSE_DECODE_LAYER", "FUSE_MLP_DP"):
            self.assertEqual(decode.env[key], "1")
        prefill = self.commands["prefill"]
        self.assertEqual(prefill.argv[2:5], ("36", "256", "512"))
        self.assertEqual(prefill.env["SPEC"], MODEL)
        self.assertEqual(prefill.env["DECODE_META"], self.recipe.outputs[0] + "/meta.json")
        self.assertEqual(prefill.env["KV_BLOCK_T"], "64")
        self.assertEqual((prefill.env["PREFILL_BFP16"], prefill.env["PREFILL_ACC"]), ("1", "0"))
        tiles = json.loads(prefill.env["GEMM_TILES_OVERRIDE"])
        self.assertEqual(tiles["m256_k4096_n2560_bf162bf16_bfp1_acc0"]["tile_n"], 16)
        self.assertEqual(tiles["m256_k2560_n9728_bf162bf16_bfp1_acc0"]["tile_n"], 64)

    def test_complete_pinned_inputs_and_transient_weights_are_bound_to_stages(self):
        self.assertEqual(len(self.recipe.outputs), 3)
        weights = self.commands["weights"]
        dump = weights.argv[weights.argv.index("--out") + 1]
        self.assertTrue(Path(dump).is_relative_to(self.root / "build"))
        self.assertNotIn(dump, self.recipe.outputs)
        self.assertEqual(weights.stage.inputs[0].root, self.recipe.inputs[0])
        self.assertEqual(weights.stage.inputs[0].patterns, ())
        self.assertIn("scripts/qwen_embedding.py", weights.stage.sources)
        self.assertIn("scripts/qwen_embedding_pins.json", weights.stage.sources)
        self.assertEqual(weights.stage.sdk_python, weights.argv[0])
        self.assertTrue({"torch", "numpy", "safetensors"}.issubset(weights.stage.sdk_packages))
        for step in ("decode", "prefill"):
            command = self.commands[step]
            self.assertTrue(command.stage.toolchain)
            self.assertEqual(command.env["IRON"], str(REPO / "third_party/iron"))
            self.assertIn("designs/decode_fused", command.stage.sources)
            self.assertIn("designs/resident_forward/scratchpad_params.py", command.stage.sources)

    def test_published_outputs_match_the_opt_in_scenario(self):
        config = tomllib.loads((REPO / SCENARIO).read_text())
        self.assertEqual(set(scenario_artifact_paths(config)), {
            "artifacts/" + str(Path(raw).relative_to(self.root / "artifacts"))
            for raw in self.recipe.outputs
        })
        self.assertEqual(self.recipe.inputs, (str(self.root / "inputs" / MODEL),))
        with self.assertRaises(ValueError):
            dispatch.recipe_for(MODEL, "scenarios/generate-qwen3-0.6b.toml", REPO,
                                self.root / "artifacts", self.root / "inputs", self.root / "build")


if __name__ == "__main__":
    unittest.main()
