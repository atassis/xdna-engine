import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_artifact_dispatch as dispatch
from lib.collect_served_artifacts import scenario_artifact_paths

REPO = Path(__file__).resolve().parents[1]


class AsrRecipeClosureTests(unittest.TestCase):
    def test_parakeet_stages_its_implicit_relpos_dependency(self):
        scenario = {"scenario": {"kind": "asr", "name": "parakeet-tdt-0.6b-v3"},
                    "artifacts": {"weights": "artifacts/parakeet"}}
        self.assertEqual(set(scenario_artifact_paths(scenario)),
                         {"artifacts/parakeet", "artifacts/relpos"})
        scenario["scenario"]["name"] = "whisper-large-v3-turbo"
        self.assertNotIn("artifacts/relpos", scenario_artifact_paths(scenario))

    def test_export_stage_does_not_require_later_relpos_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = dispatch.recipe_for("parakeet", "scenarios/asr.toml", REPO,
                                         root / "artifacts", root / "inputs", root / "build")
            primary = Path(recipe.outputs[0])
            for name in ("preprocessor.onnx", "decoder_joint.onnx", "vocab.txt", "encoder/manifest.json"):
                target = primary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fixture")
            dispatch.validate_stage_outputs(recipe, "export")
            with self.assertRaises(FileNotFoundError):
                dispatch.validate_recipe_outputs(recipe)

    def test_parakeet_recipe_builds_and_checks_every_relpos_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = dispatch.recipe_for("parakeet", "scenarios/asr.toml", REPO,
                                         root / "artifacts", root / "inputs", root / "build")
            relpos = root / "artifacts/relpos"
            self.assertIn(str(relpos), recipe.outputs)
            command = next(item for item in recipe.commands if item.step == "relpos")
            self.assertEqual(command.env["OUT_ROOT"], str(relpos))
            self.assertEqual(command.env["BUCKETS"], "100:25:bucket_100 152:38:bucket_152 172:43:single")
            self.assertEqual((command.env["HEADS"], command.env["ROWS"], command.env["TSKIP"]), ("8", "1", "0"))
            self.assertTrue(command.stage.toolchain)
            for bucket in ("bucket_100", "bucket_152", "single"):
                directory = relpos / bucket
                directory.mkdir(parents=True)
                for name in ("final.xclbin", "insts.bin", ".toolchain-stamp"):
                    (directory / name).write_bytes(b"fixture")
            dispatch.require_members(recipe, relpos, output=True)
            missing = relpos / "bucket_100/insts.bin"
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                dispatch.require_members(recipe, relpos, output=True)


if __name__ == "__main__":
    unittest.main()
