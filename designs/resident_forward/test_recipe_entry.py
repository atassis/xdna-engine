import ast
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


HERE = Path(__file__).parent
REPO = HERE.parents[1]
RECIPE = HERE / "recipes/rf48C.sh"
LADDER = HERE / "fwd_ladder.sh"


def rf48c_flags() -> list[str]:
    words = shlex.split(RECIPE.read_text().replace("\\\n", " "), comments=True)
    return words[words.index("rf48C") + 1].split()


def imports_name(path: Path, module: str, name: str) -> bool:
    tree = ast.parse(path.read_text())
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == module
        and any(alias.name == name for alias in node.names)
        for node in ast.walk(tree)
    )


class RecipeEntryTests(unittest.TestCase):
    def graphgen(self, flags: list[str]) -> subprocess.CompletedProcess[str]:
        probe = """
import sys
sys.path.insert(0, sys.argv[1])
sys.modules["pyxrt"] = None
import rlayer_design
print(len(rlayer_design.build_text(sys.argv[2:])))
"""
        env = os.environ.copy()
        env["RF_FAST"] = "1"
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(probe), str(HERE), *flags],
            cwd=HERE,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_rf48c_sring_graphgen_does_not_import_pyxrt(self) -> None:
        flags = rf48c_flags()
        flags.insert(flags.index("1"), "emit=p1")
        result = self.graphgen(flags)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreater(int(result.stdout.strip()), 0)

    def test_sring_k059_rejects_a_ring_smaller_than_the_recipe_window(self) -> None:
        flags = rf48c_flags()
        flags[flags.index("sring=1280")] = "sring=1216"
        result = self.graphgen(flags)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AssertionError: (20, 1216", result.stderr)

    def test_runtime_and_builder_use_the_shared_window_constant(self) -> None:
        kvring = HERE / "kvring.py"
        self.assertIn("DEFAULT_WINDOW_ROWS = 1024", kvring.read_text())
        self.assertTrue(imports_name(HERE / "rld_run.py", "kvring", "DEFAULT_WINDOW_ROWS"))
        self.assertTrue(imports_name(HERE / "rlayer_design.py", "kvring", "DEFAULT_WINDOW_ROWS"))

    def test_global_weight_preparation_does_not_import_pyxrt(self) -> None:
        probe = """
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
sys.modules["pyxrt"] = None
import grun
with tempfile.TemporaryDirectory() as directory:
    grun.WCACHE = str(Path(directory) / "weights.npy")
    expected = np.arange(16, dtype=np.uint8)
    np.save(grun.WCACHE, expected)
    assert np.array_equal(grun.weights(), expected)
assert grun.o_k_index_device().dtype == np.int64
print("host preparation does not require pyxrt")
"""
        result = subprocess.run([sys.executable, "-c", textwrap.dedent(probe), str(HERE)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("host preparation does not require pyxrt", result.stdout)

    def make_ladder_fixture(self, failed_part: str | None = None, slow_part: str | None = None) -> tuple[Path, Path, dict[str, str]]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        rf = root / "designs/resident_forward"
        rf.mkdir(parents=True)
        shutil.copy(LADDER, rf / "fwd_ladder.sh")
        (rf / "env.sh").write_text(f"set -euo pipefail\nPY={sys.executable!s}\nexport PY\n")
        (rf / "rf_build.py").write_text(textwrap.dedent(f"""
            import pathlib
            import sys
            part = sys.argv[1]
            if part == {failed_part!r}:
                print("fixture failure", file=sys.stderr)
                raise SystemExit(17)
            if part == {slow_part!r}:
                import time
                time.sleep(1)
            pathlib.Path("finished").write_text(part)
        """))
        (rf / "fwd_pack.py").write_text(textwrap.dedent("""
            import os
            import pathlib
            import sys
            out = pathlib.Path(os.environ["RF_BUILD"], sys.argv[1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "packed").write_text(" ".join(sys.argv[2:]))
        """))
        bins = root / "bin"
        bins.mkdir()
        launcher = bins / "systemd-run"
        launcher.write_text("#!/usr/bin/env bash\nwhile [ \"$1\" != nice ]; do shift; done\nexec \"$@\"\n")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
        build = root / "build"
        env = os.environ.copy()
        env["PATH"] = f"{bins}:{env['PATH']}"
        env["RF_BUILD"] = str(build)
        env["RF_LADDER_JOBS"] = "2"
        return root, build, env

    def run_ladder(self, failed_part: str | None = None, slow_part: str | None = None) -> tuple[subprocess.CompletedProcess[str], Path]:
        root, build, env = self.make_ladder_fixture(failed_part, slow_part)
        result = subprocess.run(
            [root / "designs/resident_forward/fwd_ladder.sh", "fixture", "m", "one", "two"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        return result, build

    def test_ladder_packs_after_successful_children(self) -> None:
        result, build = self.run_ladder()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((build / "fixture" / "packed").read_text(), "fixture_p0 fixture_p1")

    def test_ladder_reports_failed_part_log_and_skips_pack(self) -> None:
        result, build = self.run_ladder("fixture_p1", "fixture_p0")
        log = build / "fixture_p1" / "build.log"
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertIn(f"fixture_p1 failed (rc=17), log: {log}", result.stderr)
        self.assertTrue(log.is_file())
        self.assertIn("fixture failure", log.read_text())
        self.assertTrue((build / "fixture_p0" / "finished").is_file())
        self.assertFalse((build / "fixture" / "packed").exists())

    def test_stack_recipe_passes_its_declared_weight_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rf = root / "designs/resident_forward"
            (rf / "recipes").mkdir(parents=True)
            shutil.copy(HERE / "recipes/gemma4_data.sh", rf / "recipes/gemma4_data.sh")
            artifacts = root / "artifacts"
            weights = artifacts / "gemma4-12b/weights_int4g32sbf16_planar_qat_rg"
            weights.mkdir(parents=True)
            (artifacts / "gemma4-12b/store").mkdir()
            (rf / "stack_prep.py").write_text(textwrap.dedent(f"""
                import json, os
                from pathlib import Path
                assert os.environ.get("RF_WDIR") == {str(weights)!r}, os.environ.get("RF_WDIR")
                out = Path(os.environ["RF_STACK_OUT"])
                (out / "input.json").write_text(json.dumps({{"weights": os.environ["RF_WDIR"]}}))
            """))
            bins = root / "bin"
            bins.mkdir()
            df = bins / "df"
            df.write_text("#!/usr/bin/env bash\nprintf 'Avail\\n100G\\n'\n")
            df.chmod(df.stat().st_mode | stat.S_IXUSR)
            env = os.environ | {"PATH": f"{bins}:{os.environ['PATH']}",
                                "VENV_IRON": str(Path(sys.executable).parent.parent),
                                "RF_WDIR": "/incorrect-ambient-weights"}
            result = subprocess.run(["bash", str(rf / "recipes/gemma4_data.sh"),
                                     "--out", str(artifacts), "--stages", "rf_stack"],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            recorded = json.loads((artifacts / "gemma4-12b/rf_stack/input.json").read_text())
            self.assertEqual(recorded["weights"], str(weights))


if __name__ == "__main__":
    unittest.main()
