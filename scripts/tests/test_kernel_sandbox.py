#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "kernel_sandbox.sh"


class KernelSandbox(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="kernel-sandbox-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "toolchain.lock").write_text("MLIR_AIE_FORK_COMMIT=aaa\n")
        self.source = self.root / "cache/instances/pin/src"
        (self.source / "programming_examples/ml/layernorm").mkdir(parents=True)
        (self.repo / "mlir-aie/programming_examples/ml/layernorm").mkdir(parents=True)
        self.env = dict(os.environ, REPO=str(self.repo), KERNEL_SANDBOX_GC="1",
                        KERNEL_SANDBOX_KEEP_DAYS="36500")

    def run_sandbox(self, bd, source=None):
        command = 'source "$1"; if [ -n "$3" ]; then mlir_aie_root="$3"; fi; ensure_fresh_sandbox "$2"'
        return subprocess.run(["bash", "-c", command, "sandbox-test", str(SCRIPT),
                               str(bd), str(source) if source is not None else ""],
                              cwd=self.repo, env=self.env, capture_output=True, text=True)

    def assert_refused(self, bd, source=None):
        result = self.run_sandbox(bd, source)
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("refuse:", result.stderr)
        self.assertFalse((bd / ".toolchain-stamp").exists())

    def test_declared_instance_root_creates_stamp_and_preserves_same_pin(self):
        bd = self.source / "programming_examples/ml/layernorm/build"
        result = self.run_sandbox(bd, self.source)
        self.assertEqual(result.returncode, 0, result.stderr)
        stamp = (bd / ".toolchain-stamp").read_text()
        (bd / "marker.o").touch()
        self.assertEqual(self.run_sandbox(bd, self.source).returncode, 0)
        self.assertTrue((bd / "marker.o").exists())
        self.assertEqual((bd / ".toolchain-stamp").read_text(), stamp)

    def test_default_checkout_root_accepts_relative_build_path(self):
        bd = Path("mlir-aie/programming_examples/ml/layernorm/build")
        result = self.run_sandbox(bd)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.repo / bd / ".toolchain-stamp").is_file())

    def test_changed_pin_retires_fixture_without_deleting_it(self):
        bd = self.repo / "mlir-aie/programming_examples/ml/layernorm/build"
        self.assertEqual(self.run_sandbox(bd).returncode, 0)
        (bd / "marker.o").touch()
        (self.repo / "toolchain.lock").write_text("MLIR_AIE_FORK_COMMIT=bbb\n")
        self.assertEqual(self.run_sandbox(bd).returncode, 0)
        retired = list(bd.parent.glob("build.stale-*"))
        self.assertEqual(len(retired), 1)
        self.assertTrue((retired[0] / "marker.o").is_file())
        self.assertFalse((bd / "marker.o").exists())

    def test_arbitrary_matching_text_is_not_authority(self):
        bd = self.root / "unrelated/mlir-aie/programming_examples/ml/layernorm/build"
        self.assert_refused(bd)

    def test_outside_declared_root_is_refused(self):
        bd = self.repo / "mlir-aie/programming_examples/ml/layernorm/build"
        self.assert_refused(bd, self.source)

    def test_unknown_root_is_refused(self):
        bd = self.source / "programming_examples/ml/layernorm/build"
        self.assert_refused(bd)
        self.assert_refused(bd, self.root / "missing")

    def test_parent_traversal_is_refused(self):
        bd = self.source / "programming_examples/../../outside/build"
        self.assert_refused(bd, self.source)

    def test_build_symlink_escape_is_refused_without_touching_target(self):
        outside = self.root / "outside"
        outside.mkdir()
        marker = outside / "marker.o"
        marker.touch()
        bd = self.repo / "mlir-aie/programming_examples/ml/layernorm/build"
        bd.symlink_to(outside, target_is_directory=True)
        self.assert_refused(bd)
        self.assertTrue(marker.is_file())
        self.assertTrue(bd.is_symlink())

    def test_parent_symlink_escape_is_refused(self):
        outside = self.root / "outside"
        outside.mkdir()
        link = self.repo / "mlir-aie/programming_examples/escape"
        link.symlink_to(outside, target_is_directory=True)
        self.assert_refused(link / "build")

    def test_non_build_directory_is_refused(self):
        bd = self.repo / "mlir-aie/programming_examples/ml/layernorm"
        self.assert_refused(bd)

    def test_declared_source_symlink_resolves_to_authority(self):
        link = self.root / "source-alias"
        link.symlink_to(self.source, target_is_directory=True)
        bd = link / "programming_examples/ml/layernorm/build"
        result = self.run_sandbox(bd, link)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((bd / ".toolchain-stamp").is_file())

    def test_programming_examples_symlink_escape_is_refused(self):
        source = self.root / "other-source"
        source.mkdir()
        (source / "programming_examples").symlink_to(
            self.source / "programming_examples", target_is_directory=True)
        self.assert_refused(source / "programming_examples/ml/layernorm/build", source)

    def test_root_without_examples_is_refused(self):
        source = self.root / "empty-source"
        source.mkdir()
        self.assert_refused(source / "programming_examples/ml/layernorm/build", source)

    def test_default_checkout_can_create_missing_build_parents(self):
        repo = self.root / "fresh-repo"
        repo.mkdir()
        (repo / "toolchain.lock").write_text("MLIR_AIE_FORK_COMMIT=aaa\n")
        self.env["REPO"] = str(repo)
        bd = repo / "mlir-aie/programming_examples/ml/layernorm/build"
        result = self.run_sandbox(bd)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((bd / ".toolchain-stamp").is_file())


if __name__ == "__main__":
    unittest.main()
