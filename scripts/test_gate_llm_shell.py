"""Exercise prefill metadata resolution without building or using the device."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


class MetadataDiagnosticsTests(unittest.TestCase):
    def run_metadata(self, cargo_body):
        source = Path(__file__).with_name("gate_llm.sh").read_text()
        start = source.index('  tgt="$(cd "$REPO/rust" && cargo metadata')
        end = source.index('  [ -n "$tgt" ]', start)
        assignment = source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rust").mkdir()
            cargo = root / "cargo"
            cargo.write_text("#!/bin/sh\n" + cargo_body + "\n")
            cargo.chmod(0o755)
            return subprocess.run(
                ["bash", "-c", 'set -o pipefail\n' + assignment
                 + '\nstatus=$?\nprintf "%s" "$tgt"\nexit "$status"\n'],
                env=os.environ | {"REPO": str(root), "PY": sys.executable,
                                  "PATH": str(root) + os.pathsep + os.environ["PATH"]},
                capture_output=True, text=True, timeout=10,
            )

    def test_success_keeps_target_directory(self):
        target = "/tmp/build output"
        result = self.run_metadata("printf '%s\\n' " + shlex.quote(
            json.dumps({"target_directory": target})))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, target)
        self.assertEqual(result.stderr, "")

    def test_cargo_failure_keeps_its_diagnostic(self):
        result = self.run_metadata("echo 'metadata fixture failure' >&2; exit 42")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("metadata fixture failure", result.stderr)

    def test_invalid_metadata_keeps_parser_diagnostic(self):
        result = self.run_metadata("echo 'invalid metadata'")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("JSONDecodeError", result.stderr)


if __name__ == "__main__":
    unittest.main()
