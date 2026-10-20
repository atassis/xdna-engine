#!/usr/bin/env python3
"""Disposable publisher fixtures: no compiler, device, or installed-root access."""
import json
import os
import pathlib
import subprocess
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("publish_kernels.sh")
STEMS = ["1024x800x3072_32x32x32_8c_modalsilunat",
         "1024x800x768_32x32x32_8c_modalidnat",
         "1024x800x3072_32x32x32_8c_modalidnat"]


class PublishKernelsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.source = self.root / "mlir-aie"
        self.build = self.source / "programming_examples/basic/matrix_multiplication/whole_array/build"
        self.build.mkdir(parents=True)
        (self.build / ".toolchain-stamp").write_text("fixture-pin")
        self.dest = self.root / "stage/kernels"
        self.engine = self.root / "engine"
        self.engine.mkdir()
        self.config = self.root / "engine.toml"
        self.config.write_text('[[model]]\nname="fixture"\nscenario="cls.toml"\n')
        (self.engine / "cls.toml").write_text('[kernels.whole_array]\nrequired=' + json.dumps(STEMS) + '\n')
        for stem in STEMS + ["512x800x768_64x32x96_8c_modalid"]:
            (self.build / f"final_{stem}.xclbin").write_bytes(b"fake xclbin")
            (self.build / f"insts_{stem}.txt").write_bytes(b"00000001\n")
            if stem in STEMS:
                (self.build / f"contract_{stem}.json").write_bytes(b'{"original":"build receipt bytes"}\n')
        # The executable is a seam spy, NOT a substitute validation implementation.
        # Missing selected source members are checked here only to test publisher fail-closed
        # delegation before any destination writes. Rust tests own receipt semantics.
        self.fake = self.root / "manifest"
        self.calls = self.root / "calls.jsonl"
        self.fake.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['FIXTURE_CALLS'], 'a') as f: f.write(json.dumps(args) + '\\n')
if '--check-native-contracts-only' not in args and os.environ.get('FIXTURE_REJECT_DEST'):
    sys.exit(1)
if '--check-native-contracts-only' in args:
    source = next(pathlib.Path(a) for a in args if a.endswith('/whole_array/build'))
    engine = pathlib.Path(args[args.index('--engine-root')+1])
    import tomllib
    stems = tomllib.loads((engine/'cls.toml').read_text())['kernels']['whole_array']['required']
    for stem in stems:
        if not (source/f'final_{stem}.xclbin').is_file() or not any((source/f'insts_{stem}.{ext}').is_file() for ext in ['txt','bin']) or not (source/f'contract_{stem}.json').is_file():
            sys.exit(1)
''')
        self.fake.chmod(0o755)

    def publish(self, **extra_env):
        env = dict(os.environ, GEN_KERNEL_MANIFEST_BIN=str(self.fake), FIXTURE_CALLS=str(self.calls),
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        env.update(extra_env)
        return subprocess.run(["bash", str(SCRIPT), str(self.dest), str(self.source), str(self.config), str(self.engine)],
                              env=env, capture_output=True, text=True, timeout=15)

    def test_copies_original_contracts_and_forwards_selected_roots(self):
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        for stem in STEMS:
            copied = self.dest / "whole_array" / f"contract_{stem}.json"
            self.assertTrue(copied.is_file(), "Native build receipt must be published")
            self.assertEqual(copied.read_bytes(), (self.build / copied.name).read_bytes())
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual(len(calls), 2, "validate source before writing, then destination before manifest")
        for args in calls:
            self.assertEqual(args[args.index('--engine-root') + 1], str(self.engine))
            self.assertEqual(args[args.index('--config') + 1], str(self.config))
        self.assertIn('--check-native-contracts-only', calls[0])
        self.assertNotIn('--check-native-contracts-only', calls[1])

    def test_missing_selected_source_contract_or_pair_fails_before_writes(self):
        for name in [f"contract_{STEMS[0]}.json", f"final_{STEMS[0]}.xclbin", f"insts_{STEMS[0]}.txt"]:
            with self.subTest(name=name):
                path = self.build / name
                contents = path.read_bytes()
                path.unlink()
                result = self.publish()
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertFalse(self.dest.exists(), "invalid source must fail before destination writes")
                path.write_bytes(contents)

    def test_destination_validator_failure_is_fatal(self):
        result = self.publish(FIXTURE_REJECT_DEST="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kernels-manifest failed", result.stderr)

    def test_stale_destination_receipt_cannot_fill_missing_source_receipt(self):
        receipt = f"contract_{STEMS[0]}.json"
        destination = self.dest / "whole_array"
        destination.mkdir(parents=True)
        (self.dest / ".toolchain-stamp").write_text("fixture-pin")
        (destination / receipt).write_bytes(b"stale destination receipt")
        (self.build / receipt).unlink()
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((destination / receipt).read_bytes(), b"stale destination receipt")
        self.assertEqual(list(destination.iterdir()), [destination / receipt])

    def test_unselected_experiment_needs_no_receipt(self):
        stem = "1536x800x768_32x32x32_8c_modalidnat"
        (self.build / f"final_{stem}.xclbin").write_bytes(b"experiment")
        (self.build / f"insts_{stem}.txt").write_bytes(b"instructions")
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mixed_stamp_fails_before_writes(self):
        other = self.source / "programming_examples/ml/layernorm/build"
        other.mkdir(parents=True)
        (other / ".toolchain-stamp").write_text("other-pin")
        result = self.publish()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()
