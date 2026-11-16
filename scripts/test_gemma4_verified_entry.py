#!/usr/bin/env python3
"""CLI tests for the Gemma4 reviewed-proof adapter."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import types
import unittest
from unittest import mock

import gemma4_verified_entry as entry
import test_verified_gemma4_data as g4_tests


SCRIPT = Path(__file__).with_name('gemma4_verified_entry.py')


class Gemma4VerifiedEntryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = g4_tests.Gemma4ProofProducerTests('run')
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.output = self.fixture.work / 'nested' / 'proof'

    def run_cli(self, *args, cwd=None):
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(SCRIPT.parent)
        return subprocess.run([sys.executable, '-B', str(SCRIPT), *map(str, args)],
                              cwd=cwd, env=environment, text=True, capture_output=True)

    def proof_args(self, output=None):
        return ('proof', '--source', self.fixture.root, '--accepted-data-root', self.fixture.owner,
                '--witness', self.fixture.witness_path, '--expected-witness-sha256',
                self.fixture.expected, '--out', output or self.output)

    def test_proof_materializes_exact_reviewed_bytes_in_fresh_nested_output(self):
        result = self.run_cli(*self.proof_args())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            'model': 'gemma4-12b', 'variant': 'verified-data',
            'witness_sha256': self.fixture.expected,
            'payload_digest': self.fixture.witness()['payload_digest'],
            'members': 68, 'filecount': 68,
            'proofpath': str(self.output / '.verified-data-witness.json'),
        })
        self.assertEqual((self.output / '.verified-data-witness.json').read_bytes(), self.fixture.raw)

    def test_bad_pin_and_changed_source_fail_before_any_output(self):
        bad_output = self.fixture.work / 'bad-pin' / 'proof'
        wrong_pin = self.fixture.expected[:-1] + ('0' if self.fixture.expected[-1] != '0' else '1')
        result = self.run_cli(*self.proof_args(bad_output)[:-3], wrong_pin, '--out', bad_output)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(bad_output.exists())
        (self.fixture.root / 'rf_stack' / 'w0.npy').write_bytes(b'changed')
        result = self.run_cli(*self.proof_args())
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_proof_duplicate_unknown_and_missing_refuse_without_output(self):
        cases = (
            ('duplicate', self.proof_args() + ('--source=' + str(self.fixture.root),)),
            ('unknown', self.proof_args() + ('--unknown', 'value')),
            ('missing', self.proof_args()[:-2]),
            ('wrong-subcommand', ('not-a-command',)),
        )
        for label, arguments in cases:
            with self.subTest(label=label):
                result = self.run_cli(*arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.output.exists())

    def package_args(self):
        return ('package', '--source', self.fixture.root, '--accepted-data-root', self.fixture.owner,
                '--proof-root', self.fixture.work / 'proof', '--expected-witness-sha256',
                self.fixture.expected, '--artifacts-root', self.fixture.work / 'artifacts',
                '--build-dir', self.fixture.work / 'build', '--out', self.fixture.work / 'resident',
                '--tokenizer-out', self.fixture.work / 'tokenizer')

    def test_package_grammar_refuses_missing_duplicate_and_abbreviated_before_outputs(self):
        resident = self.fixture.work / 'resident'
        tokenizer = self.fixture.work / 'tokenizer'
        cases = (
            ('missing', self.package_args()[:-2]),
            ('duplicate', self.package_args() + ('--source=' + str(self.fixture.root),)),
            ('abbreviated', self.package_args() + ('--s', self.fixture.root)),
            ('unknown', self.package_args() + ('--unknown', 'value')),
        )
        for label, arguments in cases:
            with self.subTest(label=label):
                result = self.run_cli(*arguments)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(resident.exists())
                self.assertFalse(tokenizer.exists())

    def test_abbreviated_duplicate_option_is_not_an_alternate_selector(self):
        result = self.run_cli(*self.proof_args(), '--s', self.fixture.root)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(self.output.exists())

    def test_package_preflight_refuses_bad_inputs_before_packager(self):
        proof = self.fixture.work / 'proof'
        self.assertEqual(entry.main(list(map(str, self.proof_args(proof)))), 0)
        build = self.fixture.work / 'build'; (build / 'rf48C').mkdir(parents=True)
        artifacts = self.fixture.work / 'artifacts'
        base = ['package', '--source', str(self.fixture.root), '--accepted-data-root', str(self.fixture.owner),
                '--proof-root', str(proof), '--expected-witness-sha256', self.fixture.expected,
                '--artifacts-root', str(artifacts), '--build-dir', str(build),
                '--out', str(self.fixture.work / 'resident'), '--tokenizer-out', str(self.fixture.work / 'tokenizer')]
        cases = [
            ('bad-pin', base[:8] + [self.fixture.expected[:-1] + ('0' if self.fixture.expected[-1] != '0' else '1')] + base[9:]),
            ('output-overlaps-source', base[:-4] + ['--out', str(self.fixture.root / 'new-resident')] + base[-2:]),
            ('outputs-overlap', base[:-2] + ['--tokenizer-out', str(self.fixture.work / 'resident')]),
        ]
        existing = self.fixture.work / 'existing'; existing.mkdir()
        cases.append(('existing-output', base[:-4] + ['--out', str(existing)] + base[-2:]))
        alias = self.fixture.work / 'proof-alias'; alias.symlink_to(proof, target_is_directory=True)
        cases.append(('aliased-proof', base[:6] + [str(alias)] + base[7:]))
        calls = []
        dispatch = types.ModuleType('model_artifact_dispatch')
        dispatch.package_gemma4 = lambda *args, **kwargs: calls.append((args, kwargs))
        with mock.patch.dict(sys.modules, {'model_artifact_dispatch': dispatch}):
            for label, arguments in cases:
                with self.subTest(label=label):
                    self.assertNotEqual(entry.main(arguments), 0)
            (self.fixture.root / 'rf_stack' / 'w0.npy').write_bytes(b'changed')
            self.assertNotEqual(entry.main(base), 0)
        self.assertEqual(calls, [])

    def test_package_compiled_input_boundaries_refuse_before_packager(self):
        proof = self.fixture.work / 'proof'
        self.assertEqual(entry.main(list(map(str, self.proof_args(proof)))), 0)
        artifacts = self.fixture.work / 'artifacts'
        foreign = self.fixture.work / 'foreign-build' / 'rf48C'; foreign.mkdir(parents=True)
        alias_build = self.fixture.work / 'alias-build'
        alias_build.mkdir(); (alias_build / 'rf48C').symlink_to(foreign, target_is_directory=True)
        compiled_build = self.fixture.work / 'compiled-build'; (compiled_build / 'rf48C').mkdir(parents=True)
        source_build = self.fixture.root / 'unselected-build'; (source_build / 'rf48C').mkdir(parents=True)

        def arguments(build, out, tokenizer):
            return ['package', '--source', str(self.fixture.root), '--accepted-data-root', str(self.fixture.owner),
                    '--proof-root', str(proof), '--expected-witness-sha256', self.fixture.expected,
                    '--artifacts-root', str(artifacts), '--build-dir', str(build),
                    '--out', str(out), '--tokenizer-out', str(tokenizer)]

        cases = (
            ('compiled-leaf-alias', arguments(alias_build, self.fixture.work / 'alias-out', self.fixture.work / 'alias-tokenizer')),
            ('output-under-compiled-input', arguments(compiled_build, compiled_build / 'rf48C' / 'out', self.fixture.work / 'compiled-tokenizer')),
            ('build-parent-under-source', arguments(source_build, self.fixture.work / 'source-out', self.fixture.work / 'source-tokenizer')),
        )
        calls = []
        dispatch = types.ModuleType('model_artifact_dispatch')
        dispatch.package_gemma4 = lambda *args, **kwargs: calls.append((args, kwargs))
        with mock.patch.dict(sys.modules, {'model_artifact_dispatch': dispatch}):
            for label, argv in cases:
                with self.subTest(label=label):
                    self.assertNotEqual(entry.main(argv), 0)
        self.assertEqual(calls, [])

    def test_package_uses_pinned_proof_bytes_with_real_output_verifier(self):
        """The packager is the I/O boundary; all entry validation remains unmocked."""
        proof = self.fixture.work / 'proof'
        self.assertEqual(entry.main(list(map(str, self.proof_args(proof)))), 0)
        artifacts = self.fixture.work / 'artifacts'
        build = self.fixture.work / 'build'; (build / 'rf48C').mkdir(parents=True)
        resident = self.fixture.work / 'resident'; tokenizer = self.fixture.work / 'tokenizer'
        package_args = ('package', '--source', self.fixture.root, '--accepted-data-root', self.fixture.owner,
                        '--proof-root', proof, '--expected-witness-sha256', self.fixture.expected,
                        '--artifacts-root', artifacts, '--build-dir', build, '--out', resident,
                        '--tokenizer-out', tokenizer)
        original_source = (self.fixture.root / 'store' / 'manifest.json').read_bytes()
        original_proof = (proof / '.verified-data-witness.json').read_bytes()

        def package_gemma4(repo, package_artifacts, package_build, out, tokenizer_source, tokenizer_out, *,
                           source_store, source_rf_stack):
            del repo, package_build
            resident_forward = Path(__file__).resolve().parents[1] / 'designs' / 'resident_forward'
            sys.path.insert(0, str(resident_forward))
            try:
                import package_artifact
                out.mkdir()
                shutil.copytree(source_rf_stack, out / 'weights')
                shutil.copy2(source_rf_stack / 'w_head.npy', out / 'w_head.npy')
                package_artifact.derive_store(source_store, out / 'store')
            finally:
                sys.path.remove(str(resident_forward))
            checkpoint = package_artifacts / 'gemma4-12b-qat' / 'checkpoint'
            checkpoint.parent.mkdir(parents=True)
            shutil.copytree(tokenizer_source, checkpoint)
            shutil.copytree(tokenizer_source, tokenizer_out)
            manifest = json.loads((out / 'store' / 'manifest.json').read_text())
            header, payload, offset = {}, bytearray(), 0
            for name, tower in sorted(manifest['towers'].items()):
                blob = out / 'store' / 'blobs' / (tower['blob'] + '.bin')
                raw = blob.read_bytes()[tower['offset']:tower['offset'] + tower['length']]
                header[name] = {'dtype': 'F32', 'shape': tower['shape'],
                                'data_offsets': [offset, offset + len(raw)]}
                payload.extend(raw); offset += len(raw)
            encoded = json.dumps(header, separators=(',', ':')).encode()
            encoded += b' ' * ((-len(encoded)) % 8)
            (checkpoint / 'model.safetensors').write_bytes(
                struct.pack('<Q', len(encoded)) + encoded + payload)

        dispatch = types.ModuleType('model_artifact_dispatch')
        dispatch.package_gemma4 = package_gemma4
        original_read = entry.gemma4._read
        changed = False

        def mutate_after_proof_read(path, *args):
            nonlocal changed
            raw = original_read(path, *args)
            if Path(path) == proof / '.verified-data-witness.json' and not changed:
                changed = True
                Path(path).write_bytes(b'{}')
            return raw

        stdout = io.StringIO()
        with (mock.patch.dict(sys.modules, {'model_artifact_dispatch': dispatch}),
              mock.patch.object(entry.gemma4, '_read', side_effect=mutate_after_proof_read),
              contextlib.redirect_stdout(stdout)):
            self.assertEqual(entry.main(list(map(str, package_args))), 0)
        summary = json.loads(stdout.getvalue())
        self.assertTrue(changed)
        self.assertEqual(summary['witness_sha256'], self.fixture.expected)
        self.assertEqual((resident / '.verified-data-witness.json').read_bytes(), original_proof)
        self.assertEqual((self.fixture.root / 'store' / 'manifest.json').read_bytes(), original_source)
        self.assertEqual((proof / '.verified-data-witness.json').read_bytes(), b'{}')

    def test_import_in_readonly_directory_creates_nothing(self):
        readonly = self.fixture.work / 'readonly'
        readonly.mkdir()
        readonly.chmod(0o500)
        try:
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(SCRIPT.parent)
            result = subprocess.run([sys.executable, '-B', '-c', 'import gemma4_verified_entry'],
                                    cwd=readonly, env=environment, text=True, capture_output=True)
        finally:
            readonly.chmod(0o700)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(readonly.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
