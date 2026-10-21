"""Offline fixtures: prepared verification never invokes a builder or rewrites receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch

import model_artifact_dispatch as dispatch
import prepared_install as prepared


def prepared_fixture(root):
    repo = root / 'repo'
    (repo / 'scenarios').mkdir(parents=True)
    lock = repo / 'toolchain.lock'
    lock.write_text('IRON_SOURCE_COMMIT=' + 'a' * 40 + '\n')
    pin = hashlib.sha256(lock.read_bytes()).hexdigest()[:12]
    scenario = repo / 'scenarios/bge-base.toml'
    scenario.write_text('[scenario]\nkind="embedding"\n[artifacts]\nweights="artifacts/bge-base"\n')
    config = root / 'engine.toml'
    config.write_text('[[model]]\nname="bge-base"\nscenario="scenarios/bge-base.toml"\n')
    artifacts = root / 'artifacts'
    output = artifacts / 'bge-base'
    output.mkdir(parents=True)
    (output / 'model.onnx').write_bytes(b'model')
    (output / 'tokenizer.json').write_text('{}')
    recipe = dispatch.recipe_for('bge-base', 'scenarios/bge-base.toml', repo, artifacts, root / 'inputs', root / 'build')
    dispatch.write_receipt(recipe, 'b' * 64)
    kernels = root / 'kernels'
    (kernels / 'whole_array').mkdir(parents=True)
    for directory in (kernels, kernels / 'whole_array'):
        (directory / '.toolchain-stamp').write_text(pin)
    files = {'config': config, 'toolchain': lock}
    for name in ('npu', 'npu_dev', 'libonnxruntime.so.1.26.0'):
        target = root / name
        target.write_bytes(b'fixture')
        target.chmod(0o755)
        files['onnxruntime' if name.startswith('lib') else name] = target
    data = {'version': 1, 'artifacts': str(artifacts),
            'kernels': {'path': str(kernels), 'sha256': dispatch.tree_digest(kernels)},
            'scenarios': {'scenarios/bge-base.toml': prepared.sha256_file(scenario)},
            'receipts': {'bge-base': {'sha256': prepared.sha256_file(dispatch.manifest_path(recipe)), 'recipe_key': 'b' * 64}}}
    data.update({name: {'path': str(path), 'sha256': prepared.sha256_file(path)} for name, path in files.items()})
    manifest = root / 'prepared.json'
    manifest.write_text(json.dumps(data))
    return repo, files, data, manifest, recipe, kernels, scenario, pin, artifacts


class PreparedInstallTests(unittest.TestCase):
    def test_complete_preflight_and_missing_tampered_incompatible_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, files, data, manifest, recipe, kernels, scenario, pin, artifacts = prepared_fixture(root)
            config = files['config']
            with patch.object(prepared, 'run') as run:
                result = prepared.verify(manifest, repo)
                self.assertEqual(result['ENGINE_ARTIFACTS'], str(artifacts))
                commands = [list(map(str, call.args[0])) for call in run.call_args_list]
                self.assertEqual([cmd[1] for cmd in commands[:2]], ['kernels-manifest', 'kernels-verify'])
                for cmd in commands[2:]:
                    self.assertEqual(cmd, [str(files['npu']), '--config', str(config), 'model', 'ls', '--output', 'json'])
                self.assertIn('LD_LIBRARY_PATH', run.call_args_list[2].kwargs['env'])
                self.assertNotIn('LD_LIBRARY_PATH', run.call_args_list[3].kwargs['env'])
                self.assertIn('--check-native-contracts-only', commands[0])
                run.reset_mock()
                for name in ('npu', 'onnxruntime'):
                    original = files[name].read_bytes()
                    files[name].write_bytes(b'tampered')
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        prepared.verify(manifest, repo)
                    files[name].write_bytes(original)
                stamp = kernels / 'whole_array/.toolchain-stamp'
                stamp.write_text('wrong-pin')
                data['kernels']['sha256'] = dispatch.tree_digest(kernels)
                manifest.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'toolchain mismatch'):
                    prepared.verify(manifest, repo)
                stamp.write_text(pin)
                data['kernels']['sha256'] = dispatch.tree_digest(kernels)
                manifest.write_text(json.dumps(data))
                scenario.write_text(scenario.read_text() + '# changed\n')
                with self.assertRaisesRegex(ValueError, 'scenario changed'):
                    prepared.verify(manifest, repo)
                run.assert_not_called()
                files['npu'].unlink()
                with self.assertRaises(FileNotFoundError):
                    prepared.verify(manifest, repo)


    @unittest.skipUnless(os.environ.get('PREPARED_TEST_NPU') and os.environ.get('PREPARED_TEST_ORT'),
                         'set PREPARED_TEST_NPU and PREPARED_TEST_ORT for offline real-CLI check')
    def test_selected_invalid_server_rejects_before_installer_writes(self):
        source_repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, files, data, manifest, *_ = prepared_fixture(root)
            shutil.copy2(source_repo / 'install.sh', repo / 'install.sh')
            (repo / 'scripts').symlink_to(source_repo / 'scripts', target_is_directory=True)
            files['npu'] = Path(os.environ['PREPARED_TEST_NPU']).resolve()
            files['onnxruntime'] = Path(os.environ['PREPARED_TEST_ORT']).resolve()
            # Only kernel checks are fixtures: the service config parser is the real candidate CLI.
            files['npu_dev'].write_text('#!/bin/sh\nexit 0\n')
            files['config'].write_text('[server]\nport="invalid"\n' + files['config'].read_text())
            data.update({name: {'path': str(path), 'sha256': prepared.sha256_file(path)}
                         for name, path in files.items()})
            manifest.write_text(json.dumps(data))
            ambient = root / 'ambient.toml'
            ambient.write_text('[server]\nport=11434\n')
            destinations = [root / name for name in ('home', 'live', 'bin', 'lib', 'archive')]
            env = os.environ | {'HOME': str(destinations[0]), 'NPU_CONFIG': str(ambient),
                                'INSTALL_MODE': 'prepared', 'PREPARED_MANIFEST': str(manifest),
                                'ENGINE_ROOT': str(destinations[1]), 'ENGINE_BIN_DIR': str(destinations[2]),
                                'STABLE_LIB_DIR': str(destinations[3]), 'ENGINE_BIN_ARCHIVE': str(destinations[4])}
            env.pop('LD_LIBRARY_PATH', None)
            positive = subprocess.run([str(files['npu']), 'model', 'ls', '--output', 'json'],
                                      env=env, capture_output=True, text=True)
            self.assertEqual(positive.returncode, 0, positive.stderr)
            result = subprocess.run(['bash', str(repo / 'install.sh')], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid type', result.stderr, result.stdout + result.stderr)
            self.assertIn('invalid', result.stderr)
            for destination in destinations:
                self.assertFalse(destination.exists(), str(destination))

    def test_installer_pruning_preserves_prepared_history_and_retains_build_policy(self):
        installer = (Path(__file__).resolve().parents[1] / 'install.sh').read_text()
        archive_start = installer.index('# Bounded: keep the newest 10')
        archive_end = installer.index('# ---------------------------------------------------------------------------', archive_start)
        archive_block = installer[archive_start:archive_end]
        function_start = installer.index('prune_prev() {')
        function_end = installer.index('\n}\n', function_start) + 3
        prune_function = installer[function_start:function_end]
        for mode in ('prepared', 'build'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root / 'archive'
                archive.mkdir()
                for index in range(12):
                    binary = archive / f'npu-{index:02x}'
                    binary.write_text(f'original binary {index}')
                    os.utime(binary, (1000 + index, 1000 + index))
                backups = [root / f'artifacts.prev-2026100{day}T160832' for day in (1, 2, 3)]
                for backup in backups:
                    backup.mkdir()
                    (backup / 'sentinel').write_text(backup.name)
                before = {str(path.relative_to(root)): (path.stat().st_ino, path.read_bytes())
                          for path in root.rglob('*') if path.is_file()}
                result = subprocess.run(['bash', '-c', 'set -euo pipefail\ninfo() { :; }\n'
                                         + archive_block + prune_function + '\nprune_prev artifacts\n'],
                                        env=os.environ | {'INSTALL_MODE': mode,
                                                          'ENGINE_BIN_ARCHIVE': str(archive),
                                                          'ENGINE_ROOT': str(root)},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                after = {str(path.relative_to(root)): (path.stat().st_ino, path.read_bytes())
                         for path in root.rglob('*') if path.is_file()}
                if mode == 'prepared':
                    self.assertEqual(before, after)
                    self.assertTrue(all(backup.is_dir() for backup in backups))
                else:
                    self.assertEqual(len(list(archive.iterdir())), 10)
                    self.assertFalse((archive / 'npu-00').exists())
                    self.assertFalse((archive / 'npu-01').exists())
                    self.assertEqual([backup.exists() for backup in backups], [False, False, True])

    def test_cls_maps_to_exact_canonical_data_recipe(self):
        roots = [Path('/repo'), Path('/artifacts'), Path('/inputs'), Path('/build')]
        mean = dispatch.recipe_for('bge-base', 'scenarios/bge-base.toml', *roots)
        cls = dispatch.recipe_for('bge-base-cls', 'scenarios/bge-base-cls.toml', *roots)
        self.assertEqual(mean, cls)
        with self.assertRaises(ValueError):
            dispatch.recipe_for('bge-base-cls', 'scenarios/bge-base.toml', *roots)

    def test_existing_receipt_accepts_stale_authority_but_not_tampered_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'bge-base'
            out.mkdir()
            (out / 'model.onnx').write_bytes(b'model')
            (out / 'tokenizer.json').write_text('{}')
            recipe = dispatch.Recipe('bge-base', 'scenarios/bge-base.toml', (), (str(out),), ())
            dispatch.write_receipt(recipe, 'a' * 64)
            receipt = dispatch.manifest_path(recipe)
            original = receipt.read_bytes()
            witness = {'sha256': prepared.sha256_file(receipt), 'recipe_key': 'a' * 64}
            with patch.object(dispatch, 'run_commands', side_effect=AssertionError('build forbidden')):
                prepared.verify_receipt(recipe, witness)
                (out / 'model.onnx').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'output'):
                    prepared.verify_receipt(recipe, witness)
            self.assertEqual(receipt.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'receipt'):
                prepared.verify_receipt(recipe, witness | {'sha256': '0' * 64})

    def test_invalid_prepared_install_has_no_live_writes_or_build_commands(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traps = root / 'traps'
            traps.mkdir()
            log = root / 'commands'
            for name in ('cargo', 'uv', 'pip', 'systemctl'):
                trap = traps / name
                trap.write_text('#!/bin/sh\necho forbidden >> "$TRAP_LOG"\nexit 99\n')
                trap.chmod(0o755)
            env = os.environ | {'HOME': str(root), 'INSTALL_MODE': 'prepared',
                                'PREPARED_MANIFEST': str(root / 'missing.json'),
                                'PATH': str(traps) + ':' + os.environ['PATH'], 'TRAP_LOG': str(log),
                                'ENGINE_ROOT': str(root / 'live'), 'ENGINE_BIN_DIR': str(root / 'bin'),
                                'STABLE_LIB_DIR': str(root / 'lib')}
            result = subprocess.run(['bash', str(repo / 'install.sh')], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(log.exists(), result.stdout + result.stderr)
            for name in ('live', 'bin', 'lib', '.config'):
                self.assertFalse((root / name).exists(), name)


if __name__ == '__main__':
    unittest.main()
