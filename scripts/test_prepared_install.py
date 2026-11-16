"""Offline fixtures: prepared verification never invokes a builder or rewrites receipts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import model_artifact_dispatch as dispatch
import model_artifact_stages as stages
import prepared_install as prepared
import verified_gemma_data as verified
from test_verified_gemma_data import gemma_dump, witness_file

SOURCE = Path(__file__).resolve().parents[1]
# The fixture compiler stands in for scripts/build_llm_decode.sh, so its meta.json declares the same
# IRON provenance the real compiler records; prepared verification judges it like any other subject.
def served_meta(commit):
    return {'spec': 'gemma3-270m', 'compiled_by': 'fixture compiler boundary',
            'iron': {'commit': commit, 'dirty': False, 'identity': f'pinned:{commit}'}}


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

    def test_prepared_installer_never_bootstraps_export_sdks(self):
        installer = (Path(__file__).resolve().parents[1] / 'install.sh').read_text()
        section = installer.split('# 2b.', 1)[1].split('# 2c.', 1)[0]
        section = section[section.index('\n') + 1:]
        enclosing = installer.split('# 2a. cargo', 1)[1].split('# 3a.', 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'scripts').mkdir()
            (root / 'venv/bin').mkdir(parents=True)
            (root / 'bin').mkdir()
            marker = root / 'bootstrap-called'
            for name in ('setup_pyannote_community_venv.sh', 'setup_pyannote_venv.sh'):
                path = root / 'scripts' / name
                path.write_text('#!/bin/sh\necho sdk >> "$BOOTSTRAP_MARKER"\n')
                path.chmod(0o755)
            python = root / 'venv/bin/python'
            python.write_text('#!/bin/sh\ncase "$*" in *--pyannote-sdks*) printf "community\\nlegacy\\n" ;; esac\n')
            python.chmod(0o755)
            uv = root / 'bin/uv'
            uv.write_text('#!/bin/sh\necho pip >> "$BOOTSTRAP_MARKER"\n')
            uv.chmod(0o755)
            env = os.environ | {'REPO': str(root), 'ONNX_ASR_VENV': str(root / 'venv'),
                                'MODEL_IRON_PY': str(python), 'ENGINE_CONFIG': str(root / 'engine.toml'),
                                'BOOTSTRAP_MARKER': str(marker), 'PREPARED_MANIFEST': 'fixture-manifest',
                                'PYANNOTE_COMMUNITY_VENV': str(root / 'community'),
                                'PYANNOTE_VENV': str(root / 'legacy'),
                                'PATH': str(root / 'bin') + os.pathsep + os.environ['PATH']}
            (root / 'engine.toml').write_text('fixture')
            prefix = 'ok() { :; }; die() { echo "$*" >&2; exit 88; };\n'
            for mode in ('prepared', 'build'):
                exercised = enclosing if mode == 'prepared' else section
                result = subprocess.run(['bash', '-euc', prefix + exercised],
                                        env=env | {'INSTALL_MODE': mode}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                if mode == 'prepared':
                    self.assertFalse(marker.exists(), 'prepared installation invoked SDK mutation')
                else:
                    self.assertEqual(marker.read_text().splitlines(), ['sdk', 'sdk', 'pip'])

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


class VerifiedDataPreparedTests(unittest.TestCase):
    """Actual Gemma3 verified-data materialization, publication and prepared binding.

    The witness verifier/copy helper, the stage lifecycle, the receipt and the prepared check are the
    maintained implementations; only the native compiler boundary is a fixture that writes a synthetic
    program meta.json. No real receipt is faked and no real Gemma data is touched.
    """

    scenario = 'scenarios/generate-gemma3-270m.toml'

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.iron_commit = 'e' * 40
        self.artifacts, self.build = self.root / 'candidate', self.root / 'build'
        self.owner = self.root / 'data-owner'
        self.source = self.owner / 'gemma3-270m/weights_qat_int4g32'
        self.files = gemma_dump(self.source)
        self.witness = witness_file(self.source, self.root / 'evidence/data-witness.json',
                                    legacy_receipt='a' * 64, legacy_output_digest='b' * 64)
        self.inputs = self.root / 'model-inputs'
        checkpoint = self.inputs / 'gemma3-270m/checkpoint'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text('{}')
        (checkpoint / 'tokenizer.json').write_text('{}')
        (checkpoint / 'model.safetensors').write_bytes(b'checkpoint')
        self.checkpoint = checkpoint
        self.recipe = dispatch.verified_data_recipe('gemma3-270m', self.scenario, SOURCE, self.artifacts,
                                                   self.inputs, self.build, witness=self.witness,
                                                   data_source=self.source, accepted_data_root=self.owner)
        self.calls = []

    def execute(self, recipe, repo):
        step = recipe.commands[0].step
        self.calls.append(step)
        if step == dispatch.VERIFIED_DATA_STEP:
            # The declared verifier/copy helper really runs, with the staged output root.
            subprocess.run(list(recipe.commands[0].argv), check=True, cwd=repo,
                           env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'})
            return
        target = Path(recipe.commands[0].stage.outputs[0].root)
        target.mkdir(parents=True, exist_ok=True)
        if step == 'decode':
            (target / 'meta.json').write_text(json.dumps(served_meta(self.iron_commit)))
        else:
            dispatch.copy_tokenizer(self.checkpoint, target)

    def build_variant(self):
        self.calls.clear()
        with patch.object(dispatch, 'LOCK_ROOT', self.root / 'locks'), \
             patch.object(dispatch, 'run_commands', side_effect=self.execute), \
             patch.object(stages, 'sdk_identity', return_value='fixture SDK'), \
             patch.object(stages, 'toolchain_identity', return_value='fixture toolchain'):
            return dispatch.build_recipe(self.recipe, SOURCE, SOURCE / self.scenario,
                                         self.artifacts, self.build)

    def receipt(self):
        return json.loads(dispatch.manifest_path(self.recipe).read_text())

    def entry(self, record=None, **overrides):
        record = record or self.receipt()
        weights = Path(self.recipe.outputs[1])
        entry = {'variant': verified.VARIANT, 'recipe_key': record['recipe_key'],
                 'source_root': str(self.source.resolve()),
                 'witness': {'sha256': prepared.sha256_file(weights / verified.SIDECAR)}}
        entry.update(overrides)
        return entry

    def test_recorded_duplicate_flags_cannot_claim_the_first_value_was_consumed(self):
        self.assertEqual(self.build_variant(), 'BUILT')
        record = self.receipt()
        command = next(item for item in record['commands'] if item['step'] == dispatch.VERIFIED_DATA_STEP)
        command['argv'].extend(['--out', '/unrecorded-output'])
        stage = record['stages'][dispatch.VERIFIED_DATA_STEP]
        stage['producer']['command']['argv'] = list(command['argv'])
        stage['producer_source'] = dispatch.stage_digest(stage['producer'])
        stage['key'] = dispatch.stage_digest({'version': dispatch.StageRunner.VERSION,
                                             'source': stage['producer_source'], 'dependencies': {}})
        with self.assertRaises(ValueError):
            prepared.verify_data_variant(self.recipe, record, self.entry(record))

    def test_recorded_owner_must_actually_contain_the_declared_model_input(self):
        self.assertEqual(self.build_variant(), 'BUILT')
        record = self.receipt()
        command = next(item for item in record['commands'] if item['step'] == dispatch.VERIFIED_DATA_STEP)
        command['argv'][command['argv'].index('--accepted-data-root') + 1] = '/different-owner'
        stage = record['stages'][dispatch.VERIFIED_DATA_STEP]
        stage['producer']['command']['argv'] = list(command['argv'])
        stage['producer_source'] = dispatch.stage_digest(stage['producer'])
        stage['key'] = dispatch.stage_digest({'version': dispatch.StageRunner.VERSION,
                                             'source': stage['producer_source'], 'dependencies': {}})
        with self.assertRaises(ValueError):
            prepared.verify_data_variant(self.recipe, record, self.entry(record))

    def test_materialize_publish_and_prepared_binding_then_no_rebuild(self):
        self.assertEqual(self.build_variant(), 'BUILT')
        self.assertEqual(self.calls, [dispatch.VERIFIED_DATA_STEP, 'decode', 'tokenizer'])
        record = self.receipt()
        self.assertEqual(record['model'], 'gemma3-270m')
        self.assertEqual(set(record['stages']), {dispatch.VERIFIED_DATA_STEP, 'decode', 'tokenizer'})
        producer = record['stages'][dispatch.VERIFIED_DATA_STEP]
        self.assertEqual(producer['action'], 'executed')
        self.assertIsNone(producer['producer']['sdk'])
        self.assertIsNone(producer['producer']['toolchain'])
        self.assertEqual(list(producer['producer']['sources']), ['scripts/verified_gemma_data.py'])
        self.assertNotIn('dump_llm_weights', json.dumps(producer['producer']))
        self.assertEqual({Path(name).name for name in producer['producer']['inputs']},
                         {path.name for path in self.source.iterdir()} | {self.witness.name})
        self.assertTrue((Path(self.recipe.outputs[0]) / 'meta.json').is_file())
        published = {path.name: path.read_bytes() for path in Path(self.recipe.outputs[1]).iterdir()
                     if path.name != verified.SIDECAR}
        self.assertEqual(published, {name: path.read_bytes() for name, path in self.files.items()})
        prepared.verify_data_variant(self.recipe, record, self.entry(record))
        self.assertEqual(self.build_variant(), 'HIT')
        self.assertEqual(self.calls, [], 'a witnessed hit must not rebuild anything')

    def test_prepared_binding_survives_losing_the_legacy_input_and_the_review_copy(self):
        self.assertEqual(self.build_variant(), 'BUILT')
        record, entry = self.receipt(), self.entry()
        shutil.rmtree(self.owner)
        self.witness.unlink()
        self.assertFalse(self.source.exists())
        prepared.verify_data_variant(self.recipe, record, entry)

    def test_prepared_binding_refuses_every_mismatched_or_fabricated_claim(self):
        self.assertEqual(self.build_variant(), 'BUILT')
        record, entry = self.receipt(), self.entry()
        weights = Path(self.recipe.outputs[1])
        sidecar = weights / verified.SIDECAR
        self.assertFalse(any(Path(raw) != weights and (Path(raw) / verified.SIDECAR).is_file()
                             for raw in self.recipe.outputs))
        original = sidecar.read_bytes()
        payload = weights / 'w.gate.npy'
        payload_bytes = payload.read_bytes()
        for harm in ('variant', 'key', 'witness', 'payload', 'foreign', 'no-stages', 'lineage',
                     'quantizer-producer', 'cross-model'):
            with self.subTest(harm=harm):
                if harm == 'payload':
                    payload.write_bytes(payload_bytes[:-1] + bytes([payload_bytes[-1] ^ 0xFF]))
                    self.addCleanup(payload.write_bytes, payload_bytes)
                elif harm == 'witness':
                    sidecar.write_text(json.dumps(json.loads(original.decode()), sort_keys=True))
                    self.addCleanup(sidecar.write_bytes, original)
                declared = dict(entry)
                if harm == 'variant':
                    declared['variant'] = 'source-quantized'
                elif harm == 'key':
                    declared['recipe_key'] = 'c' * 64
                elif harm == 'witness':
                    declared['witness'] = {'sha256': 'd' * 64}
                elif harm == 'cross-model':
                    declared['model'] = 'gemma4-12b'
                altered = record
                if harm in ('no-stages', 'lineage', 'quantizer-producer'):
                    altered = json.loads(json.dumps(record))
                    if harm == 'no-stages':
                        altered.pop('stages')
                    elif harm == 'lineage':
                        altered['stages'][dispatch.VERIFIED_DATA_STEP]['action'] = 'lineage-only'
                    else:
                        altered['stages'][dispatch.VERIFIED_DATA_STEP]['producer']['sources'] = \
                            {'scripts/dump_llm_weights.py': '0' * 64}
                with self.assertRaises(ValueError):
                    prepared.verify_data_variant(self.recipe, altered, declared)

    def test_a_historical_receipt_without_stage_witnesses_still_refuses_the_variant(self):
        """The maintained no-stage guard stays the answer to an unwitnessed legacy receipt.

        The real retained Gemma3 receipt records no stages. A variant build must refuse it before any
        stage executes, and must not restamp, seed or rebuild it; only prepared installation or a
        separately qualified frontier adoption may consume it.
        """
        for raw in self.recipe.outputs:
            Path(raw).mkdir(parents=True)
        (Path(self.recipe.outputs[0]) / 'meta.json').write_text(json.dumps({'spec': 'gemma3-270m',
                                                                           'origin': 'legacy'}))
        for name, source in self.files.items():
            shutil.copy2(source, Path(self.recipe.outputs[1]) / name)
        dispatch.copy_tokenizer(self.checkpoint, Path(self.recipe.outputs[2]))
        ordinary = dispatch.recipe_for('gemma3-270m', self.scenario, SOURCE, self.artifacts,
                                       self.inputs, self.build)
        dispatch.write_receipt(ordinary, 'a' * 64)
        self.assertNotIn('stages', json.loads(dispatch.manifest_path(ordinary).read_text()))
        before = {str(path.relative_to(self.artifacts)): (path.read_bytes(), path.stat().st_ino)
                  for path in sorted(self.artifacts.rglob('*')) if path.is_file()}
        with self.assertRaisesRegex(RuntimeError, 'historical receipt'):
            self.build_variant()
        self.assertEqual(self.calls, [], 'the refusal must happen before any stage executes')
        after = {str(path.relative_to(self.artifacts)): (path.read_bytes(), path.stat().st_ino)
                 for path in sorted(self.artifacts.rglob('*')) if path.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.build / 'model-artifact-state').exists(),
                         'a refused variant build must not seed stage state')

    def test_manifest_version_and_model_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, files, data, manifest, recipe, kernels, scenario, pin, artifacts = prepared_fixture(root)
            declared = {'gemma3-270m': {'variant': verified.VARIANT, 'recipe_key': 'e' * 64,
                                        'source_root': '/data/gemma3-270m/weights_qat_int4g32',
                                        'witness': {'sha256': 'f' * 64}}}
            data['data_variants'] = declared
            manifest.write_text(json.dumps(data))
            with patch.object(prepared, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'version 2'):
                    prepared.verify(manifest, repo)
                run.assert_not_called()
            data['version'] = 2
            data.pop('data_variants')
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'needs data_variants'):
                prepared.verify(manifest, repo)
            data['data_variants'] = {'bge-base': declared['gemma3-270m']}
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Gemma-only'):
                prepared.verify(manifest, repo)
            data['data_variants'] = {'gemma3-270m': dict(declared['gemma3-270m'], variant='source-quantized')}
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'unknown verified-data declaration'):
                prepared.verify(manifest, repo)
            # A well-formed binding must still belong to a model this config actually selects.
            data['data_variants'] = declared
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'models its config does not select'):
                prepared.verify(manifest, repo)

    def test_recipe_key_binding_uses_the_witnessed_receipt_key(self):
        self.assertEqual(self.build_variant(), 'BUILT')
        record = self.receipt()
        witness = {'sha256': prepared.sha256_file(dispatch.manifest_path(self.recipe)),
                   'recipe_key': record['recipe_key']}
        prepared.verify_receipt(self.recipe, witness)
        # The variant serves the same scope, but its receipt declares the verifier, never the quantizer.
        ordinary = dispatch.recipe_for('gemma3-270m', self.scenario, SOURCE, self.artifacts,
                                       self.inputs, self.build)
        self.assertEqual(dispatch.manifest_path(ordinary), dispatch.manifest_path(self.recipe))
        self.assertEqual([item.step for item in ordinary.commands], ['weights', 'decode', 'tokenizer'])
        self.assertIn('dump_llm_weights.py', json.dumps([list(item.argv) for item in ordinary.commands]))
        self.assertEqual([step['step'] for step in record['commands']],
                         [dispatch.VERIFIED_DATA_STEP, 'decode', 'tokenizer'])
        self.assertNotIn('dump_llm_weights', json.dumps(record['commands']))

    def _fabricated_stage(self, record):
        """A marker-only stage record: the labels a variant needs, and no consumed input at all."""
        altered = json.loads(json.dumps(record))
        altered['stages'][dispatch.VERIFIED_DATA_STEP] = {
            'action': 'executed',
            'producer': {'sources': {'scripts/verified_gemma_data.py': '0' * 64}, 'sdk': None,
                         'toolchain': None}}
        return altered

    def _harm_producer(self, record, harm):
        """Damage one published field, leaving every other acceptance predicate intact."""
        altered = json.loads(json.dumps(record))
        stage = altered['stages'][dispatch.VERIFIED_DATA_STEP]
        inputs = stage['producer']['inputs']
        argv = stage['producer']['command']['argv']
        prefix = str(self.source.resolve()) + '/'
        members = sorted(json.loads((Path(self.recipe.outputs[1]) / verified.SIDECAR)
                                    .read_text())['members'])
        if harm == 'drop-member':
            inputs.pop(prefix + members[0])
        elif harm == 'member-sha':
            inputs[prefix + members[0]] = '0' * 64
        elif harm == 'extra-member':
            inputs[prefix + 'not-witnessed.npy'] = '1' * 64
        elif harm == 'drop-witness':
            inputs.pop(str(self.witness))
        elif harm == 'witness-sha':
            inputs[str(self.witness)] = '2' * 64
        elif harm == 'foreign-source':
            stage['producer']['inputs'] = {('elsewhere/' + Path(raw).name): value
                                           for raw, value in inputs.items()}
        elif harm == 'quantizer-command':
            altered['commands'].append({'step': 'weights', 'env': {},
                                        'argv': ['/iron/python', 'scripts/dump_llm_weights.py',
                                                 '--out', self.recipe.outputs[1]], 'stage': None})
        elif harm == 'command-mismatch':
            altered['commands'][0]['argv'][0] = '/some/other/python'
        elif harm == 'model-flag':
            argv[argv.index('--model') + 1] = 'gemma4-12b'
        elif harm == 'out-flag':
            argv[argv.index('--out') + 1] = self.recipe.outputs[0]
        elif harm == 'producer-source':
            stage['producer_source'] = 'a' * 64
        elif harm == 'stale-key':
            stage['key'] = 'b' * 64
        elif harm == 'stage-dependency':
            altered['commands'][0]['stage']['dependencies'] = ['decode']
        else:
            raise AssertionError(f'unknown harm {harm}')
        return altered

    def test_the_receipt_record_is_the_owners_own_stage_record(self):
        """The published stage key is the maintained owner formula over the recorded producer."""
        self.assertEqual(self.build_variant(), 'BUILT')
        record = self.receipt()
        stage = record['stages'][dispatch.VERIFIED_DATA_STEP]
        self.assertEqual(stage['producer_source'], dispatch.stage_digest(stage['producer']))
        self.assertEqual(stage['key'], dispatch.stage_digest({'version': stages.StageRunner.VERSION,
                                                             'source': stage['producer_source'],
                                                             'dependencies': {}}))
        prepared.verify_data_variant(self.recipe, record, self.entry(record))

    def test_the_producer_record_must_prove_which_bytes_the_stage_consumed(self):
        self.assertEqual(self.build_variant(), 'BUILT')
        record, entry = self.receipt(), self.entry()
        with self.assertRaisesRegex(ValueError, 'gemma3-270m'):
            prepared.verify_data_variant(self.recipe, self._fabricated_stage(record), entry)
        harms = ('drop-member', 'member-sha', 'extra-member', 'drop-witness', 'witness-sha',
                 'foreign-source', 'quantizer-command', 'command-mismatch', 'model-flag', 'out-flag',
                 'producer-source', 'stale-key', 'stage-dependency')
        for harm in harms:
            with self.subTest(harm=harm):
                with self.assertRaisesRegex(ValueError, 'gemma3-270m'):
                    prepared.verify_data_variant(self.recipe, self._harm_producer(record, harm), entry)

    def test_a_republished_witness_that_is_not_the_consumed_bytes_is_refused(self):
        """Equivalent JSON is not the same witness: the receipt binds the consumed bytes."""
        self.assertEqual(self.build_variant(), 'BUILT')
        record, entry = self.receipt(), self.entry()
        sidecar = Path(self.recipe.outputs[1]) / verified.SIDECAR
        original = sidecar.read_bytes()
        sidecar.write_text(json.dumps(json.loads(original.decode()), indent=4, sort_keys=True) + '\n')
        self.addCleanup(sidecar.write_bytes, original)
        with self.assertRaisesRegex(ValueError, 'gemma3-270m'):
            prepared.verify_data_variant(self.recipe, record,
                                         self.entry(record, witness={'sha256': prepared.sha256_file(sidecar)}))

    def test_a_manifest_cannot_bind_a_data_root_its_model_never_declared(self):
        declaration = {'variant': verified.VARIANT, 'recipe_key': 'e' * 64, 'source_root': '',
                       'witness': {'sha256': 'f' * 64}}
        for label, root in (('other-model', '/data/qwen3-0.6b/weights_qat_int4g32'),
                            ('lane', '/data/weights'), ('relative', 'gemma3-270m/weights_qat_int4g32')):
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, 'source_root'):
                    prepared.data_variant_gate({'version': 2, 'data_variants': {
                        'gemma3-270m': dict(declaration, source_root=root)}})
        # The legal spelling is the model's declared location under any accepted data owner.
        prepared.data_variant_gate({'version': 2, 'data_variants': {
            'gemma3-270m': dict(declaration, source_root='/data/owner/gemma3-270m/weights_qat_int4g32')}})

    def _prepared_store(self, *, version, subject, variants=None):
        """A prepared store around a real publication: repo, config, kernels, receipts, manifest."""
        store = self.root / 'prepared'
        repo = store / 'repo'
        (repo / 'scenarios').mkdir(parents=True)
        lock = repo / 'toolchain.lock'
        lock.write_text('IRON_SOURCE_COMMIT=' + 'e' * 40 + '\n')
        pin = hashlib.sha256(lock.read_text().encode()).hexdigest()[:12]
        shutil.copy2(SOURCE / self.scenario, repo / self.scenario)
        kernels = store / 'kernels'
        (kernels / 'whole_array').mkdir(parents=True)
        for directory in (kernels, kernels / 'whole_array'):
            (directory / '.toolchain-stamp').write_text(pin)
        config = store / 'engine.toml'
        config.write_text(f'[[model]]\nname="gemma3-270m"\nscenario="{self.scenario}"\n')
        recipe, key = subject
        binaries = {}
        for name in ('npu', 'npu_dev', 'libonnxruntime.so.1.26.0'):
            target = store / name
            target.write_bytes(b'fixture')
            target.chmod(0o755)
            binaries['onnxruntime' if name.startswith('lib') else name] = target
        files = dict(binaries, config=config, toolchain=lock)
        data = {'version': version, 'artifacts': str(self.artifacts),
                'kernels': {'path': str(kernels), 'sha256': dispatch.tree_digest(kernels)},
                'scenarios': {self.scenario: prepared.sha256_file(repo / self.scenario)},
                'receipts': {recipe.model: {'sha256': prepared.sha256_file(dispatch.manifest_path(recipe)),
                                            'recipe_key': key}}}
        data.update({name: {'path': str(path), 'sha256': prepared.sha256_file(path)}
                     for name, path in files.items()})
        if variants:
            data['data_variants'] = variants
        manifest = store / 'prepared.json'
        manifest.write_text(json.dumps(data))
        return repo, manifest

    def test_a_variant_closure_verifies_end_to_end_and_cannot_downgrade_to_version_one(self):
        self.assertEqual(self.build_variant(), 'BUILT')
        record = self.receipt()
        repo, manifest = self._prepared_store(version=2, subject=(self.recipe, record['recipe_key']),
                                              variants={'gemma3-270m': self.entry(record)})
        with patch.object(prepared, 'run') as run:
            result = prepared.verify(manifest, repo)
            self.assertEqual(result['ENGINE_ARTIFACTS'], str(self.artifacts))
            self.assertTrue(run.called, 'the native CLI boundary is the only mocked seam')
            data = json.loads(manifest.read_text())
            data['version'] = 1
            data.pop('data_variants')
            manifest.write_text(json.dumps(data))
            run.reset_mock()
            with self.assertRaisesRegex(ValueError, 'does not declare'):
                prepared.verify(manifest, repo)
            run.assert_not_called()
            data['version'] = 2
            data['data_variants'] = {}
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'needs data_variants'):
                prepared.verify(manifest, repo)

    def test_a_normal_version_one_gemma3_receipt_still_verifies_without_a_variant_declaration(self):
        ordinary = dispatch.recipe_for('gemma3-270m', self.scenario, SOURCE, self.artifacts,
                                       self.inputs, self.build)
        for raw in ordinary.outputs:
            Path(raw).mkdir(parents=True, exist_ok=True)
        (Path(ordinary.outputs[0]) / 'meta.json').write_text(json.dumps(served_meta(self.iron_commit)))
        # A real quantizer payload with no witness marker anywhere: the version 1 shape stays legal.
        for name, path in self.files.items():
            shutil.copy2(path, Path(ordinary.outputs[1]) / name)
        (Path(ordinary.outputs[2]) / 'config.json').write_text('{}')
        (Path(ordinary.outputs[2]) / 'tokenizer.json').write_text('{}')
        (Path(ordinary.outputs[2]) / 'model.safetensors').write_bytes(b'checkpoint')
        dispatch.write_receipt(ordinary, 'a' * 64)
        repo, manifest = self._prepared_store(version=1, subject=(ordinary, 'a' * 64))
        with patch.object(prepared, 'run') as run:
            self.assertEqual(prepared.verify(manifest, repo)['ENGINE_ARTIFACTS'], str(self.artifacts))
            self.assertTrue(run.called)
if __name__ == '__main__':
    unittest.main()
