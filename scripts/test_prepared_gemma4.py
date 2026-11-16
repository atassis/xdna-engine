"""Small offline Gemma4 prepared-binding tests; no compiler, SDK, or model payload runs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import model_artifact_dispatch as dispatch
import model_artifact_stages as stages
import prepared_install as prepared
from test_verified_gemma4_data import Gemma4PublishedDataTests


REPO = Path(__file__).resolve().parents[1]
SCENARIO = 'scenarios/generate-gemma4-12b-resident-256k.toml'


class Gemma4PreparedBindingTests(unittest.TestCase):
    def entry(self):
        return {'variant': 'verified-data', 'recipe_key': 'a' * 64,
                'source_root': '/retired/accepted/gemma4-12b',
                'witness': {'sha256': 'b' * 64}}

    def test_v2_gate_is_strict_and_does_not_send_g4_to_g3_spec(self):
        self.assertEqual(prepared.data_variant_gate({'version': 2, 'data_variants': {'gemma4-12b': self.entry()}}),
                         {'gemma4-12b': self.entry()})
        for version, entry in ((True, self.entry()), (2.0, self.entry()), (2, {**self.entry(), 'extra': 1}),
                               (2, {**self.entry(), 'source_root': '/a/../gemma4-12b'}),
                               (1, self.entry())):
            with self.subTest(version=version, entry=entry):
                with self.assertRaises(ValueError):
                    prepared.data_variant_gate({'version': version, 'data_variants': {'gemma4-12b': entry}})

    def test_stage_runner_receipt_and_prepared_verification_use_published_g4_roots(self):
        """A tiny published-data fixture, with only native/package boundaries synthetic.

        The proof command, StageRunner source snapshots and receipt are maintained code.  The
        native boundary only writes its seven declared files; the package boundary copies the
        independently verified tiny published resident/tokenizer/checkpoint fixture.
        """
        tmpdir = Path(os.environ['TMPDIR'])
        # This test owns the explicitly named, bounded fixture parent.
        tmpdir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='g4-prepared-', dir=tmpdir) as temporary:
            root = Path(temporary)
            accepted_lock_bytes = (REPO / 'toolchain.lock').read_bytes()
            accepted_lock_sha = hashlib.sha256(accepted_lock_bytes).hexdigest()
            iron_commit = next(line.split('=', 1)[1].strip() for line in accepted_lock_bytes.decode().splitlines()
                               if line.startswith('IRON_SOURCE_COMMIT='))
            published = Gemma4PublishedDataTests('run')
            published.setUp()
            self.addCleanup(published.doCleanups)
            owner, source = published.fixture.owner, published.fixture.root
            witness = root / 'review' / 'witness.json'
            witness.parent.mkdir(parents=True)
            witness.write_bytes(published.fixture.sidecar_raw)
            pin = hashlib.sha256(witness.read_bytes()).hexdigest()
            artifacts, build = root / 'artifacts', root / 'build'
            historical_repo = root / 'historical-repo'
            historical_repo.symlink_to(REPO, target_is_directory=True)
            custom_sdk = root / 'synthetic-sdk'
            (custom_sdk / 'bin').mkdir(parents=True)
            custom_python = custom_sdk / 'bin/python'
            custom_python.symlink_to(REPO / '.venv-iron/bin/python')
            self.enterContext(patch.dict(os.environ, {'MODEL_IRON_PY': str(custom_python)}))
            recipe = dispatch.gemma4_verified_recipe(SCENARIO, historical_repo, artifacts, root / 'inputs', build,
                                                      witness=witness, witness_sha256=pin,
                                                      data_source=source, accepted_data_root=owner)
            calls = []

            def execute(staged, repo):
                command = staged.commands[0]
                calls.append(command.step)
                if command.step == dispatch.VERIFIED_DATA_STEP:
                    subprocess.run(command.argv, check=True, cwd=repo,
                                   env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1',
                                                     'PYTHONPATH': str(REPO / 'scripts')})
                    return
                if command.step == 'build':
                    output = Path(command.stage.outputs[0].root)
                    output.mkdir(parents=True)
                    for name in ('design.elf.zst', 'full_elf_config.json', 'fwd_layout.json', 'pack.json',
                                 'gen_args.txt', 'gen_env.txt', 'params.txt'):
                        (output / name).write_text('synthetic native fixture\n')
                    cache = Path(command.env['RF_BUILD']) / 'kobj/run-fixture'
                    cache.mkdir(parents=True)
                    (cache / 'fixture.o').write_bytes(b'synthetic retained object\n')
                    (output.parent / 'meta.json').write_text(json.dumps({'compiled_by': 'synthetic native fixture'}))
                    return
                destinations = tuple(Path(view.root) for view in command.stage.outputs)
                for fixture_root, destination in zip((published.resident, published.tokenizer, published.checkpoint),
                                                     destinations, strict=True):
                    shutil.copytree(fixture_root, destination)
                native_root = Path(command.argv[command.argv.index('--build-dir') + 1]) / 'rf48C'
                for name in ('design.elf.zst', 'pack.json', 'fwd_layout.json', 'params.txt', 'gen_args.txt'):
                    shutil.copy2(native_root / name, destinations[0] / name)
                # The serving metadata is a plainly labelled package-boundary fixture; the
                # published-data verifier below still reads the copied real fixture roots.
                (destinations[0] / 'meta.json').write_text(json.dumps({
                    'spec': 'gemma4-12b', 'elf': 'design.elf', 'compiled_by': 'synthetic package fixture',
                    'fixture_artifact_root': str(destinations[0]),
                    'iron': {'commit': iron_commit, 'dirty': False, 'identity': 'pinned:' + iron_commit}}))

            def fixture_sdk_identity(python, packages):
                return json.dumps({package: {'version': '0+fixture',
                                             'record': hashlib.sha256(('fixture:' + package).encode()).hexdigest()}
                                   for package in packages}, sort_keys=True)

            def fixture_toolchain_identity(item, repo):
                instance = root / 'synthetic-instance'
                venv = Path(item.env['VENV_IRON'])
                peano = venv / 'lib/python3.14/site-packages/llvm-aie'
                locations = (instance / 'bin/fixture', instance / 'python/fixture',
                             instance / 'build/include/fixture', instance / 'build/lib/fixture',
                             peano / 'bin/fixture', peano / 'include/fixture', peano / 'lib/fixture')
                return {'instance': str(instance), 'peano': str(peano),
                        'files': {str(location): {'sha256': hashlib.sha256(str(location).encode()).hexdigest()}
                                  for location in locations}}

            with patch.object(dispatch, 'LOCK_ROOT', root / 'locks'), \
                 patch.object(dispatch, 'run_commands', side_effect=execute), \
                 patch.object(stages, 'sdk_identity', side_effect=fixture_sdk_identity), \
                 patch.object(stages, 'toolchain_identity', side_effect=fixture_toolchain_identity):
                result = dispatch.build_recipe(recipe, historical_repo, historical_repo / SCENARIO, artifacts, build)
            self.assertEqual(result, 'BUILT')
            _stage, state = dispatch.stage_locations(recipe, artifacts, build)
            kept = state / 'frontiers/build/build/gemma4-12b/rf48C/kobj/run-fixture/fixture.o'
            self.assertEqual(kept.read_bytes(), b'synthetic retained object\n')
            self.assertEqual(calls, ['verified-data', 'build', 'package'])
            record = json.loads(dispatch.manifest_path(recipe).read_text())
            self.assertEqual(set(record['stages']), {'verified-data', 'build', 'package'})
            self.assertEqual(record['stages']['verified-data']['producer']['inputs'][str(witness)], pin)
            package_meta = next(name for name in record['stages']['package']['files'] if name.endswith('/meta.json'))
            self.assertNotEqual(record['stages']['package']['producer_outputs'][package_meta]['sha256'],
                                record['stages']['package']['files'][package_meta]['sha256'])
            self.assertEqual(record['stages']['package']['producer_outputs'][package_meta]['identity'],
                             record['stages']['package']['files'][package_meta]['identity'])
            self.assertEqual(record['stages']['package']['producer_outputs'][package_meta]['executable'],
                             record['stages']['package']['files'][package_meta]['executable'])
            # This is the real offline verifier, not the package seam.
            entry = {'variant': 'verified-data', 'recipe_key': record['recipe_key'],
                     'source_root': str(source), 'witness': {'sha256': pin}}
            self.assertIsNone(prepared._verify_g4_variant(recipe, record, entry))
            # Rehashing a forged receipt must not bless a changed structural declaration.
            def forge_frontier(value, step, operation):
                stage = value['stages'][step]
                operation(stage['files'])
                stage['producer_outputs'] = copy.deepcopy(stage['files'])
                stage['output_identity'] = dispatch.stage_digest({
                    name: [fingerprint['identity'], fingerprint['executable']]
                    for name, fingerprint in stage['files'].items()})

            def foreign_package_file(files):
                name = next(name for name in files if not name.endswith('/meta.json'))
                files['artifacts/foreign/file'] = files.pop(name)

            def false_file_hash(files, suffix):
                name = next(name for name in files if name.endswith(suffix))
                files[name].update(sha256='0' * 64, identity='0' * 64)

            mutations = {
                'boolean stage version': lambda value: value['stages']['verified-data'].update(version=True),
                'empty proof input views': lambda value: value['commands'][0]['stage'].update(inputs=[]),
                'foreign proof helper': lambda value: value['commands'][0]['argv'].__setitem__(1, '/foreign/evil.py'),
                'foreign accepted owner': lambda value: value['commands'][0]['argv'].__setitem__(6, '/foreign/owner'),
                'foreign package output': lambda value: value['commands'][2]['argv'].__setitem__(16, '/foreign/output'),
                'noncompiler native command': lambda value: value['commands'][1].update(argv=['bash', '/retired/copy-old-program.sh']),
                # Parent receipt source/output counterexamples.  Each receipt is rehashed below,
                # so a rejection must be semantic rather than merely pin mismatch.
                'native producer no consumed inputs': lambda value: value['stages']['build']['producer'].update(inputs={}),
                'package producer no consumed inputs': lambda value: value['stages']['package']['producer'].update(inputs={}),
                'native producer no source hashes': lambda value: value['stages']['build']['producer'].update(sources={}),
                'package producer no source hashes': lambda value: value['stages']['package']['producer'].update(sources={}),
                'unbound proof output identity': lambda value: value['stages']['verified-data'].update(output_identity='0' * 64),
                'numeric native view require_each': lambda value: value['commands'][1]['stage']['outputs'][0].update(require_each=1),
                'undeclared proof RF env': lambda value: value['commands'][0]['env'].update(RF_STORE='/foreign/store'),
                'accepted source lock mismatch': lambda value: value['stages']['build']['producer']['sources'].update({'toolchain.lock': '0' * 64}),
                'numeric fingerprint executable': lambda value: next(iter(value['stages']['package']['files'].values())).update(executable=1),
                'producer manifest divergence': lambda value: next(iter(value['stages']['build']['producer_outputs'].values())).update(identity='0' * 64),
                'foreign captured environment': lambda value: value['stages']['verified-data']['producer']['environment'].update(FOREIGN='1'),
                'conflicting captured environment': lambda value: value['stages']['build']['producer']['environment'].update(PYTHONPATH='/foreign'),
                'foreign native symbols': lambda value: value['stages']['build']['producer'].update(symbols={'foreign': 'def foreign(): pass'}),
                'missing package symbols': lambda value: value['stages']['package']['producer'].update(symbols={}),
                'missing compiler SDK': lambda value: value['stages']['build']['producer'].update(sdk=None),
                'missing compiler toolchain': lambda value: value['stages']['build']['producer'].update(toolchain=None),
                'package SDK executable mismatch': lambda value: value['stages']['package']['producer']['sdk'].update(sha256='0' * 64),
                'foreign package manifest path': lambda value: forge_frontier(value, 'package', foreign_package_file),
                'omitted proof output': lambda value: forge_frontier(value, 'verified-data', lambda files: files.clear()),
                'proof frontier pin mismatch': lambda value: forge_frontier(value, 'verified-data', lambda files: false_file_hash(files, prepared.gemma4.SIDECAR)),
                'native output missing': lambda value: forge_frontier(value, 'build', lambda files: files.pop(next(iter(files)))),
                'packaged native program mismatch': lambda value: forge_frontier(value, 'package', lambda files: false_file_hash(files, '/design.elf.zst')),
                'false live tokenizer fingerprint': lambda value: forge_frontier(value, 'package', lambda files: false_file_hash(files, '/tokenizer/tokenizer.json')),
            }
            receipt = dispatch.manifest_path(recipe)
            saved = receipt.read_bytes()
            try:
                for label, mutate in mutations.items():
                    with self.subTest(label=label):
                        altered = copy.deepcopy(record)
                        mutate(altered)
                        for command in altered['commands']:
                            stage = altered['stages'][command['step']]
                            stage['producer']['command'] = {key: command[key] for key in ('step', 'argv', 'env')}
                            stage['producer_source'] = dispatch.stage_digest(stage['producer'])
                            dependencies = {dep: {'key': altered['stages'][dep]['key'],
                                                   'outputs': altered['stages'][dep]['output_identity']}
                                            for dep in command['stage']['dependencies']}
                            stage['key'] = dispatch.stage_digest({'version': dispatch.StageRunner.VERSION,
                                                                  'source': stage['producer_source'],
                                                                  'dependencies': dependencies})
                        raw = json.dumps(altered).encode()
                        receipt.write_bytes(raw)
                        revised = {'recipe_key': entry['recipe_key'],
                                   'sha256': hashlib.sha256(raw).hexdigest()}
                        with self.assertRaises(ValueError):
                            checked = prepared._g4_receipt(recipe, revised, entry, accepted_lock_sha)
                            prepared._verify_g4_variant(recipe, checked, entry)
            finally:
                receipt.write_bytes(saved)
            shutil.rmtree(owner)
            witness.unlink()
            self.assertFalse(source.exists())
            self.assertIsNone(prepared._verify_g4_variant(recipe, record, entry))
            # Exercise the public prepared verifier with fake CLIs only at its native-check
            # boundary.  The Gemma4 receipt and published-data verifier remain unmocked.
            repo = root / 'prepared-repo'
            (repo / 'scenarios').mkdir(parents=True)
            shutil.copy2(REPO / SCENARIO, repo / SCENARIO)
            lock = repo / 'toolchain.lock'
            lock.write_bytes(accepted_lock_bytes)
            normalized_lock = '\n'.join(line for raw in lock.read_text().splitlines()
                                        if (line := __import__('re').sub(r'\s*#.*$', '', raw)).strip())
            kernel_pin = hashlib.sha256((normalized_lock + '\n').encode()).hexdigest()[:12]
            kernels = root / 'kernels'; (kernels / 'whole_array').mkdir(parents=True)
            for directory in (kernels, kernels / 'whole_array'):
                (directory / '.toolchain-stamp').write_text(kernel_pin)
            config = root / 'engine.toml'
            config.write_text('[[model]]\nname="gemma4-12b"\nscenario="' + SCENARIO + '"\n')
            native = {}
            for name in ('npu', 'npu_dev', 'libonnxruntime.so.1.26.0'):
                path = root / name
                path.write_text('#!/bin/sh\nexit 0\n')
                path.chmod(0o755)
                native['onnxruntime' if name.startswith('lib') else name] = path
            manifest_data = {'version': 2, 'artifacts': str(artifacts),
                             'kernels': {'path': str(kernels), 'sha256': dispatch.tree_digest(kernels)},
                             'scenarios': {SCENARIO: prepared.sha256_file(repo / SCENARIO)},
                             'receipts': {'gemma4-12b': {'sha256': prepared.sha256_file(dispatch.manifest_path(recipe)),
                                                         'recipe_key': record['recipe_key']}},
                             'data_variants': {'gemma4-12b': entry}}
            manifest_data.update({name: {'path': str(path), 'sha256': prepared.sha256_file(path)}
                                  for name, path in (native | {'config': config, 'toolchain': lock}).items()})
            manifest = root / 'prepared.json'; manifest.write_text(json.dumps(manifest_data))
            # The retained synthetic native/proof state is not an input to prepared verification.
            shutil.rmtree(build)
            historical_repo.unlink()
            shutil.rmtree(custom_sdk)
            # This reaches the public verifier's Gemma4 three-root receipt branch.  Only its
            # final native CLI checks are fake executables; receipt, payload and verifier stay real.
            self.assertEqual(prepared.verify(manifest, repo)['ENGINE_ARTIFACTS'], str(artifacts))
            program_params = Path(recipe.outputs[0]) / 'params.txt'
            params_bytes = program_params.read_bytes()
            try:
                program_params.write_bytes(params_bytes + b'changed')
                with self.assertRaisesRegex(ValueError, 'published files differ'):
                    prepared._verify_g4_variant(recipe, record, entry)
            finally:
                program_params.write_bytes(params_bytes)
            manifest_bytes = manifest.read_bytes()
            try:
                lock.write_bytes(accepted_lock_bytes + b'\n# mismatched captured lock\n')
                mismatched = copy.deepcopy(manifest_data)
                mismatched['toolchain']['sha256'] = prepared.sha256_file(lock)
                manifest.write_text(json.dumps(mismatched))
                with self.assertRaisesRegex(ValueError, 'native source lock'):
                    prepared.verify(manifest, repo)
            finally:
                lock.write_bytes(accepted_lock_bytes)
                manifest.write_bytes(manifest_bytes)
            sidecar = Path(recipe.outputs[0]) / prepared.gemma4.SIDECAR
            sidecar_bytes = sidecar.read_bytes()
            try:
                omitted = copy.deepcopy(manifest_data)
                omitted['version'] = 1
                omitted.pop('data_variants')
                manifest.write_text(json.dumps(omitted))
                with self.assertRaisesRegex(ValueError, 'marker lacks'):
                    prepared.verify(manifest, repo)
                sidecar.unlink()
                with self.assertRaisesRegex(ValueError, 'marker lacks'):
                    prepared.verify(manifest, repo)
                sidecar.write_bytes(sidecar_bytes)
                orphan = copy.deepcopy(record)
                orphan['stages'].pop('verified-data')
                receipt.write_text(json.dumps(orphan))
                omitted['receipts']['gemma4-12b']['sha256'] = prepared.sha256_file(receipt)
                manifest.write_text(json.dumps(omitted))
                with self.assertRaisesRegex(ValueError, 'marker lacks'):
                    prepared.verify(manifest, repo)
            finally:
                sidecar.write_bytes(sidecar_bytes)
                receipt.write_bytes(saved)
                manifest.write_bytes(manifest_bytes)
            self.assertEqual(prepared.verify(manifest, repo)['ENGINE_ARTIFACTS'], str(artifacts))
            altered = json.loads(json.dumps(record))
            altered['stages']['verified-data']['producer']['inputs'][str(witness)] = '0' * 64
            altered['stages']['verified-data']['producer_source'] = dispatch.stage_digest(
                altered['stages']['verified-data']['producer'])
            altered['stages']['verified-data']['key'] = dispatch.stage_digest({
                'version': dispatch.StageRunner.VERSION,
                'source': altered['stages']['verified-data']['producer_source'], 'dependencies': {}})
            with self.assertRaisesRegex(ValueError, 'did not consume exactly'):
                prepared._verify_g4_variant(recipe, altered, entry)

    def test_pinned_receipt_rejects_command_or_dependency_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = tuple(str(root / name) for name in ('resident', 'tokenizer', 'checkpoint'))
            for output in outputs:
                Path(output).mkdir()
                (Path(output) / 'payload').write_text(output)
            recipe = dispatch.Recipe('gemma4-12b', 'scenarios/generate-gemma4-12b-resident-256k.toml', (), outputs, ())
            entry = self.entry()
            source, pin = entry['source_root'], entry['witness']['sha256']
            commands = [
                {'step': 'verified-data', 'argv': ['/historic/python', '/historic/gemma4_verified_entry.py', 'proof', '--source', source, '--accepted-data-root', '/retired/accepted', '--witness', '/review/witness', '--expected-witness-sha256', pin, '--out', '/build/proof'], 'env': {}, 'stage': {'dependencies': []}},
                {'step': 'build', 'argv': ['/bin/bash', '/historic/rf48C.sh'], 'env': {'RF_STORE': source + '/store', 'RF_HF_CONFIG': source + '/hf_config/config.json'}, 'stage': {'dependencies': ['verified-data']}},
                {'step': 'package', 'argv': ['/historic/python', '/historic/gemma4_verified_entry.py', 'package', '--source', source, '--accepted-data-root', '/retired/accepted', '--proof-root', '/build/proof', '--expected-witness-sha256', pin, '--artifacts-root', '/artifacts', '--build-dir', '/build/rf48C', '--out', outputs[0], '--tokenizer-out', outputs[1]], 'env': {}, 'stage': {'dependencies': ['verified-data', 'build']}},
            ]
            stages = {}
            identities = {}
            for command in commands:
                step = command['step']
                deps = {} if step == 'verified-data' else ({'verified-data': {'key': identities['verified-data'][0], 'outputs': identities['verified-data'][1]}} if step == 'build' else {'verified-data': {'key': identities['verified-data'][0], 'outputs': identities['verified-data'][1]}, 'build': {'key': identities['build'][0], 'outputs': identities['build'][1]}})
                producer_command = {key: command[key] for key in ('step', 'argv', 'env')}
                sources = ({'scripts/gemma4_verified_entry.py': 'c' * 64,
                            'scripts/verified_gemma4_data.py': 'd' * 64,
                            'scripts/verified_gemma_data.py': 'e' * 64}
                           if step == 'verified-data' else {'historic/' + step: 'c' * 64})
                producer = {'command': producer_command, 'sources': sources,
                            'inputs': {}, 'sdk': None, 'toolchain': None}
                producer_source = dispatch.stage_digest(producer)
                key = dispatch.stage_digest({'version': dispatch.StageRunner.VERSION, 'source': producer_source,
                                             'dependencies': deps})
                stages[step] = {'version': dispatch.StageRunner.VERSION, 'stage': step,
                                'action': 'executed', 'producer': producer, 'producer_source': producer_source,
                                'key': key, 'output_identity': step + '-out'}
                identities[step] = (key, step + '-out')
            record = {'model': recipe.model, 'scenario': recipe.scenario, 'recipe_key': entry['recipe_key'],
                      'output_digest': dispatch.output_digest(recipe.outputs), 'commands': commands, 'stages': stages}
            receipt = dispatch.manifest_path(recipe)
            receipt.write_text(json.dumps(record))
            witness = {'recipe_key': entry['recipe_key'], 'sha256': hashlib.sha256(receipt.read_bytes()).hexdigest()}
            # This deliberately incomplete metadata-only receipt has no actual factory stage
            # contracts and must not become a compatibility bypass for structural verification.
            with self.assertRaises(ValueError):
                prepared._g4_receipt(recipe, witness, entry, 'f' * 64)


if __name__ == '__main__':
    unittest.main()
