#!/usr/bin/env python3
"""CPU-only stage lifecycle controls; no model, toolchain, service or device work."""
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import model_artifact_dispatch as dispatch
from model_artifact_stages import Stage, StageRunner, View


class StageLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.artifacts, self.build = self.root / 'artifacts', self.root / 'build'
        self.workspace = self.build / 'transaction'
        self.state = self.build / 'completed'
        self.source = self.root / 'checkpoint'
        self.source.mkdir()
        (self.source / 'weight.bin').write_bytes(b'weights')
        (self.source / 'tokenizer.json').write_text('{}')
        self.calls = []
        self.fail = None
        self.producers = {}
        for name in ('weights', 'compile', 'package', 'tokenizer'):
            producer = self.root / (name + '.py')
            producer.write_text(name)
            self.producers[name] = str(producer)
        out = self.artifacts
        self.commands = (
            dispatch.Command('weights', ('fixture',), {}, Stage(
                inputs=(View(str(self.source), ('weight.bin',)),),
                outputs=(View(str(out / 'weights')),), sources=(self.producers['weights'],))),
            dispatch.Command('compile', ('fixture',), {}, Stage(
                dependencies=('weights',), outputs=(View(str(self.build / 'compiler'), ('design.elf', 'layout.json')),),
                sources=(self.producers['compile'],), retain=True)),
            dispatch.Command('package', ('fixture',), {}, Stage(
                dependencies=('compile',), outputs=(View(str(out / 'model')),), sources=(self.producers['package'],))),
            dispatch.Command('tokenizer', ('fixture',), {}, Stage(
                inputs=(View(str(self.source), ('tokenizer.json',)),), outputs=(View(str(out / 'tokenizer')),),
                sources=(self.producers['tokenizer'],))),
        )
        self.recipe = dispatch.Recipe('fixture', 'scenario.toml', (str(self.source),),
                                     (str(out / 'model'), str(out / 'tokenizer'), str(out / 'weights')), self.commands)

    def execute(self, recipe, repo):
        name = recipe.commands[0].step
        self.calls.append(name)
        if name == self.fail:
            raise subprocess.CalledProcessError(17, name)
        for view in recipe.commands[0].stage.outputs:
            target = Path(view.root)
            target.mkdir(parents=True, exist_ok=True)
            for rel in view.patterns or ('value.bin',):
                (target / rel).write_bytes(name.encode())
        if name == 'compile':
            (target / 'huge-disposable.ll').write_text('temporary')
        if name == 'package':
            # Packaging must not invalidate the compiler frontier.
            compiler = self.workspace / 'build/compiler'
            (compiler / 'meta.json').write_text('{}')

    def runner(self):
        staged = dispatch.rebase_recipe(self.recipe, self.artifacts, self.workspace / 'artifacts',
                                        self.build, self.workspace / 'build')
        return StageRunner(self.recipe, staged, self.root, self.workspace, self.state,
                           self.artifacts, self.build, self.execute)

    def publish(self, runner):
        dispatch.relocate_recipe_metadata(self.recipe, runner.staged)
        runner.prepare_settlement()
        for raw, fresh in zip(self.recipe.outputs, runner.staged.outputs, strict=True):
            destination = Path(raw)
            if destination.exists():
                import shutil
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Path(fresh).rename(destination)
        runner.settle()
        import shutil
        shutil.rmtree(self.workspace)

    def dispatched_build(self):
        (self.root / 'scenario.toml').write_text('fixture scenario')
        config = self.root / 'engine.toml'
        config.write_text('fixture deployment')
        self.workspace, self.state = dispatch.stage_locations(self.recipe, self.artifacts, self.build)
        with patch('model_artifact_dispatch.LOCK_ROOT', self.root / 'locks'), \
             patch('model_artifact_dispatch.run_commands', side_effect=self.execute):
            return dispatch.build_recipe(self.recipe, self.root, config, self.artifacts, self.build)

    def test_dispatcher_cold_resume_publish_and_noop(self):
        self.fail = 'package'
        with self.assertRaises(subprocess.CalledProcessError):
            self.dispatched_build()
        original_production = json.loads((self.state / 'weights.json').read_text())['production']
        self.calls.clear()
        self.fail = None
        self.assertEqual(self.dispatched_build(), 'BUILT')
        self.assertEqual(self.calls, ['package', 'tokenizer'])
        self.assertFalse(self.workspace.exists())
        self.assertFalse((self.state / 'pending.json').exists())
        receipt = dispatch.manifest_path(self.recipe).read_text()
        self.assertNotIn('model-artifact-staging', receipt)
        stages = json.loads(receipt)['stages']
        self.assertEqual(stages['weights']['production'], original_production)
        self.assertEqual(stages['weights']['action'], 'verified-reuse')
        self.assertEqual(stages['compile']['action'], 'verified-reuse')
        self.assertEqual(stages['package']['action'], 'executed')
        self.assertNotIn('prepared_locations', stages['compile'])
        self.calls.clear()
        self.assertEqual(self.dispatched_build(), 'HIT')
        self.assertEqual(self.calls, [])
        Path(self.producers['package']).write_text('package-only change')
        self.assertEqual(self.dispatched_build(), 'BUILT')
        self.assertEqual(self.calls, ['package'])

    def test_state_owner_alone_cannot_adopt_a_historical_receipt(self):
        self.assertEqual(self.dispatched_build(), 'BUILT')
        path = dispatch.manifest_path(self.recipe)
        receipt = json.loads(path.read_text())
        receipt.pop('stages')
        path.write_text(json.dumps(receipt))
        Path(self.producers['package']).write_text('different producer')
        self.calls.clear()
        with self.assertRaisesRegex(RuntimeError, 'historical receipt'):
            self.dispatched_build()
        self.assertEqual(self.calls, [])

    def test_successful_publication_recovers_interrupted_settlement(self):
        with patch.object(StageRunner, 'settle', side_effect=OSError('fixture settlement interruption')):
            self.assertEqual(self.dispatched_build(), 'BUILT')
        self.assertTrue(self.workspace.exists())
        self.assertTrue((self.state / 'pending.json').is_file())
        self.calls.clear()
        self.assertEqual(self.dispatched_build(), 'HIT')
        self.assertEqual(self.calls, [])
        self.assertFalse((self.state / 'pending.json').exists())
        self.assertFalse(self.workspace.exists())
        self.assertFalse(dispatch.stage_witness_path(self.workspace).exists())

    def test_partial_compact_frontier_promotion_resumes_after_verified_publication(self):
        import os
        self.assertEqual(self.dispatched_build(), 'BUILT')
        Path(self.producers['compile']).write_text('new compiler source')
        execute = self.execute
        def changed(recipe, repo):
            execute(recipe, repo)
            if recipe.commands[0].step == 'compile':
                for name in ('design.elf', 'layout.json'):
                    (self.workspace / 'build/compiler' / name).write_bytes(b'new compiler output')
        self.execute = changed
        replace_file = os.replace
        def interrupt(source, target):
            if 'pending-frontiers' in str(source) and Path(target).name == 'layout.json':
                raise OSError('fixture partial promotion interruption')
            return replace_file(source, target)
        with patch('model_artifact_stages.os.replace', side_effect=interrupt):
            self.assertEqual(self.dispatched_build(), 'BUILT')
        self.assertTrue((self.state / 'pending.json').exists())
        self.calls.clear()
        self.assertEqual(self.dispatched_build(), 'HIT')
        self.assertEqual(self.calls, [])
        self.assertFalse(self.workspace.exists())
        self.assertFalse((self.state / 'pending.json').exists())
        for name in ('design.elf', 'layout.json'):
            self.assertEqual((self.state / 'frontiers/compile/build/compiler' / name).read_bytes(), b'new compiler output')

    def test_unresolved_publication_cannot_be_retried_under_another_key(self):
        with patch('model_artifact_dispatch._publish_recipe', side_effect=OSError('fixture publication interruption')):
            with self.assertRaises(OSError):
                self.dispatched_build()
        Path(self.producers['package']).write_text('different key')
        self.calls.clear()
        with self.assertRaisesRegex(RuntimeError, 'recovery requires inspection'):
            self.dispatched_build()
        self.assertEqual(self.calls, [])

    def test_late_failure_retains_verified_predecessors(self):
        self.fail = 'package'
        with self.assertRaises(subprocess.CalledProcessError):
            self.runner().run()
        self.assertEqual(self.calls, ['weights', 'compile', 'package'])
        self.calls.clear()
        self.fail = None
        self.runner().run()
        self.assertEqual(self.calls, ['package', 'tokenizer'])

    def test_preparing_new_publication_preserves_the_accepted_compiler_frontier(self):
        first = self.runner()
        first.run()
        self.publish(first)
        accepted = self.state / 'frontiers/compile/build/compiler/design.elf'
        original = accepted.read_bytes()
        Path(self.producers['compile']).write_text('new compiler source')
        execute = self.execute
        def changed(recipe, repo):
            execute(recipe, repo)
            if recipe.commands[0].step == 'compile':
                (self.workspace / 'build/compiler/design.elf').write_bytes(b'new compiler output')
        self.execute = changed
        candidate = self.runner()
        candidate.run()
        candidate.prepare_settlement('candidate')
        self.assertEqual(accepted.read_bytes(), original)

    def test_compiler_frontier_survives_publication_without_intermediates(self):
        first = self.runner()
        first.run()
        self.publish(first)
        self.assertFalse(list(self.state.rglob('huge-disposable.ll')))
        self.assertEqual(len(list(self.state.rglob('design.elf'))), 1)
        self.calls.clear()
        Path(self.producers['package']).write_text('package changed')
        self.runner().run()
        self.assertEqual(self.calls, ['package'])

    def test_tokenizer_only_edit_does_not_run_weights_or_compiler(self):
        first = self.runner()
        first.run()
        self.publish(first)
        self.calls.clear()
        (self.source / 'tokenizer.json').write_text('{"changed": true}')
        self.runner().run()
        self.assertEqual(self.calls, ['tokenizer'])

    def test_mutated_weight_invalidates_its_transitive_consumers_only(self):
        first = self.runner()
        first.run()
        self.publish(first)
        self.calls.clear()
        (self.source / 'weight.bin').write_bytes(b'changed')
        self.runner().run()
        self.assertEqual(self.calls, ['weights', 'compile', 'package'])

    def test_changed_compiler_does_not_rerun_weights(self):
        first = self.runner()
        first.run()
        self.publish(first)
        self.calls.clear()
        Path(self.producers['compile']).write_text('new compiler')
        self.runner().run()
        self.assertEqual(self.calls, ['compile', 'package'])

    def test_corrupt_frontier_cannot_be_reused(self):
        first = self.runner()
        first.run()
        self.publish(first)
        next(self.state.rglob('design.elf')).write_bytes(b'corrupt')
        self.calls.clear()
        Path(self.producers['package']).write_text('new package')
        self.runner().run()
        self.assertEqual(self.calls, ['compile', 'package'])

    def test_missing_published_companion_prevents_reuse(self):
        first = self.runner()
        first.run()
        self.publish(first)
        (self.artifacts / 'tokenizer/value.bin').unlink()
        self.calls.clear()
        self.runner().run()
        self.assertEqual(self.calls, ['tokenizer'])

    def test_noop_uses_verified_outputs_and_creates_no_new_frontier_variant(self):
        first = self.runner()
        first.run()
        self.publish(first)
        before = {str(p.relative_to(self.state)) for p in self.state.rglob('*')}
        self.calls.clear()
        second = self.runner()
        second.run()
        self.assertEqual(self.calls, [])
        self.publish(second)
        self.assertEqual(before, {str(p.relative_to(self.state)) for p in self.state.rglob('*')})

    def test_unowned_workspace_is_not_deleted(self):
        self.workspace.mkdir(parents=True)
        kept = self.workspace / 'unique'
        kept.write_bytes(b'keep')
        with self.assertRaisesRegex(RuntimeError, 'unowned'):
            self.runner().run()
        self.assertEqual(kept.read_bytes(), b'keep')

    def test_unknown_or_symlinked_state_is_not_overwritten(self):
        self.state.mkdir(parents=True)
        external = self.root / 'external.json'
        external.write_text('{}')
        (self.state / 'owner.json').symlink_to(external)
        with self.assertRaises((RuntimeError, ValueError)):
            self.runner().run()
        self.assertEqual(external.read_text(), '{}')

    def test_output_overlap_is_rejected(self):
        commands = list(self.commands)
        commands[1] = replace(commands[1], stage=replace(commands[1].stage, outputs=self.commands[0].stage.outputs))
        self.recipe = replace(self.recipe, commands=tuple(commands))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.runner().run()
        self.assertEqual(self.calls, [])

    def test_missing_dependency_or_cycle_is_rejected_before_execution(self):
        for dependencies in (('absent',), ('package',)):
            with self.subTest(dependencies=dependencies):
                commands = (replace(self.commands[0], stage=replace(self.commands[0].stage, dependencies=dependencies)), *self.commands[1:])
                original = self.recipe
                self.recipe = replace(original, commands=commands)
                with self.assertRaises(ValueError):
                    self.runner().run()
                self.recipe = original
        self.assertEqual(self.calls, [])

    def test_input_mutation_during_execution_cannot_complete(self):
        execute = self.execute
        def mutate(recipe, repo):
            execute(recipe, repo)
            if recipe.commands[0].step == 'weights':
                (self.source / 'weight.bin').write_bytes(b'mutated during run')
        self.execute = mutate
        with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
            self.runner().run()
        self.assertFalse((self.state / 'weights.json').exists())

    def test_private_restore_protects_live_data_from_a_bad_consumer(self):
        first = self.runner()
        first.run()
        self.publish(first)
        original = (self.artifacts / 'weights/value.bin').read_bytes()
        Path(self.producers['package']).write_text('new package')
        execute = self.execute
        def mutate(recipe, repo):
            execute(recipe, repo)
            if recipe.commands[0].step == 'package':
                (self.workspace / 'artifacts/weights/value.bin').write_bytes(b'bad write')
        self.execute = mutate
        with self.assertRaisesRegex(RuntimeError, 'mutated predecessor'):
            self.runner().run()
        self.assertEqual((self.artifacts / 'weights/value.bin').read_bytes(), original)

    def test_symlinked_state_parent_cannot_create_external_records(self):
        outside = self.root / 'outside'
        outside.mkdir()
        self.build.mkdir()
        (self.build / 'redirect').symlink_to(outside, target_is_directory=True)
        self.state = self.build / 'redirect/completed'
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.runner().run()
        self.assertEqual(list(outside.iterdir()), [])

    def test_patterned_output_cannot_overlap_another_owner(self):
        command = replace(self.commands[1], stage=replace(self.commands[1].stage,
                          outputs=(View(str(self.artifacts / 'weights'), ('*.bin',)),)))
        weights = replace(self.commands[0], stage=replace(self.commands[0].stage,
                          outputs=(View(str(self.artifacts / 'weights'), ('value.bin',)),)))
        self.recipe = replace(self.recipe, commands=(weights, command, *self.commands[2:]))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.runner().run()
        self.assertEqual(self.calls, [])

    def test_scratch_cannot_clear_another_stage_output(self):
        command = replace(self.commands[1], stage=replace(self.commands[1].stage,
                          scratch=(str(self.artifacts / 'weights'),)))
        self.recipe = replace(self.recipe, commands=(self.commands[0], command, *self.commands[2:]))
        with self.assertRaisesRegex(ValueError, 'scratch overlaps'):
            self.runner().run()
        self.assertEqual(self.calls, [])

    def test_forged_location_cannot_read_or_overwrite_an_external_file(self):
        self.fail = 'package'
        with self.assertRaises(subprocess.CalledProcessError):
            self.runner().run()
        external = self.root / 'external'
        external.write_bytes(b'keep')
        path = self.state / 'weights.json'
        record = json.loads(path.read_text())
        rel = next(iter(record['files']))
        record['locations'][rel] = str(external)
        path.write_text(json.dumps(record))
        self.fail = None
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.runner().run()
        self.assertEqual(external.read_bytes(), b'keep')

    def test_parent_path_in_record_is_rejected_even_with_consistent_digest(self):
        self.fail = 'package'
        with self.assertRaises(subprocess.CalledProcessError):
            self.runner().run()
        path = self.state / 'weights.json'
        record = json.loads(path.read_text())
        original = next(iter(record['files']))
        unsafe = 'artifacts/../../outside'
        record['files'][unsafe] = record['files'].pop(original)
        record['locations'][unsafe] = record['locations'].pop(original)
        record['output_identity'] = self.runner()._output_identity(record['files'])
        path.write_text(json.dumps(record))
        self.fail = None
        with self.assertRaisesRegex(ValueError, 'unowned|unsafe'):
            self.runner().run()

    def test_stage_cannot_write_a_live_output_alias(self):
        self.fail = 'package'
        with self.assertRaises(subprocess.CalledProcessError):
            self.runner().run()
        staged_weight = self.workspace / 'artifacts/weights/value.bin'
        staged_weight.unlink()
        external = self.root / 'external'
        external.write_bytes(b'keep')
        staged_weight.symlink_to(external)
        self.fail = None
        with self.assertRaisesRegex((ValueError, RuntimeError), 'symlink'):
            self.runner().run()
        self.assertEqual(external.read_bytes(), b'keep')


class DeclaredStageTests(unittest.TestCase):
    def test_every_supported_recipe_declares_ordered_owned_stages(self):
        repo = Path(dispatch.__file__).resolve().parents[1]
        bindings = (
            ('parakeet', 'asr'), ('bge-base', 'bge-base'), ('bge-base-cls', 'bge-base-cls'),
            ('whisper-turbo', 'asr-whisper-turbo'), ('gemma3-270m', 'generate-gemma3-270m'),
            ('qwen3-0.6b', 'generate-qwen3-0.6b'), ('qwen3.5-4b', 'generate-qwen3.5-4b'),
            ('gemma4-12b', 'generate-gemma4-12b-resident-256k'),
            ('pyannote-community-1', 'diarize-pyannote-community-1'),
            ('pyannote-3.1', 'diarize-pyannote-3.1'), ('espcn', 'upscale-espcn'),
        )
        for model, scenario in bindings:
            with self.subTest(model=model):
                recipe = dispatch.recipe_for(model, f'scenarios/{scenario}.toml', repo,
                                             repo / 'artifacts', repo / 'inputs', repo / 'build')
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    isolated = dispatch.recipe_for(model, f'scenarios/{scenario}.toml', repo,
                                                   root / 'artifacts', root / 'inputs', root / 'build')
                    workspace, state = dispatch.stage_locations(isolated, root / 'artifacts', root / 'build')
                    staged = dispatch.rebase_recipe(isolated, root / 'artifacts', workspace / 'artifacts',
                                                     root / 'build', workspace / 'build')
                    StageRunner(isolated, staged, repo, workspace, state, root / 'artifacts', root / 'build', None)._validate()
                seen = set()
                for command in recipe.commands:
                    self.assertIsNotNone(command.stage)
                    self.assertLessEqual(set(command.stage.dependencies), seen)
                    for raw in command.stage.sources:
                        self.assertTrue((repo / raw).exists(), raw)
                    seen.add(command.step)

    def test_gemma_tokenizer_only_edit_does_not_invalidate_compilation(self):
        from model_artifact_stages import source_identity
        repo = Path(dispatch.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'inputs/gemma4-12b/checkpoint'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'config.json').write_text('{}')
            (checkpoint / 'tokenizer.json').write_text('{"tokenizer":1}')
            recipe = dispatch.recipe_for('gemma4-12b', 'scenarios/generate-gemma4-12b-resident-256k.toml',
                                         repo, root / 'artifacts', root / 'inputs', root / 'build')
            commands = {item.step: item for item in recipe.commands}
            self.assertNotIn('data-hf-config', commands['build'].stage.dependencies)
            self.assertNotIn('data-hf-config', commands['data-rf-stack'].stage.dependencies)
            with patch('model_artifact_stages.sdk_identity', return_value='fixture SDK'), \
                 patch('model_artifact_stages.toolchain_identity', return_value='fixture toolchain'):
                before = {name: source_identity(commands[name], repo, symbols=vars(dispatch))
                          for name in ('build', 'data-rf-stack', 'data-hf-config', 'package')}
                (checkpoint / 'tokenizer.json').write_text('{"tokenizer":2}')
                after = {name: source_identity(commands[name], repo, symbols=vars(dispatch)) for name in before}
                for name in ('build', 'data-rf-stack'):
                    self.assertEqual(before[name], after[name], name)
                for name in ('data-hf-config', 'package'):
                    self.assertNotEqual(before[name], after[name], name)

    def test_toolchain_fingerprint_binds_bytes_not_carrier_symlinks(self):
        from model_artifact_stages import toolchain_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instance, venv = root / 'instance', root / 'venv'
            peano = venv / 'lib/python3.14/site-packages/llvm-aie'
            for base, names in ((instance, ('bin/tool', 'python/sdk.py', 'build/include/sdk.h', 'build/lib/sdk.so')),
                                (peano, ('bin/clang', 'include/sdk.h', 'lib/sdk.so'))):
                for name in names:
                    path = base / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('fixture SDK')
            for name in ('carrier-a', 'carrier-b'):
                (root / name).mkdir()
                (root / name / 'header.h').write_text('same pinned header')
            link = instance / 'build/include/aie_api'
            link.symlink_to(root / 'carrier-a', target_is_directory=True)
            command = dispatch.Command('compile', ('fixture',), {'VENV_IRON': str(venv)}, Stage(toolchain=True))
            with patch.dict('os.environ', {'RF_INST': '', 'AIECC_PATH': '', 'PEANO_INSTALL_DIR': ''}), \
                 patch('model_artifact_stages.subprocess.check_output', return_value=str(instance)) as resolve:
                first = toolchain_identity(command, root)
                self.assertEqual(resolve.call_args.kwargs['env']['TOOLCHAIN_RESOLVE_ONLY'], '1')
                link.unlink()
                link.symlink_to(root / 'carrier-b', target_is_directory=True)
                self.assertEqual(first, toolchain_identity(command, root))
                (root / 'carrier-b/header.h').write_text('changed header')
                self.assertNotEqual(first, toolchain_identity(command, root))

    def test_unrelated_test_and_model_edits_do_not_invalidate_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'scripts').mkdir()
            (root / 'scripts/export_bge.py').write_text('producer')
            source = root / 'inputs/bge-base'
            source.mkdir(parents=True)
            (source / 'config.json').write_text('{}')
            recipe = dispatch.recipe_for('bge-base', 'scenarios/bge-base.toml', root,
                                         root / 'artifacts', root / 'inputs', root / 'build')
            with patch('model_artifact_stages.subprocess.check_output', return_value='fixture SDK'):
                first = dispatch.authority_digest(recipe, root)
                (root / 'scripts/test_other.py').write_text('changed test')
                (root / 'scripts/export_pyannote.py').write_text('another model')
                self.assertEqual(first, dispatch.authority_digest(recipe, root))
                (root / 'scripts/export_bge.py').write_text('changed producer')
                self.assertNotEqual(first, dispatch.authority_digest(recipe, root))

    def test_transient_launcher_path_does_not_invalidate_a_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            producer = root / 'producer.py'
            producer.write_text('producer')
            command = dispatch.Command('export', (sys.executable, str(producer)), {}, Stage(sources=(str(producer),)))
            from model_artifact_stages import source_identity
            with patch.dict('os.environ', {'PATH': '/temporary/launcher-one:/usr/bin'}):
                first = source_identity(command, root)
            with patch.dict('os.environ', {'PATH': '/temporary/launcher-two:/usr/bin'}):
                self.assertEqual(first, source_identity(command, root))

    def test_default_only_config_edit_does_not_change_recipe_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'engine.toml'
            config.write_text('[defaults]\nembedding="mean"\n')
            (root / 'scenario.toml').write_text('fixture')
            recipe = dispatch.Recipe('fixture', 'scenario.toml', (), (), ())
            first = dispatch.recipe_key(recipe, root, config)
            config.write_text('[defaults]\nembedding="cls"\n')
            self.assertEqual(first, dispatch.recipe_key(recipe, root, config))


if __name__ == '__main__':
    unittest.main()
