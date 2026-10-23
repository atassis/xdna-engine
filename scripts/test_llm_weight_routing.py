"""Execute the shell entry points with inert Python/toolchain boundaries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parent


class WeightRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / 'scripts'
        (scripts / 'lib').mkdir(parents=True)
        for name in ['build_llm_decode.sh', 'gate_llm.sh', 'run_llm_perplexity.sh', 'require_disk_backed.sh']:
            shutil.copyfile(SCRIPTS / name, scripts / name)
        shutil.copyfile(SCRIPTS / 'lib/data_root.sh', scripts / 'lib/data_root.sh')
        (scripts / 'amd_paths.sh').write_text('exit 63\n')
        venv = self.root / 'venv'
        (venv / 'bin').mkdir(parents=True)
        python = venv / 'bin/python'
        python.write_text('#!' + sys.executable + '''
import json, os, sys
from pathlib import Path
with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
raise SystemExit(37 if Path(sys.argv[1]).name == 'with_llm_weights.py' else 0)
''')
        python.chmod(0o755)
        self.log = self.root / 'calls.jsonl'
        self.env = {**os.environ, 'VENV_IRON': str(venv), 'XDNA_DATA': str(self.root / 'data'),
                    'XDNA_BUILD': str(self.root / 'data/build'), 'CALLS': str(self.log),
                    'ALLOW_TMPFS_BUILD': '1', 'GATE_SPEC': 'qwen3.5-4b',
                    'GATE_REF': str(self.root / 'ref.json')}
        self.env.pop('WEIGHTS', None)
        self.corpus = self.root / 'corpus with spaces.txt'
        self.corpus.write_text('fixture')

    def run_shell(self, name, *args, **env):
        return subprocess.run(['bash', str(self.root / 'scripts' / name), *args],
                              env={**self.env, **env}, capture_output=True, text=True, timeout=10)

    def calls(self):
        return [json.loads(row) for row in self.log.read_text().splitlines()] if self.log.exists() else []

    def assert_routed(self, name, args, **env):
        result = self.run_shell(name, *args, **env)
        self.assertEqual(result.returncode, 37, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(Path(calls[0][0]).name, 'with_llm_weights.py')
        separator = calls[0].index('--')
        self.assertEqual(calls[0][separator+1:], ['bash', str(self.root / 'scripts' / name), *args])

    def test_build_default_routes_without_rewriting_argv(self):
        self.assert_routed('build_llm_decode.sh', ['qwen3.5-4b', '', str(self.root / 'output with spaces')])

    def test_perplexity_default_routes_before_toolchain_or_device(self):
        self.assert_routed('run_llm_perplexity.sh', ['qwen3.5-4b', str(self.corpus), str(self.root / 'ppl'), '31'])

    def test_invalid_corpus_does_not_export(self):
        result = self.run_shell('run_llm_perplexity.sh', 'qwen3.5-4b', str(self.root / 'absent'), 'out')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls())

    def test_reference_modes_route_with_all_arguments_preserved(self):
        for mode in ['--make-ref', '--make-prefill-refs', '--refresh-goldens']:
            with self.subTest(mode=mode):
                self.log.unlink(missing_ok=True)
                self.assert_routed('gate_llm.sh', [mode, 'argument with spaces'])

    def test_tier2_judge_only_routes_only_when_reference_is_missing(self):
        self.assert_routed('gate_llm.sh', ['--tier2', '--judge-only'])
        self.log.unlink()
        Path(self.env['GATE_REF']).write_text('{}')
        result = self.run_shell('gate_llm.sh', '--tier2', '--judge-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(Path(call[0]).name == 'with_llm_weights.py' for call in self.calls()))

    def test_tier1_judge_does_not_export(self):
        artifact = self.root / 'artifact'
        artifact.mkdir()
        (artifact / 'meta.json').write_text('{}')
        result = self.run_shell('gate_llm.sh', '--tier1', '--judge-only', GATE_ARTIFACTS=str(artifact))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(Path(call[0]).name == 'with_llm_weights.py' for call in self.calls()))

    def test_default_reference_belongs_to_selected_model(self):
        self.env.pop('GATE_REF')
        for model in ['qwen3.5-4b', 'qwen3-0.6b']:
            with self.subTest(model=model):
                self.log.unlink(missing_ok=True)
                result = self.run_shell('gate_llm.sh', '--make-ref', GATE_SPEC=model, WEIGHTS=str(self.root))
                self.assertEqual(result.returncode, 0, result.stderr)
                call = self.calls()[0]
                self.assertEqual(call[call.index('--out') + 1],
                                 str(self.root / 'tests/refs' / model / 'gate_ref_n32.json'))

    def test_explicit_weights_and_other_models_do_not_route(self):
        for env in [{'WEIGHTS': str(self.root)}, {'GATE_SPEC': 'qwen3-0.6b'}]:
            with self.subTest(env=env):
                self.log.unlink(missing_ok=True)
                result = self.run_shell('gate_llm.sh', '--make-ref', **env)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(any(Path(call[0]).name == 'with_llm_weights.py' for call in self.calls()))


if __name__ == '__main__':
    unittest.main()
