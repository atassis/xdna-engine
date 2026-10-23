import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPTS = Path(__file__).resolve().parent


class TransientWeightsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repo'
        scripts = self.repo / 'scripts'
        (scripts / 'lib').mkdir(parents=True)
        for name in ['with_llm_weights.py', 'require_disk_backed.sh']:
            if (SCRIPTS / name).exists():
                shutil.copyfile(SCRIPTS / name, scripts / name)
        for name in ['data_root.py', 'data_root.sh']:
            shutil.copyfile(SCRIPTS / 'lib' / name, scripts / 'lib' / name)
        self.data = self.root / 'data'
        self.checkpoint = self.data / 'model-inputs/qwen3.5-4b/checkpoint'
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / 'model.safetensors').write_bytes(b'fixture')
        self.result = self.root / 'result.json'
        self.export_log = self.root / 'export.json'
        (scripts / 'dump_llm_weights.py').write_text('''import argparse, json, os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--spec'); p.add_argument('--checkpoint-dir'); p.add_argument('--out')
a=p.parse_args()
Path(os.environ['EXPORT_LOG']).write_text(json.dumps(vars(a)))
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out)/'tensor.npy').write_bytes(b'unchanged-values')
raise SystemExit(int(os.environ.get('EXPORT_EXIT','0')))
''')
        self.consumer = self.root / 'consumer.py'
        self.consumer.write_text('''import json, os, time
from pathlib import Path
w=Path(os.environ['WEIGHTS'])
Path(os.environ['RESULT']).write_text(json.dumps({'weights':str(w),'bytes':(w/'tensor.npy').read_text(),'pid':os.getpid()}))
if os.environ.get('CHILD_SLEEP') == '1': time.sleep(60)
raise SystemExit(int(os.environ.get('CHILD_EXIT','0')))
''')
        self.env = {**os.environ, 'XDNA_DATA': str(self.data),
                    'XDNA_BUILD': str(self.data / 'build'), 'ALLOW_TMPFS_BUILD': '1',
                    'EXPORT_LOG': str(self.export_log), 'RESULT': str(self.result),
                    'PYTHONPATH': str(SCRIPTS) + os.pathsep + os.environ.get('PYTHONPATH', '')}
        self.env.pop('WEIGHTS', None)
        self.argv = [sys.executable, str(scripts / 'with_llm_weights.py'),
                     '--spec', 'qwen3.5-4b', '--', sys.executable, str(self.consumer)]

    def run_adapter(self, **overrides):
        return subprocess.run(self.argv, env={**self.env, **overrides},
                              text=True, capture_output=True, timeout=15)

    def assert_no_dump(self):
        root = self.data / 'build/llm-weight-transactions'
        self.assertFalse(root.exists() and list(root.iterdir()))

    def test_export_is_alive_for_child_then_removed_and_second_run_does_not_grow(self):
        for _ in range(2):
            result = self.run_adapter()
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(self.result.read_text())
            self.assertEqual(row['bytes'], 'unchanged-values')
            self.assertFalse(Path(row['weights']).exists())
            self.assertEqual(json.loads(self.export_log.read_text())['checkpoint_dir'], str(self.checkpoint))
            self.assert_no_dump()

    def test_explicit_weights_are_never_exported_or_collected(self):
        weights = self.root / 'owned weights'
        weights.mkdir()
        (weights / 'tensor.npy').write_bytes(b'caller-values')
        result = self.run_adapter(WEIGHTS=str(weights))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.export_log.exists())
        self.assertEqual(json.loads(self.result.read_text())['bytes'], 'caller-values')
        self.assertEqual((weights / 'tensor.npy').read_bytes(), b'caller-values')

    def test_export_failure_never_starts_consumer_and_does_not_leave_raw_dump(self):
        result = self.run_adapter(EXPORT_EXIT='41')
        self.assertEqual(result.returncode, 41, result.stderr)
        self.assertFalse(self.result.exists())
        self.assert_no_dump()

    def test_consumer_failure_preserves_external_result_and_propagates_status(self):
        result = self.run_adapter(CHILD_EXIT='42')
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertTrue(self.result.exists())
        self.assert_no_dump()

    def test_gate_reference_reexecutes_with_one_export_and_cleans_it(self):
        scripts = self.repo / 'scripts'
        shutil.copyfile(SCRIPTS / 'gate_llm.sh', scripts / 'gate_llm.sh')
        shutil.copyfile(self.consumer, scripts / 'gate_llm_reference.py')
        venv = self.repo / '.venv-iron'
        (venv / 'bin').mkdir(parents=True)
        (venv / 'bin/python').symlink_to(sys.executable)
        result = subprocess.run(['bash', str(scripts / 'gate_llm.sh'), '--make-ref'],
                                env={**self.env, 'GATE_SPEC': 'qwen3.5-4b', 'WEIGHTS': ''},
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(self.result.read_text())
        self.assertEqual(row['bytes'], 'unchanged-values')
        self.assertFalse(Path(row['weights']).exists())
        self.assert_no_dump()

    def test_missing_local_checkpoint_fails_before_export(self):
        shutil.rmtree(self.checkpoint)
        result = self.run_adapter()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checkpoint', result.stderr.lower())
        self.assertFalse(self.export_log.exists())
        self.assertFalse(self.result.exists())

    def test_missing_explicit_weights_do_not_silently_export(self):
        result = self.run_adapter(WEIGHTS=str(self.root / 'missing'))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.export_log.exists())

    def test_sigterm_stops_owned_child_before_cleaning_raw_weights(self):
        process = subprocess.Popen(self.argv, env={**self.env, 'CHILD_SLEEP': '1'},
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 8
            while not self.result.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(self.result.exists(), 'consumer never started')
            child = json.loads(self.result.read_text())
            lock = self.data / 'locks/model-artifacts' / (hashlib.sha256(os.fsencode(self.checkpoint.resolve())).hexdigest() + '.lock')
            self.assertTrue(lock.is_file(), 'checkpoint lease was not held through consumption')
            with lock.open('a+b') as handle:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            process.send_signal(signal.SIGTERM)
            _, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 143, stderr)
            self.assertFalse(Path('/proc', str(child['pid'])).exists())
            self.assertFalse(Path(child['weights']).exists())
            self.assert_no_dump()
        finally:
            if process.poll() is None:
                process.terminate()
            process.communicate(timeout=10)


if __name__ == '__main__':
    unittest.main()
