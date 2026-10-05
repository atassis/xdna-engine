import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).parent


class DraftNormDefinesTests(unittest.TestCase):
    def test_odd_head_padding_has_its_completion_token_route(self):
        code = '''import rlayer_design as L
import rseg_design
text = L.build_text('m h arena nbw=20 1 2'.split())
assert 'aiex.dma_await_task(%hp0)' in text
m = L.M()
rseg_design.tct_flow(m, '%mem_0', '%shim_0')
assert '\\n'.join(m.lines) in text, 'awaited MemTile completion has no token route'
'''
        for model in ('gemma4-12b', 'gemma3-270m-rail'):
            with self.subTest(model=model):
                result = subprocess.run([sys.executable, '-c', code], cwd=HERE,
                                        env=os.environ | {'RF_MODEL': model, 'RF_ATTN_H': '1', 'RF_FAST': '1'},
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_head_norm_broadcast_matches_the_core_ring_unit(self):
        code = '''import re
import rlayer_design as L
import rhead
from m1_design import AB_UNIT
L.configure('m h arena nbw=10 1 2'.split())
m = L.M()
rhead.sequence(L, m)
text = '\\n'.join(m.lines)
start = text.index('%hnb0 =')
length = int(re.search(r'len = (\\d+)', text[start:]).group(1))
assert length == 2 * AB_UNIT, (length, 2 * AB_UNIT)
'''
        for model in ('gemma4-12b', 'gemma3-270m-rail'):
            with self.subTest(model=model):
                result = subprocess.run([sys.executable, '-c', code], cwd=HERE,
                                        env=os.environ | {'RF_MODEL': model}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_host_window_and_ring_gate_follow_the_model(self):
        code = '''import rld_run
import rlayer_design as L
from rf_model import MODEL
import kvring
assert rld_run.WIN == MODEL.sliding_window
first = 512
widths = rld_run.widths(1024, 1, 1, first)
assert widths[0, 1, 0] == max(0, 1024 - (MODEL.sliding_window - 1) - first)
if MODEL.name == 'gemma3-270m-rail':
    assert widths[0, 1, 0] == 1
    kvring.check(1280, 10, (1, 2, 3), MODEL.sliding_window, 1280 * L.KVROW)
'''
        for model in ('gemma4-12b', 'gemma3-270m-rail'):
            with self.subTest(model=model):
                result = subprocess.run([sys.executable, '-c', code], cwd=HERE,
                                        env=os.environ | {'RF_MODEL': model}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_kernel_and_reference_epsilon_follow_model_padding(self):
        code = '''import struct
from rf_model import MODEL
import rmlp_design as R
import rlayer_design as L
import rmlp_ref
bits = lambda f: struct.pack('<f', f)
expected = MODEL.norm_eps * MODEL.norm_width / MODEL.D
for name, src, defs in L.kernels():
    if name not in (R.NORM, L.NORMZ):
        continue
    values = dict(d[2:].split('=', 1) for d in defs if d.startswith('-D') and '=' in d)
    assert int(values['RF_NORM_W']) == R.KC
    assert bits(float(values['RF_NORM_EPS'].removesuffix('f'))) == bits(expected), values
assert bits(rmlp_ref.EPS) == bits(expected), (rmlp_ref.EPS, expected)
'''
        for model in ('gemma4-12b', 'gemma3-270m-rail'):
            with self.subTest(model=model):
                result = subprocess.run([sys.executable, '-c', code], cwd=HERE,
                                        env=os.environ | {'RF_MODEL': model}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
