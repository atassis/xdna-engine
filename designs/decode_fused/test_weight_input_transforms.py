# SPDX-License-Identifier: Apache-2.0
"""Exercise the generator's weight-only branches without compiling an NPU graph."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import ml_dtypes
from weight_inputs import WeightInputs

TREE = ast.parse(Path(__file__).with_name('gen_llm_decode.py').read_text())


def execute_node(predicate, env):
    matches = [n for n in ast.walk(TREE) if predicate(n)]
    assert len(matches) == 1, len(matches)
    exec(compile(ast.Module(body=matches, type_ignores=[]), '<generator weight branch>', 'exec'), env)


class TransformTests(unittest.TestCase):
    def test_real_qkv_reorder_records_identical_row_order(self):
        arrays = [np.arange(16, dtype=np.uint16).reshape(8, 2),
                  np.arange(100, 108, dtype=np.uint16).reshape(4, 2),
                  np.arange(200, 208, dtype=np.uint16).reshape(4, 2)]
        r = WeightInputs(True)
        env = dict(np=np, op_decode_layer=SimpleNamespace(wqkv_head_major=True),
                   g=SimpleNamespace(hkv=2, hd=2, has_v=True, op_attn_block=None), Hq=4, D=2,
                   precision=SimpleNamespace(wire_row_units=lambda *a: 2),
                   _spec=lambda site: None, _BUILD_STATE={'scale_dtype': 'f32', 'layout': 'header_first'},
                   qkv_parts=[a.reshape(-1) for a in arrays],
                   qkv_sources=[(n, list(a.shape)) for n, a in zip(('q', 'k', 'v'), arrays)],
                   p='L0_', weights={}, weight_inputs=r, _stream_pad_rows=lambda a, op: a)
        execute_node(lambda n: isinstance(n, ast.If) and
                     'op_decode_layer.wqkv_head_major' in ast.unparse(n.test), env)
        result = r.buffers['L0_Wqkv']
        source = dict(zip(('q', 'k', 'v'), arrays))
        recorded = np.concatenate([source[o['tensor']][o['first']:o['first'] + o['count']]
                                   for o in result]).reshape(-1)
        np.testing.assert_array_equal(env['weights']['L0_Wqkv'], recorded)
        independent = np.concatenate([arrays[0][:4], arrays[1][:2], arrays[2][:2],
                                      arrays[0][4:], arrays[1][2:], arrays[2][2:]]).reshape(-1)
        np.testing.assert_array_equal(recorded, independent)
        self.assertNotEqual(recorded.tobytes(), np.concatenate(arrays).tobytes())

    def test_real_norm_fold_records_float32_scale_before_bf16(self):
        r = WeightInputs(True)
        dtype = ml_dtypes.bfloat16
        values = dict(input=np.array([1, 2, 3, 4], dtype=dtype),
                      q=np.array([1.5, -2.5], dtype=dtype), k=np.array([0.5, 2], dtype=dtype))
        env = dict(np=np, nm={'n_in': 'input', 'n_qn': 'q', 'n_kn': 'k'},
                   load_norm=values.__getitem__, scale_in_qnorm=True,
                   sp=SimpleNamespace(attn_scale=0.125), bf16=lambda a: np.asarray(a, dtype=dtype),
                   weights={}, p='L0_', weight_inputs=r)
        execute_node(lambda n: isinstance(n, ast.For) and ast.unparse(n.target) == '(key, tensor)'
                     and ast.unparse(n.iter) == 'nm.items()', env)
        self.assertEqual(r.buffers['L0_n_qn'][0]['scale'], 0.125)
        self.assertNotIn('scale', r.buffers['L0_n_in'][0])
        self.assertEqual(env['weights']['L0_n_qn'].tobytes(),
                         np.array([0.1875, -0.3125], dtype=dtype).tobytes())


if __name__ == '__main__':
    unittest.main()
