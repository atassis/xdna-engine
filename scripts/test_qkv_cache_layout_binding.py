#!/usr/bin/env python3
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'designs/decode_fused/gen_llm_decode.py'
CALLS = [node for node in ast.walk(ast.parse(SOURCE.read_text()))
         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
         and node.func.id == 'QKVHeadDataParallel']
assert len(CALLS) == 1
CALL = compile(ast.Expression(CALLS[0]), str(SOURCE), 'eval')


class QkvCacheLayoutBindingTests(unittest.TestCase):
    def arguments(self, window, capacity, block):
        env = dict(D=2560, hd=128, Hq=32, hkv=8, KVA=capacity, S=window, T=block,
                   COLS=8, TSI=4, WEIGHT_DEPTH=2, ctx=object(),
                   sp=SimpleNamespace(eps=1e-6, qkv_dp_cols=lambda cols, n_kv_heads: n_kv_heads),
                   _quant_kw=lambda site: {}, QKVHeadDataParallel=lambda **kwargs: kwargs)
        return eval(CALL, env)

    def test_blocked_cache_append_uses_declared_block(self):
        args = self.arguments(window=512, capacity=512, block=64)
        self.assertEqual(args['max_seq'], 512)
        self.assertEqual(args.get('kv_block_size'), 64)

    def test_flat_default_stays_byte_identical(self):
        args = self.arguments(window=512, capacity=512, block=512)
        self.assertIsNone(args.get('kv_block_size'))

    def test_block_is_relative_to_capacity_not_attention_window(self):
        args = self.arguments(window=128, capacity=512, block=128)
        self.assertEqual(args['max_seq'], 512)
        self.assertEqual(args.get('kv_block_size'), 128)


if __name__ == '__main__':
    unittest.main()
