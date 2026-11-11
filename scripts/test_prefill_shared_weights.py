#!/usr/bin/env python3
import ast
import contextlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import ml_dtypes
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'designs/decode_fused/gen_llm_prefill.py'
namespace = {'os': os, 'np': np, 'BF16': ml_dtypes.bfloat16}
tree = ast.parse(SOURCE.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('quant_source_files', 'check_shared_weights')]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), namespace)


class SharedWeightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dump = self.root / 'weights'
        self.buffers = self.root / 'decode/buffers'
        self.dump.mkdir()
        self.buffers.mkdir(parents=True)
        self.sp = SimpleNamespace(weight_prefix='', d_model=4, head_dim=2, gqa_group=2,
                                  q_dim=4, kv_dim=2, n_kv_heads=1, sandwich_norms=False,
                                  layer_scalar=False, mixer_for=lambda _: 'full_attention')
        self.save('post_attention_layernorm', np.ones(4, np.float32))
        self.wire('n_pf', np.ones(4, np.float32))
        qkv = []
        for role, rows in [('q', 4), ('k', 2), ('v', 2)]:
            value = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
            self.save(f'self_attn.{role}_proj', value)
            qkv.append(value)
        self.wire('Wqkv', np.concatenate(qkv))
        for name, leaf in [('Wo', 'self_attn.o_proj'), ('Wg', 'mlp.gate_proj'),
                           ('Wu', 'mlp.up_proj')]:
            value = np.arange(16, dtype=np.float32).reshape(4, 4)
            self.save(leaf, value)
            self.wire(name, value)
        self.down = np.arange(24, dtype=np.float32).reshape(4, 6)
        self.save('mlp.down_proj', self.down)

    def save(self, leaf, value):
        np.save(self.dump / f'layers.0.{leaf}.weight.npy', value)

    def wire(self, name, value):
        np.asarray(value).astype(ml_dtypes.bfloat16).tofile(self.buffers / f'L0_{name}.bin')

    def chunks(self):
        for index, value in enumerate(np.split(self.down, 2, axis=1)):
            self.wire(f'Wdk{index}', value)

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            namespace['check_shared_weights'](str(self.root / 'decode/meta.json'),
                                               str(self.dump), self.sp,
                                               {'wqkv_head_major': False})

    def test_whole_float_dump_and_whole_arena(self):
        self.wire('Wd', self.down)
        self.check()

    def test_whole_float_dump_and_k_split_arena(self):
        self.chunks()
        self.check()

    def test_matching_prechunked_dump(self):
        self.chunks()
        (self.dump / 'layers.0.mlp.down_proj.weight.npy').unlink()
        for index, value in enumerate(np.split(self.down, 2, axis=1)):
            np.save(self.dump / f'layers.0.mlp.down_proj.weight.kchunk{index}.npy', value)
        self.check()

    def test_corrupted_chunk_rejects(self):
        self.chunks()
        damaged = self.down[:, :3].copy()
        damaged[0, 0] = -10
        self.wire('Wdk0', damaged)
        with self.assertRaises(SystemExit):
            self.check()

    def test_swapped_chunks_reject(self):
        for index, value in enumerate(reversed(np.split(self.down, 2, axis=1))):
            self.wire(f'Wdk{index}', value)
        with self.assertRaises(SystemExit):
            self.check()

    def test_nondivisible_k_rejects(self):
        self.chunks()
        self.save('mlp.down_proj', np.arange(20, dtype=np.float32).reshape(4, 5))
        with self.assertRaises(SystemExit):
            self.check()

    def test_packed_dump_partition_mismatch_still_rejects(self):
        self.chunks()
        self.save('mlp.down_proj', np.arange(24, dtype=np.uint8).reshape(4, 6))
        with self.assertRaises(SystemExit):
            self.check()


if __name__ == '__main__':
    unittest.main()
