#!/usr/bin/env python3
import ast
from pathlib import Path
import unittest

import ml_dtypes
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'designs/decode_fused/llm_decode_spec.py'
NODE = next(node for node in ast.parse(SOURCE.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == 'pad_vocab_weight')
NAMESPACE = {}
exec(compile(ast.Module(body=[NODE], type_ignores=[]), str(SOURCE), 'exec'), NAMESPACE)
pad = NAMESPACE['pad_vocab_weight']


class VocabPaddingDtypeTests(unittest.TestCase):
    def test_head_cast_happens_inside_padding(self):
        calls = [node for node in ast.walk(ast.parse((SOURCE.parent / 'gen_llm_decode.py').read_text()))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == 'pad_vocab_weight']
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual(calls[0].args[0].id, 'head_f32')
        self.assertTrue(any(keyword.arg == 'dtype' and isinstance(keyword.value, ast.Name)
                            and keyword.value.id == 'BF16' for keyword in calls[0].keywords))

    def test_bf16_bits_match_convert_then_pad_across_row_chunks(self):
        source = np.random.default_rng(17).standard_normal((1027, 17)).astype(np.float32)
        source[0, :8] = np.array([0, 0x80000000, 0x3f808000, 0x3f818000,
                                  0x7f800000, 0xff800000, 0x7fc12345, 0x00000001], np.uint32).view(np.float32)
        before = source.tobytes()
        expected = np.pad(source.astype(ml_dtypes.bfloat16), ((0, 5), (0, 0)))
        actual = pad(source, 1032, dtype=ml_dtypes.bfloat16)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual(source.tobytes(), before)

    def test_no_padding_still_converts_to_selected_dtype(self):
        source = np.arange(21, dtype=np.float32).reshape(7, 3) / 13
        expected = source.astype(ml_dtypes.bfloat16)
        self.assertEqual(pad(source, 7, dtype=ml_dtypes.bfloat16).tobytes(), expected.tobytes())

    def test_default_retains_original_dtype_and_identity_without_padding(self):
        source = np.arange(21, dtype=np.float32).reshape(7, 3)
        self.assertIs(pad(source, 7), source)
        self.assertEqual(pad(source, 9).dtype, source.dtype)
        np.testing.assert_array_equal(pad(source, 9), np.pad(source, ((0, 2), (0, 0))))

    def test_readonly_strided_input(self):
        source = np.arange(42, dtype=np.float32).reshape(7, 6)[:, ::2]
        source.flags.writeable = False
        expected = np.pad(source.astype(ml_dtypes.bfloat16), ((0, 2), (0, 0)))
        self.assertEqual(pad(source, 9, dtype=ml_dtypes.bfloat16).tobytes(), expected.tobytes())

    def test_invalid_rank_and_smaller_padding_reject(self):
        for source, rows in [(np.zeros(7), 9), (np.zeros((7, 3)), 6)]:
            with self.assertRaises(ValueError):
                pad(source, rows, dtype=ml_dtypes.bfloat16)


if __name__ == '__main__':
    unittest.main()
