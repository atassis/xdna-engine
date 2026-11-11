import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

import qwen_embedding as qe


class RecordingModel:
    """Minimal backbone that records the exact tensors passed to each forward."""
    def __init__(self):
        self.forwards = []

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, use_cache, **unused):
        self.forwards.append(dict(input_ids=input_ids.clone(), attention_mask=attention_mask.clone(),
                                  use_cache=use_cache))
        hidden = torch.stack((input_ids.float(), torch.ones_like(input_ids, dtype=torch.float)), dim=-1)
        return type('Output', (), {'last_hidden_state': hidden})()


class ContractTests(unittest.TestCase):
    def tokenizer(self):
        tok = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, 'a': 2, 'b': 3}, unk_token='[UNK]'))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]', padding_side='left')

    def test_query_document_are_explicit_and_exact(self):
        self.assertEqual(qe.prepare_texts(['a'], 'document'), ['a'])
        self.assertEqual(qe.prepare_texts(['a'], 'query'), [f'Instruct: {qe.TASK}\nQuery:a'])
        for texts, role in [([], 'document'), ([''], 'document'), (['a'], 'guess'), ('a', 'document'), ([1], 'query')]:
            with self.subTest(texts=texts, role=role), self.assertRaises(ValueError):
                qe.prepare_texts(texts, role)

    def test_token_limit_never_truncates_and_includes_instruction(self):
        tok = self.tokenizer()
        batch = qe.tokenize(tok, ['a', 'a b'], 'document', 2)
        self.assertEqual(batch['input_ids'].tolist(), [[1, 2], [2, 3]])
        self.assertEqual(batch['attention_mask'].tolist(), [[0, 1], [1, 1]])
        with self.assertRaisesRegex(ValueError, 'item 1'):
            qe.tokenize(tok, ['a', 'a b a'], 'document', 2)
        with self.assertRaises(ValueError):
            qe.tokenize(tok, ['a'], 'query', 2)
        for limit in [0, True, 513]:
            with self.assertRaises(ValueError):
                qe.tokenize(tok, ['a'], 'document', limit)

    def test_last_valid_pool_not_cls_mean_or_padding(self):
        hidden = torch.tensor([[[99., 99.], [3., 4.], [8., 1.]], [[99., 99.], [0., 7.], [0., 2.]]])
        mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
        actual = qe.pool(hidden, mask, 2)
        torch.testing.assert_close(actual, torch.tensor([[.6, .8], [0., 1.]]), rtol=0, atol=1e-7)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(qe.pool(hidden.bfloat16(), mask, 2), actual)

    def test_pool_rejects_invalid_masks_shapes_and_vectors(self):
        for mask in [torch.tensor([[0, 0, 0]]), torch.tensor([[1, 0, 1]]), torch.tensor([[1, 2, 0]]), torch.ones(1, 2)]:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                qe.pool(torch.ones(1, 3, 2), mask, 2)
        for value in [0., float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                qe.pool(torch.full((1, 2, 2), value), torch.ones(1, 2), 2)
        with self.assertRaises(ValueError):
            qe.pool(torch.ones(1, 2, 3), torch.ones(1, 2), 2)
        with self.assertRaises(ValueError):
            qe.pool(torch.ones(0, 2, 3), torch.ones(0, 2), 3)

    def test_pins_are_embedding_models_full_dimension_only(self):
        a = qe.model_pin('qwen3-embedding-0.6b')
        self.assertEqual(a['revision'], '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3')
        self.assertEqual(a['dimension'], 1024)
        self.assertEqual(qe.model_pin('qwen3-embedding-4b')['dimension'], 2560)
        for name in ['qwen3-0.6b', 'bge-base', 'qwen3-embedding-8b']:
            with self.assertRaises(ValueError):
                qe.model_pin(name)

    def test_snapshot_checks_bytes_not_directory_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = b'configuration'
            (root / 'config.json').write_bytes(data)
            pin = {'files': {'config.json': {'size': len(data), 'git_sha1': hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}}}
            receipt = qe.verify_files(root, pin)
            self.assertEqual(receipt['config.json'], hashlib.sha256(data).hexdigest())
            (root / 'config.json').write_bytes(b'Configuration')
            with self.assertRaises(ValueError):
                qe.verify_files(root, pin)
            (root / 'config.json').unlink()
            with self.assertRaises(ValueError):
                qe.verify_files(root, pin)

    def test_lfs_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'model.safetensors').write_bytes(b'fake')
            with self.assertRaises(ValueError):
                qe.verify_files(root, {'files': {'model.safetensors': {'size': 4, 'sha256': '0'*64}}})

    def test_profile_binds_model_tokenizer_policy_and_dimension(self):
        pin = qe.model_pin('qwen3-embedding-0.6b')
        identity = qe.profile(pin, {'tokenizer.json': 'a', 'model.safetensors': 'b'}, 512)
        for changed in [qe.profile(pin, {'tokenizer.json': 'x', 'model.safetensors': 'b'}, 512),
                        qe.profile(pin, {'tokenizer.json': 'a', 'model.safetensors': 'x'}, 512),
                        qe.profile(pin, {'tokenizer.json': 'a', 'model.safetensors': 'b'}, 256)]:
            self.assertNotEqual(identity, changed)
        self.assertEqual(identity['pooling'], 'final-backbone-last-valid-token')
        self.assertEqual(identity['document_prefix'], '')

    def test_verified_local_loader_ignores_unpinned_tokenizer_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Qwen3Config(vocab_size=4, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                              num_attention_heads=2, num_key_value_heads=1, head_dim=8, pad_token_id=1)
            Qwen3Model(cfg).save_pretrained(root)
            self.tokenizer().save_pretrained(root)
            names = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']
            files = {n: {'size': (root/n).stat().st_size, 'sha256': hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in names}
            # This file is deliberately not pinned; a direct HF directory load would consume it.
            (root/'special_tokens_map.json').write_text(json.dumps({'pad_token': 'a'}))
            model, tok, hashes = qe.load_reference(root, {'files': files, 'dimension': 16}, torch.float32)
            self.assertEqual(tok.pad_token, '[PAD]')
            self.assertEqual(tok.padding_side, 'left')
            self.assertEqual(set(hashes), set(names))
            self.assertEqual(qe.encode(model, tok, ['a', 'a b'], 'document', 8, 16).shape, (2, 16))

    def test_batched_capture_preserves_order_and_exact_forward_evidence(self):
        tok = self.tokenizer()
        texts = ['a', 'a b', 'b', 'a b a', 'a']
        for batch_size, expected_sizes in [(1, [1, 1, 1, 1, 1]), (2, [2, 2, 1])]:
            with self.subTest(batch_size=batch_size):
                model = RecordingModel()
                vectors, input_ids, masks, forward_batch_sizes = qe.encode_with_evidence(
                    model, tok, texts, 'document', 8, 2, batch_size=batch_size)
                self.assertEqual(forward_batch_sizes, expected_sizes)
                self.assertEqual([call['input_ids'].shape[0] for call in model.forwards], expected_sizes)
                self.assertTrue(all(call['input_ids'].shape[0] <= batch_size for call in model.forwards))
                self.assertTrue(all(call['use_cache'] is False for call in model.forwards))
                expected_ids = [row for call in model.forwards for row in call['input_ids'].tolist()]
                expected_masks = [row for call in model.forwards for row in call['attention_mask'].tolist()]
                self.assertEqual(input_ids, expected_ids)
                self.assertEqual(masks, expected_masks)
                self.assertEqual(len(vectors), len(texts))
                expected = torch.tensor([[2., 1.], [3., 1.], [3., 1.], [2., 1.], [2., 1.]])
                expected = expected / expected.norm(dim=1, keepdim=True)
                torch.testing.assert_close(vectors, expected)
                row_widths = [1, 2, 1, 3, 1] if batch_size == 1 else [2, 2, 3, 3, 1]
                self.assertEqual([len(row) for row in input_ids], row_widths)
                self.assertEqual([len(row) for row in masks], row_widths)

    def test_batch_size_is_strictly_positive_and_overflow_prevalidates_before_forward(self):
        tok = self.tokenizer()
        for batch_size in [0, -1, True, False, 1.0, '1']:
            with self.subTest(batch_size=batch_size), self.assertRaisesRegex(ValueError, 'batch_size'):
                qe.encode_with_evidence(RecordingModel(), tok, ['a'], 'document', 8, 2, batch_size=batch_size)
        model = RecordingModel()
        with self.assertRaisesRegex(ValueError, 'item 1'):
            qe.encode_with_evidence(model, tok, ['a', 'a b a'], 'document', 2, 2, batch_size=1)
        self.assertEqual(model.forwards, [])

    def test_cli_invalid_batch_size_rejects_before_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'inputs.json'
            inputs.write_text('["a"]')
            for value in ['0', '-1']:
                argv = ['qwen_embedding.py', '--model', 'qwen3-embedding-0.6b',
                        '--snapshot', str(root), '--input', str(inputs),
                        '--output', str(root / 'output.json'), '--batch-size', value]
                with self.subTest(value=value), mock.patch('sys.argv', argv), \
                        mock.patch.object(qe, 'load_reference', side_effect=AssertionError('checkpoint load')) as load:
                    with self.assertRaises(SystemExit) as raised:
                        qe.main()
                    self.assertEqual(raised.exception.code, 2)
                    load.assert_not_called()
                    self.assertFalse((root / 'output.json').exists())

    def test_encode_compatibility_and_batched_repeat_stability(self):
        tok = self.tokenizer()
        texts = ['a', 'a b', 'b']
        old_style = qe.encode(RecordingModel(), tok, texts, 'document', 8, 2)
        batched = qe.encode(RecordingModel(), tok, texts, 'document', 8, 2, batch_size=2)
        first, _, _, _ = qe.encode_with_evidence(RecordingModel(), tok, texts, 'document', 8, 2, batch_size=2)
        repeated, _, _, _ = qe.encode_with_evidence(RecordingModel(), tok, texts, 'document', 8, 2, batch_size=2)
        torch.testing.assert_close(old_style, batched)
        torch.testing.assert_close(first, repeated)

    def test_real_random_qwen_backbone_batch_isolation(self):
        # Random miniature HF backbone checks harness mechanics, not checkpoint quality or NPU parity.
        torch.manual_seed(42)
        cfg = Qwen3Config(vocab_size=4, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=8, pad_token_id=1)
        model = Qwen3Model(cfg).eval()
        tok = self.tokenizer()
        texts = ['a', 'a b', 'a', 'b b a']
        batch = qe.encode(model, tok, texts, 'document', 8, 16, batch_size=2)
        for i, text in enumerate(texts):
            single = qe.encode(model, tok, [text], 'document', 8, 16)
            torch.testing.assert_close(batch[i], single[0], rtol=1e-5, atol=1e-6)
        reordered = qe.encode(model, tok, [texts[i] for i in [3, 0, 1, 2]], 'document', 8, 16)
        torch.testing.assert_close(reordered, batch[[3, 0, 1, 2]], rtol=1e-5, atol=1e-6)
        changed = qe.encode(model, tok, ['a', 'b', 'a', 'b'], 'document', 8, 16)
        torch.testing.assert_close(changed[[0, 2]], batch[[0, 2]], rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(batch).all())
        torch.testing.assert_close(batch.norm(dim=1), torch.ones(4), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
