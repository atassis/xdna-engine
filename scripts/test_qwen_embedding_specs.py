import json
import hashlib
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'designs/decode_fused'))
from llm_decode_spec import SPECS


class EmbeddingSpecTests(unittest.TestCase):
    def test_embedding_specs_have_their_own_actual_checkpoint_geometry(self):
        expected = {
            'qwen3-embedding-0.6b': (1024, 3072, 28, 16, 8, 128, 151669),
            'qwen3-embedding-4b': (2560, 9728, 36, 32, 8, 128, 151665),
        }
        for name, geometry in expected.items():
            with self.subTest(name=name):
                self.assertIn(name, SPECS)
                spec = SPECS[name]
                self.assertEqual((spec.d_model,spec.ffn,spec.n_layers,spec.n_q_heads,
                                  spec.n_kv_heads,spec.head_dim,spec.vocab), geometry)
                self.assertEqual(spec.weight_prefix, '')
                self.assertEqual(spec.norm_gain, 'w')
                self.assertEqual(spec.embed_scale, 'none')
                self.assertTrue(spec.qk_norm)
                self.assertFalse(spec.sandwich_norms)
                self.assertIsNone(spec.sliding_window)
                self.assertEqual(spec.rope_theta_global, 1_000_000.)
        self.assertNotEqual(SPECS['qwen3-embedding-0.6b'].vocab, SPECS['qwen3-0.6b'].vocab)

    def test_embedding_runtime_metadata_files_have_content_sha256_pins(self):
        from qwen_embedding import model_pin
        for model_id in ('qwen3-embedding-0.6b', 'qwen3-embedding-4b'):
            for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
                with self.subTest(model=model_id, file=name):
                    digest = model_pin(model_id)['files'][name].get('sha256', '')
                    self.assertRegex(digest, r'^[0-9a-f]{64}$')

    def test_unaligned_vocabulary_head_is_zero_padded_without_changing_real_rows(self):
        import numpy as np
        from llm_decode_spec import pad_vocab_weight
        spec = SPECS['qwen3-embedding-0.6b']
        original = np.arange(spec.vocab, dtype=np.uint32).reshape(-1,1)
        padded = pad_vocab_weight(original, spec.padded_vocab(8,4))
        self.assertEqual(padded.shape, (151680,1))
        self.assertEqual(padded.dtype, original.dtype)
        np.testing.assert_array_equal(padded[:spec.vocab], original)
        self.assertTrue((padded[spec.vocab:] == 0).all())
        self.assertIs(pad_vocab_weight(original,spec.vocab), original)
        with self.assertRaises(ValueError):
            pad_vocab_weight(original,spec.vocab-1)

    def test_build_metadata_requires_the_pinned_embedding_source(self):
        from llm_decode_spec import embedding_source_metadata
        from qwen_embedding import model_pin
        model_id = 'qwen3-embedding-0.6b'
        pin = model_pin(model_id)
        source = {'model':pin['repo'], 'revision':pin['revision'], 'files':{
            name:entry.get('sha256', 'a'*64) for name,entry in pin['files'].items()}}
        manifest = {'spec':model_id, 'layers':28, 'embedding_source':source}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'quant.json'
            path.write_text(json.dumps(manifest))
            self.assertEqual(embedding_source_metadata(tmp, model_id, 28), source)
            self.assertIsNone(embedding_source_metadata(tmp, 'qwen3-0.6b', 28))
            for changed in [dict(manifest, spec='qwen3-0.6b'), dict(manifest, layers=1),
                            dict(manifest, embedding_source=dict(source, revision='b'*40)),
                            dict(manifest, embedding_source=dict(source, files={}))]:
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    embedding_source_metadata(tmp, model_id, 28)

    def test_embedding_dump_refuses_implicit_provisioning_and_repo_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extra in [[], ['--repo','Qwen/Qwen3-0.6B']]:
                dest = Path(tmp)/'output'
                result = subprocess.run([sys.executable, str(ROOT/'scripts/dump_llm_weights.py'),
                                         '--spec','qwen3-embedding-0.6b','--out',str(dest),*extra],
                                        capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('requires a pinned local --checkpoint-dir',result.stderr)
                self.assertFalse(dest.exists())

    def test_miniature_pinned_export_records_identity_and_refuses_changed_resume(self):
        import torch
        from safetensors.torch import save_file
        import dump_llm_weights as dump
        spec = replace(SPECS['qwen3-embedding-0.6b'], d_model=8, ffn=16, n_layers=1,
                       n_q_heads=2, n_kv_heads=1, head_dim=4, vocab=8)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            source.mkdir()
            tensors = {'embed_tokens.weight':torch.ones(8,8), 'norm.weight':torch.ones(8)}
            tensors.update({name:torch.ones(4 if name.endswith(('q_norm.weight','k_norm.weight')) else 8)
                            for name in spec.norm_weight_names(0).values()})
            for leaf, shape in {'self_attn.q_proj':(8,8), 'self_attn.k_proj':(4,8),
                                'self_attn.v_proj':(4,8), 'self_attn.o_proj':(8,8),
                                'mlp.gate_proj':(16,8), 'mlp.up_proj':(16,8), 'mlp.down_proj':(8,16)}.items():
                tensors[f'layers.0.{leaf}.weight'] = torch.ones(*shape)
            save_file(tensors, source/'model.safetensors')
            raw = (source/'model.safetensors').read_bytes()
            pin = {'repo':'fixture/embedding','revision':'a'*40,'files':{
                'model.safetensors':{'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}}}
            dest = Path(tmp)/'output'
            args = ['dump', '--spec',spec.name,'--checkpoint-dir',str(source),'--out',str(dest)]
            with patch.dict(dump.SPECS, {spec.name:spec}), patch('qwen_embedding.model_pin', return_value=pin):
                with patch.object(sys,'argv',args):
                    dump.main()
                manifest = json.loads((dest/'quant.json').read_text())
                self.assertEqual(manifest['embedding_source']['revision'], 'a'*40)
                self.assertEqual(manifest['layers'], 1)
                self.assertEqual(manifest['spec'], spec.name)
                with patch.object(sys,'argv',args+['--resume']):
                    dump.main()
                before = (dest/'quant.json').read_bytes()
                changed = dict(pin,revision='b'*40)
                with patch('qwen_embedding.model_pin',return_value=changed), patch.object(sys,'argv',args+['--resume']):
                    with self.assertRaisesRegex(SystemExit,'different format'):
                        dump.main()
                self.assertEqual(before,(dest/'quant.json').read_bytes())

    def test_embedding_dump_rejects_wrong_weights_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            source.mkdir()
            (source/'model.safetensors').write_bytes(b'not embedding weights')
            dest = Path(tmp)/'output'
            result = subprocess.run([sys.executable, str(ROOT/'scripts/dump_llm_weights.py'),
                                     '--spec','qwen3-embedding-0.6b','--checkpoint-dir',str(source),
                                     '--out',str(dest)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('pinned',result.stderr)
            self.assertFalse(dest.exists())


if __name__ == '__main__':
    unittest.main()
