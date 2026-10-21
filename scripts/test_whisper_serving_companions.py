import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace
import shutil

import onnx
from onnx import TensorProto, helper
import whisper_serving_companions as companions
import model_artifact_dispatch as dispatch


def graph(path, inputs, outputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    def values(spec):
        return [helper.make_tensor_value_info(name, typ, shape) for name, typ, shape in spec]
    model = helper.make_model(helper.make_graph([], 'schema-fixture', values(inputs), values(outputs)))
    onnx.save(model, path)


def fixture(root):
    f, i = TensorProto.FLOAT, TensorProto.INT64
    graph(root / 'preprocessor.onnx', [('waveform', f, [1, 480000])], [('input_features', f, [1, 128, 3000])])
    present = [(f'present.{layer}.{kind}.{kv}', f, [1, 20, 'sequence', 64])
               for layer in range(4) for kind in ('decoder', 'encoder') for kv in ('key', 'value')]
    logits = [('logits', f, [1, 'sequence', 51866])]
    graph(root / 'onnx/decoder_model.onnx', [('input_ids', i, [1, 'sequence']),
          ('encoder_hidden_states', f, [1, 1500, 1280])], logits + present)
    graph(root / 'onnx/decoder_with_past_model.onnx', [('input_ids', i, [1, 1])] +
          [(name.replace('present.', 'past_key_values.'), typ, shape) for name, typ, shape in present],
          logits + [entry for entry in present if '.decoder.' in entry[0]])
    (root / 'tokenizer.json').write_text(json.dumps({'model': {'vocab': companions.TOKENS}, 'added_tokens': []}))
    (root / 'config.json').write_text(json.dumps({'d_model':1280,'n_layers':32,'n_heads':20,'ffn':5120,'n_mels':128,'max_src':1500}))
    (root / 'decode_rail').mkdir()
    (root / 'decode_rail/meta.json').write_text('{}')


class WhisperServingCompanionsTests(unittest.TestCase):
    def test_actual_preprocessor_export_qualification_and_negative_controls(self):
        import numpy as np
        import torch
        from transformers import WhisperFeatureExtractor
        from export_whisper_preproc import WhisperPreproc, N_SAMPLES
        torch.set_num_threads(1)
        module = WhisperPreproc(np.asarray(WhisperFeatureExtractor(feature_size=128).mel_filters)).eval()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'preprocessor.onnx'
            torch.onnx.export(module, (torch.zeros(1, N_SAMPLES),), str(path),
                              input_names=['waveform'], output_names=['input_features'],
                              opset_version=17, dynamo=False)
            exported = path.read_bytes()
            before, _ = companions.inspect_graph(path)
            self.assertEqual(before['inputs'], {'waveform': [1, [1, 480000]]})
            companions.qualify_preprocessor(root)
            after, _ = companions.inspect_graph(path)
            self.assertEqual(after, companions.expected_signatures()['preprocessor.onnx'])
            # Only the formerly symbolic final dimension may change.
            original_graph = onnx.load_model_from_string(exported)
            original_graph.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 3000
            self.assertEqual(onnx.load(path), original_graph)
            for bad in ('shape', 'nonfinite', 'input', 'static-metadata'):
                with self.subTest(bad=bad):
                    model = onnx.load_model_from_string(exported)
                    if bad == 'input':
                        model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 479999
                    elif bad == 'static-metadata':
                        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = 2999
                    else:
                        for node in model.graph.node:
                            for index, name in enumerate(node.output):
                                if name == 'input_features':
                                    node.output[index] = 'original_features'
                        if bad == 'shape':
                            for name, value in (('start', 0), ('end', 2999), ('axis', 2)):
                                model.graph.initializer.append(helper.make_tensor(name, TensorProto.INT64, [1], [value]))
                            model.graph.node.append(helper.make_node('Slice', ['original_features', 'start', 'end', 'axis'], ['input_features']))
                        else:
                            model.graph.initializer.append(helper.make_tensor('nan_factor', TensorProto.FLOAT, [], [float('nan')]))
                            model.graph.node.append(helper.make_node('Mul', ['original_features', 'nan_factor'], ['input_features']))
                    onnx.save(model, path)
                    bad_bytes = path.read_bytes()
                    with self.assertRaises(ValueError):
                        companions.qualify_preprocessor(root)
                    self.assertEqual(path.read_bytes(), bad_bytes)

    def test_installed_exporter_and_cache_protocol_with_tiny_random_model(self):
        import torch
        from transformers import WhisperConfig, WhisperForConditionalGeneration
        from optimum.exporters.onnx import export_models
        from optimum.exporters.onnx.model_configs import WhisperOnnxConfig
        from optimum.exporters.utils import get_encoder_decoder_models_for_export
        torch.set_num_threads(1)
        config = WhisperConfig(d_model=8, encoder_layers=1, decoder_layers=1,
            encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=16,
            decoder_ffn_dim=16, num_mel_bins=128, vocab_size=32, max_source_positions=16,
            max_target_positions=64, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)
        config._attn_implementation = 'eager'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'checkpoint'
            original = WhisperForConditionalGeneration(config).half().eval()
            original.save_pretrained(checkpoint, safe_serialization=True)
            from safetensors.torch import load_file, save_file
            saved = load_file(checkpoint / 'model.safetensors')
            expected = {name: value.float() for name, value in saved.items()
                        if not name.startswith('model.encoder.')}
            expected['proj_out.weight'] = expected['model.decoder.embed_tokens.weight']
            del original
            original_float = WhisperForConditionalGeneration.float
            widened = []
            def checked_float(model):
                self.assertIsNone(model.model.encoder)
                self.assertTrue(all(value.dtype == torch.float16 for value in model.parameters()))
                widened.append(True)
                return original_float(model)
            with patch.object(WhisperForConditionalGeneration, 'float', checked_float):
                model = companions.load_decoder_for_export(checkpoint)
            self.assertEqual(widened, [True])
            self.assertIsNone(model.model.encoder)
            self.assertIs(model.proj_out.weight, model.model.decoder.embed_tokens.weight)
            self.assertEqual(set(model.state_dict()), set(expected))
            for name, tensor in model.state_dict().items():
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertTrue(torch.equal(tensor, expected[name]), name)
            parts = get_encoder_decoder_models_for_export(model,
                WhisperOnnxConfig(model.config, task='automatic-speech-recognition', use_past=True))
            parts.pop('encoder_model')
            for part, _ in parts.values():
                self.assertIsNone(part.model.encoder)
                self.assertFalse(any(name.startswith('model.encoder.') for name, _ in part.named_parameters()))
            export_models(parts, root / 'onnx', opset=17, disable_dynamic_axes_fix=True,
                          input_shapes={'batch_size':1, 'decoder_sequence_length':4})
            companions.qualify_decoders(root, hidden=8, heads=2, layers=1, vocab=32, token=1)
            signatures, _ = companions.inspect_graph(root / 'onnx/decoder_model.onnx')
            self.assertEqual(signatures['outputs']['present.0.decoder.key'][1][1], 2)
            self.assertEqual(signatures['outputs']['present.0.decoder.key'][1][3], 4)
            self.assertFalse((root / 'onnx/encoder_model.onnx').exists())
            # Real ORT negative: keep the ABI but force logits to NaN in the exported graph.
            graph_path = root / 'onnx/decoder_model.onnx'
            exported = onnx.load(graph_path)
            for node in exported.graph.node:
                for index, name in enumerate(node.output):
                    if name == 'logits':
                        node.output[index] = 'finite_logits'
            exported.graph.initializer.append(helper.make_tensor('nan_factor', TensorProto.FLOAT, [], [float('nan')]))
            exported.graph.node.append(helper.make_node('Mul', ['finite_logits', 'nan_factor'], ['logits']))
            onnx.save(exported, graph_path)
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                companions.qualify_decoders(root, hidden=8, heads=2, layers=1, vocab=32, token=1)
            saved.pop('model.decoder.layer_norm.weight')
            save_file(saved, checkpoint / 'model.safetensors', metadata={'format': 'pt'})
            with self.assertRaisesRegex(ValueError, 'incomplete Whisper checkpoint'):
                companions.load_decoder_for_export(checkpoint)


    def test_decoder_loader_refuses_missing_or_non_f16_checkpoint(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                companions.load_decoder_for_export(checkpoint)
            for tensors in ({}, {'weight': torch.zeros(1, dtype=torch.float32)},
                            {'half': torch.zeros(1, dtype=torch.float16), 'float': torch.zeros(1)}):
                with self.subTest(tensors=list(tensors)):
                    save_file(tensors, checkpoint / 'model.safetensors')
                    with self.assertRaisesRegex(ValueError, 'F16'):
                        companions.load_decoder_for_export(checkpoint)

    def test_assembly_issues_derived_receipt_without_mutating_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / 'source'
            source.mkdir()
            fixture(source)
            for rel in (*companions.GRAPHS, 'tokenizer.json'):
                (source / rel).unlink()
            repo = Path(__file__).resolve().parents[1]
            iron = next(line.split('=', 1)[1].split('#', 1)[0].strip()
                        for line in (repo / 'toolchain.lock').read_text().splitlines()
                        if line.startswith('IRON_SOURCE_COMMIT='))
            (source / 'decode_rail/meta.json').write_text(json.dumps({'iron': {'commit':iron, 'dirty':False, 'identity':f'pinned:{iron}'}}))
            recipe = dispatch.Recipe('whisper-turbo','scenarios/asr-whisper-turbo.toml',(),(str(source),),())
            dispatch.write_receipt(recipe, 'a' * 64)
            before = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            checkpoint = base / 'checkpoint'; checkpoint.mkdir()
            (checkpoint / 'config.json').write_text(json.dumps(companions.HF_SHAPE))
            inventory = base / 'inventory.json'; inventory.write_text('{}')
            donor = base / 'donor'; donor.mkdir(); fixture(donor)
            def fake_export(checkpoint, out):
                for rel in (*companions.GRAPHS, 'tokenizer.json'):
                    shutil.copy2(donor / rel, out / rel)
                companions.record(out)
            args = Namespace(source=source, out=base/'derived', checkpoint=checkpoint,
                             source_inventory=inventory,
                             source_receipt_sha256=companions.sha256(source/'.model-recipe.json'))
            with patch.object(companions, 'producer_witness', return_value={'producer':'verified'}), patch.object(companions, 'export', side_effect=fake_export):
                companions.assemble(args)
            self.assertEqual(before, {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()})
            record = json.loads((args.out/'.model-recipe.json').read_text())
            self.assertEqual([cmd['step'] for cmd in record['commands']], ['assemble-serving-companions'])
            self.assertNotEqual(record['recipe_key'], 'a' * 64)
            companions.validate(args.out)

    def test_complete_closure_and_missing_or_wrong_architecture_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            companions.record(root)
            recipe = dispatch.Recipe('whisper-turbo', 'scenarios/asr-whisper-turbo.toml', (), (str(root),), ())
            dispatch.validate_recipe_outputs(recipe)
            original = (root / 'preprocessor.onnx').read_bytes()
            (root / 'preprocessor.onnx').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                dispatch.validate_recipe_outputs(recipe)
            (root / 'preprocessor.onnx').write_bytes(original)
            graph(root / 'preprocessor.onnx', [('waveform', TensorProto.FLOAT, [1,480000])],
                  [('input_features', TensorProto.FLOAT, [1,80,3000])])
            with self.assertRaisesRegex(ValueError, 'shape'):
                companions.record(root)

    def test_missing_external_data_and_tampered_tokenizer_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / 'onnx/decoder_model.onnx'
            model = onnx.load(path)
            tensor = model.graph.initializer.add()
            tensor.name = 'weight'
            tensor.data_type = TensorProto.FLOAT
            tensor.dims.append(1)
            tensor.data_location = TensorProto.EXTERNAL
            item = tensor.external_data.add(); item.key = 'location'; item.value = 'missing.bin'
            onnx.save(model, path)
            with self.assertRaises(FileNotFoundError):
                companions.record(root)
            (root / 'onnx/missing.bin').write_bytes(b'1234')
            companions.record(root)
            (root / 'onnx/missing.bin').write_bytes(b'4321')
            with self.assertRaisesRegex(ValueError, 'hash'):
                companions.validate(root)
            (root / 'tokenizer.json').write_text('{}')
            with self.assertRaises((ValueError, KeyError)):
                companions.record(root)

    def test_missing_companions_reject_before_prepared_receipt_acceptance(self):
        import prepared_install
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'decode_rail').mkdir()
            (root / 'decode_rail/meta.json').write_text('{}')
            recipe = dispatch.Recipe('whisper-turbo', 'scenarios/asr-whisper-turbo.toml', (), (str(root),), ())
            dispatch.write_receipt(recipe, 'a' * 64)
            witness = {'recipe_key': 'a' * 64, 'sha256': companions.sha256(dispatch.manifest_path(recipe))}
            with self.assertRaises((ValueError, FileNotFoundError)):
                prepared_install.verify_receipt(recipe, witness)


if __name__ == '__main__':
    unittest.main()
