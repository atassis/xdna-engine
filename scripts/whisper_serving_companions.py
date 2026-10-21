#!/usr/bin/env python3
"""Whisper-turbo serving closure: decoder-only local export and verified new-root assembly.

Export/record requires the existing ONNX SDK; validation is stdlib-only for install.
The companion witness binds inspected graph signatures and every graph/data file byte.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tomllib

MANIFEST = 'serving-companions.json'
GRAPHS = ('preprocessor.onnx', 'onnx/decoder_model.onnx', 'onnx/decoder_with_past_model.onnx')
TOKENS = {'<|endoftext|>': 50257, '<|startoftranscript|>': 50258, '<|en|>': 50259,
          '<|translate|>': 50359, '<|transcribe|>': 50360, '<|notimestamps|>': 50364}
HF_SHAPE = {'model_type': 'whisper', 'd_model': 1280, 'encoder_layers': 32, 'decoder_layers': 4,
            'encoder_attention_heads': 20, 'decoder_attention_heads': 20, 'num_mel_bins': 128,
            'vocab_size': 51866, 'max_source_positions': 1500}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_owner(root, checkpoint=None):
    scenario = tomllib.loads((Path(__file__).resolve().parents[1] / 'scenarios/asr-whisper-turbo.toml').read_text())['model']
    expected = {'hidden':1280, 'n_layers':32, 'n_decoder_layers':4, 'n_heads':20, 'head_dim':64, 'n_mels':128}
    if any(scenario.get(k) != v for k, v in expected.items()):
        raise ValueError('Whisper companion contract disagrees with scenario owner')
    if checkpoint is not None:
        config = json.loads((checkpoint / 'config.json').read_text())
        if any(config.get(k) != v for k, v in HF_SHAPE.items()):
            raise ValueError('checkpoint is not the scenario Whisper-turbo architecture')
    config = json.loads((root / 'config.json').read_text())
    for key, value in {'d_model':1280, 'n_layers':32, 'n_heads':20, 'n_mels':128, 'max_src':1500}.items():
        if config.get(key) != value:
            raise ValueError(f'encoder architecture mismatch: {key}')


def check_tokenizer(path):
    data = json.loads(path.read_text())
    vocab = data['model']['vocab'] | {entry['content']: entry['id'] for entry in data.get('added_tokens', [])}
    if any(vocab.get(token) != value for token, value in TOKENS.items()):
        raise ValueError('tokenizer is not compatible with Whisper-turbo token IDs')


def expected_signatures():
    # FLOAT=1, INT64=7 in ONNX. None denotes a dynamic or fixed valid sequence axis.
    kv = [(f'present.{layer}.{kind}.{part}', [1, [None, 20, None, 64]])
          for layer in range(4) for kind in ('decoder', 'encoder') for part in ('key', 'value')]
    logits = {'logits': [1, [None, None, 51866]]}
    return {
        GRAPHS[0]: {'inputs': {'waveform': [1, [1,480000]]}, 'outputs': {'input_features': [1,[1,128,3000]]}},
        GRAPHS[1]: {'inputs': {'input_ids': [7,[None,None]], 'encoder_hidden_states': [1,[None,None,1280]]},
                    'outputs': logits | dict(kv)},
        GRAPHS[2]: {'inputs': {'input_ids': [7,[None,None]]} |
                    {name.replace('present.', 'past_key_values.'): spec for name, spec in kv},
                    'outputs': logits | {name: spec for name, spec in kv if '.decoder.' in name}},
    }


def check_signatures(graphs):
    for graph, sections in expected_signatures().items():
        for section, expected in sections.items():
            actual = graphs[graph][section]
            if set(actual) != set(expected):
                raise ValueError(f'Whisper graph names mismatch: {graph} {section}')
            for name, (dtype, shape) in expected.items():
                got_type, got_shape = actual[name]
                if (got_type != dtype or len(got_shape) != len(shape)
                        or any(want is not None and want != got for want, got in zip(shape, got_shape))):
                    raise ValueError(f'Whisper graph type/shape mismatch: {graph} {name}')


def inspect_graph(path):
    import onnx
    from onnx.external_data_helper import _get_all_tensors
    model = onnx.load(path, load_external_data=False)
    files = {path}
    for tensor in _get_all_tensors(model):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {entry.key: entry.value for entry in tensor.external_data}
        rel = Path(fields['location'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('external ONNX data escapes companion directory')
        target = path.parent / rel
        if not target.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('external ONNX data resolves outside companion directory')
        size = target.stat().st_size
        offset = int(fields.get('offset', 0))
        length = int(fields.get('length', size - offset))
        if offset < 0 or length <= 0 or offset + length > size:
            raise ValueError(f'external ONNX data range invalid: {target}')
        files.add(target)
    def signature(values):
        return {value.name: [value.type.tensor_type.elem_type,
                [dim.dim_value if dim.HasField('dim_value') else dim.dim_param
                 for dim in value.type.tensor_type.shape.dim]] for value in values}
    return {'inputs': signature(model.graph.input), 'outputs': signature(model.graph.output)}, files


def record(root):
    check_owner(root)
    check_tokenizer(root / 'tokenizer.json')
    graphs, files = {}, {root / 'tokenizer.json'}
    for rel in GRAPHS:
        graphs[rel], members = inspect_graph(root / rel)
        files.update(members)
    check_signatures(graphs)
    manifest = {'version': 1, 'graphs': graphs,
                'files': {str(path.relative_to(root)): sha256(path) for path in sorted(files)}}
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    validate(root)


def validate(root):
    check_owner(root)
    check_tokenizer(root / 'tokenizer.json')
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest['version'] != 1:
        raise ValueError('unsupported Whisper serving companion witness')
    check_signatures(manifest['graphs'])
    required = {*GRAPHS, 'tokenizer.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('incomplete Whisper companion file inventory')
    for rel, digest in manifest['files'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or sha256(root / rel) != digest:
            raise ValueError(f'Whisper companion hash mismatch: {rel}')


def qualify_preprocessor(root):
    """Qualify fixed-input mel output before specializing its missing static frame axis."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    path = root / GRAPHS[0]
    signature, _ = inspect_graph(path)
    expected = expected_signatures()[GRAPHS[0]]
    if signature['inputs'] != expected['inputs'] or set(signature['outputs']) != {'input_features'}:
        raise ValueError('preprocessor must have exactly the fixed FLOAT waveform[1,480000] input')
    dtype, shape = signature['outputs']['input_features']
    if (dtype != 1 or len(shape) != 3 or shape[:2] != [1, 128]
            or isinstance(shape[2], int) and shape[2] != 3000):
        raise ValueError('preprocessor output type/shape metadata disagrees with the fixed mel contract')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    output = session.run(['input_features'], {'waveform': np.zeros((1, 480000), dtype=np.float32)})[0]
    if output.dtype != np.float32 or output.shape != (1, 128, 3000):
        raise ValueError('preprocessor actual output type/shape disagrees with the fixed mel contract')
    if not np.isfinite(output).all():
        raise ValueError('preprocessor returned non-finite mel features')
    del session
    if not isinstance(shape[2], int):
        model = onnx.load(path, load_external_data=False)
        # No graph/input/other output axis rewrite: only specialize the observed missing frame count.
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value = output.shape[2]
        onnx.save(model, path)


def qualify_decoders(root, hidden=1280, heads=20, layers=4, vocab=51866, token=50258):
    """Observe cache dimensions with the Rust two-step protocol, bounded to two CPU threads."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    cache = {}
    for index, rel in enumerate(GRAPHS[1:]):
        path = root / rel
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        inputs = {'input_ids': np.full((1, 4 if index == 0 else 1), token, dtype=np.int64)}
        if index == 0:
            inputs['encoder_hidden_states'] = np.zeros((1, 1500, hidden), dtype=np.float32)
        else:
            inputs.update(cache)
        names = ['logits'] + [f'present.{layer}.{kind}.{part}' for layer in range(layers)
                for kind in (('decoder', 'encoder') if index == 0 else ('decoder',)) for part in ('key', 'value')]
        outputs = session.run(names, inputs)
        if any(not np.isfinite(value).all() for value in outputs):
            raise ValueError('exported decoder returned non-finite logits or cache')
        if outputs[0].shape != (1, 4 if index == 0 else 1, vocab):
            raise ValueError('exported decoder logits disagree with source architecture')
        observed = {}
        for name, value in zip(names[1:], outputs[1:]):
            expected_length = 1500 if '.encoder.' in name else (4 if index == 0 else 5)
            if value.shape != (1, heads, expected_length, hidden // heads):
                raise ValueError(f'exported decoder cache shape mismatch: {name}')
            observed[name] = value.shape
            cache[name.replace('present.', 'past_key_values.')] = value
        # Legacy tracing leaves constant head axes symbolic; specialize only observed axes.
        model = onnx.load(path, load_external_data=False)
        for output in model.graph.output:
            if output.name in observed:
                for axis in (1, 3):
                    output.type.tensor_type.shape.dim[axis].dim_value = observed[output.name][axis]
        onnx.save(model, path)
        del session


def load_decoder_for_export(checkpoint):
    """Load exact native F16, release the unused encoder, then widen only retained weights."""
    import gc
    import torch
    from safetensors import safe_open
    from transformers import WhisperForConditionalGeneration

    weights = checkpoint / 'model.safetensors'
    if not weights.is_file():
        raise FileNotFoundError(f'missing native F16 checkpoint: {weights}')
    # Header-only gate: requesting float16 from Transformers must not downcast F32 sources.
    with safe_open(weights, framework='pt', device='cpu') as tensors:
        names = list(tensors.keys())
        if not names or any(tensors.get_slice(name).get_dtype() != 'F16' for name in names):
            raise ValueError('decoder export requires an entirely native F16 safetensors checkpoint')
    model, info = WhisperForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float16, attn_implementation='eager', output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'incomplete Whisper checkpoint: {info}')
    # Optimum's two decoder partitions retain this entire model, not decoder-only modules.
    # Their forward calls supply encoder_outputs, so the encoder is never needed or exported.
    model.model.encoder = None
    gc.collect()
    model.float()
    if model.config.tie_word_embeddings and model.proj_out.weight is not model.model.decoder.embed_tokens.weight:
        raise ValueError('Whisper decoder embedding/output weight tie was lost')
    return model.eval()


def export(checkpoint, root):
    check_owner(root, checkpoint)
    for name in (*GRAPHS, 'tokenizer.json', MANIFEST):
        if (root / name).exists():
            raise ValueError(f'refusing to overwrite serving companion: {name}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import numpy as np
    from transformers import WhisperFeatureExtractor
    from optimum.exporters.onnx import export_models
    from optimum.exporters.onnx.model_configs import WhisperOnnxConfig
    from optimum.exporters.utils import get_encoder_decoder_models_for_export
    from export_whisper_preproc import WhisperPreproc, N_SAMPLES
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = load_decoder_for_export(checkpoint)
    config = WhisperOnnxConfig(model.config, task='automatic-speech-recognition', use_past=True)
    parts = get_encoder_decoder_models_for_export(model, config)
    parts.pop('encoder_model')
    if set(parts) != {'decoder_model', 'decoder_with_past_model'}:
        raise ValueError('unexpected Optimum decoder export partition')
    export_models(parts, root / 'onnx', opset=17, device='cpu',
                  input_shapes={'batch_size':1, 'decoder_sequence_length':4}, disable_dynamic_axes_fix=True)
    del parts, model
    import gc
    gc.collect()
    qualify_decoders(root)
    feature = WhisperFeatureExtractor.from_pretrained(str(checkpoint), local_files_only=True)
    module = WhisperPreproc(np.asarray(feature.mel_filters)).eval()
    torch.onnx.export(module, (torch.zeros(1, N_SAMPLES),), str(root / GRAPHS[0]),
                      input_names=['waveform'], output_names=['input_features'], opset_version=17, dynamo=False)
    qualify_preprocessor(root)
    shutil.copy2(checkpoint / 'tokenizer.json', root / 'tokenizer.json')
    from tokenizers import Tokenizer
    Tokenizer.from_file(str(root / 'tokenizer.json'))
    record(root)


def producer_witness(repo, inventory_path):
    inventory = json.loads(inventory_path.read_text())['public_files']
    exact = {'toolchain.lock', 'scripts/extract_whisper_encoder.py', 'scripts/build_whisper_rail.sh',
             'scripts/build_llm_decode.sh', 'scripts/dump_llm_weights.py'}
    prefixes = ('scripts/', 'designs/decode_fused/', 'aie_kernels/', 'third_party/iron/')
    # Dispatcher/installer repairs do not change the retained encoder or rail producers.
    reviewed_dispatch_seam = {'scripts/model_artifact_dispatch.py', 'scripts/test_model_artifact_dispatch.py'}
    selected = {name: entry for name, entry in inventory.items()
                if (name in exact or name.startswith(prefixes)) and name not in reviewed_dispatch_seam}
    if not exact.issubset(selected):
        raise ValueError('source inventory lacks Whisper producer inputs')
    for name, entry in selected.items():
        if sha256(repo / name) != entry['sha256']:
            raise ValueError(f'original Whisper producer source changed: {name}')
    return {name: entry['sha256'] for name, entry in selected.items()}


def assemble(args):
    import model_artifact_dispatch as dispatch
    repo = Path(__file__).resolve().parents[1]
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists() or out.is_symlink() or out.is_relative_to(source):
        raise ValueError('assembly requires a new output outside the original tree')
    receipt = source / '.model-recipe.json'
    if sha256(receipt) != args.source_receipt_sha256:
        raise ValueError('original model receipt hash mismatch')
    original = json.loads(receipt.read_text())
    if (original['model'] != 'whisper-turbo' or original['scenario'] != 'scenarios/asr-whisper-turbo.toml'
            or dispatch.output_digest((str(source),)) != original['output_digest']):
        raise ValueError('original Whisper receipt/output digest mismatch')
    check_owner(source, args.checkpoint)
    from check_iron_artifact_provenance import validate as validate_iron
    expected_iron = next(line.split('=', 1)[1].split('#', 1)[0].strip()
                         for line in (repo / 'toolchain.lock').read_text().splitlines()
                         if line.startswith('IRON_SOURCE_COMMIT='))
    error = validate_iron(source / 'decode_rail/meta.json', expected_iron)
    if error:
        raise ValueError(error)
    producers = producer_witness(repo, args.source_inventory)
    inventory_sha256 = sha256(args.source_inventory)
    assembly_sources = {name: sha256(repo / name) for name in
                        ('scripts/whisper_serving_companions.py', 'scripts/export_whisper_preproc.py')}
    checkpoint_digest = dispatch.source_digest((str(args.checkpoint),))
    dispatch.copy_tree(source, out)
    (out / '.model-recipe.json').unlink()
    export(args.checkpoint, out)
    if (sha256(receipt) != args.source_receipt_sha256 or producer_witness(repo, args.source_inventory) != producers
            or sha256(args.source_inventory) != inventory_sha256
            or any(sha256(repo / name) != digest for name, digest in assembly_sources.items())
            or dispatch.source_digest((str(args.checkpoint),)) != checkpoint_digest
            or dispatch.output_digest((str(source),)) != original['output_digest']):
        raise ValueError('assembly inputs changed during export')
    derivation = {'operation': 'whisper-serving-assembly-v1', 'source_receipt_sha256': args.source_receipt_sha256,
                  'source_output_digest': original['output_digest'], 'source_inventory_sha256': inventory_sha256,
                  'producer_files': producers, 'assembly_sources': assembly_sources, 'checkpoint_digest': checkpoint_digest,
                  'companion_manifest_sha256': sha256(out / MANIFEST)}
    (out / 'serving-assembly.json').write_text(json.dumps(derivation, indent=2, sort_keys=True) + '\n')
    key = hashlib.sha256(json.dumps(derivation, sort_keys=True).encode()).hexdigest()
    operation = dispatch.command('assemble-serving-companions', [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    recipe = dispatch.Recipe('whisper-turbo', 'scenarios/asr-whisper-turbo.toml', (str(args.checkpoint),), (str(out),), (operation,))
    dispatch.validate_recipe_outputs(recipe)
    dispatch.write_receipt(recipe, key)
    if not dispatch.cache_hit(recipe, key):
        raise ValueError('derived Whisper serving receipt did not verify')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('export', 'validate', 'assemble'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--source-receipt-sha256')
    parser.add_argument('--source-inventory', type=Path)
    args = parser.parse_args()
    if args.action == 'validate':
        validate(args.out)
    elif args.action == 'export':
        if args.checkpoint is None:
            parser.error('export needs --checkpoint')
        export(args.checkpoint, args.out)
    else:
        if any(getattr(args, name) is None for name in ('checkpoint', 'source', 'source_receipt_sha256', 'source_inventory')):
            parser.error('assemble needs --checkpoint --source --source-receipt-sha256 --source-inventory')
        assemble(args)


if __name__ == '__main__':
    main()
