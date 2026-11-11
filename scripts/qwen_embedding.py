#!/usr/bin/env python3
"""Offline Qwen3 Embedding reference. Never downloads or selects a device implicitly."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import tempfile

import torch

TASK = 'Given a web search query, retrieve relevant passages that answer the query'
MAX_TOKENS = 512


def model_pin(model_id):
    pins = json.loads(Path(__file__).with_name('qwen_embedding_pins.json').read_text())
    if model_id not in pins:
        raise ValueError(f'unsupported embedding model: {model_id}')
    return pins[model_id]


def prepare_texts(texts, input_type):
    if input_type not in ('query', 'document'):
        raise ValueError('input_type must be query or document')
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError('input must be a nonempty string array')
    for i, text in enumerate(texts):
        if not isinstance(text, str) or not text:
            raise ValueError(f'item {i} must be a nonempty string')
    prefix = f'Instruct: {TASK}\nQuery:' if input_type == 'query' else ''
    return [prefix + text for text in texts]


def tokenized_items(tokenizer, texts, input_type, max_tokens):
    """Validate every input once, before any forward, and retain its unpadded tokens."""
    if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_TOKENS:
        raise ValueError(f'max_tokens must be 1..={MAX_TOKENS}')
    if tokenizer.padding_side != 'left':
        raise ValueError('reference tokenizer must use left padding')
    prepared = prepare_texts(texts, input_type)
    unpadded = tokenizer(prepared, padding=False, truncation=False, add_special_tokens=True)
    for i, ids in enumerate(unpadded['input_ids']):
        if not 1 <= len(ids) <= max_tokens:
            raise ValueError(f'item {i} has {len(ids)} tokens; expected 1..={max_tokens}, including instruction')
    return [{name: values[i] for name, values in unpadded.items()} for i in range(len(prepared))]


def tokenize(tokenizer, texts, input_type, max_tokens):
    return tokenizer.pad(tokenized_items(tokenizer, texts, input_type, max_tokens),
                         padding=True, return_tensors='pt')


def pool(hidden, mask, dimension):
    if hidden.ndim != 3 or mask.ndim != 2 or hidden.shape[:2] != mask.shape:
        raise ValueError('hidden states and attention mask shapes disagree')
    if hidden.shape[0] == 0 or hidden.shape[1] == 0 or hidden.shape[2] != dimension or dimension <= 0:
        raise ValueError('empty output or incorrect full embedding dimension')
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('attention mask must be binary')
    valid = mask.bool()
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    first = torch.where(valid, positions, mask.shape[1]).min(dim=1).values
    last = torch.where(valid, positions, -1).max(dim=1).values
    if (last < 0).any() or not torch.equal(valid.sum(dim=1), last - first + 1):
        raise ValueError('attention mask must contain one nonempty contiguous token span')
    values = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
    norms = values.norm(dim=1, keepdim=True)
    if not torch.isfinite(values).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError('invalid final-token embedding: nonfinite or zero norm')
    return values / norms


def valid_batch_size(batch_size):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a strictly positive integer')
    return batch_size


@torch.inference_mode()
def encode_with_evidence(model, tokenizer, texts, input_type, max_tokens, dimension, batch_size=1):
    """Execute bounded forwards and return vectors plus submitted IDs/masks in input order."""
    batch_size = valid_batch_size(batch_size)
    items = tokenized_items(tokenizer, texts, input_type, max_tokens)
    model.eval()
    vectors, input_ids, attention_mask, forward_batch_sizes = [], [], [], []
    for start in range(0, len(items), batch_size):
        batch = tokenizer.pad(items[start:start + batch_size], padding=True, return_tensors='pt')
        output = model(**batch, use_cache=False)
        vectors.append(pool(output.last_hidden_state, batch['attention_mask'], dimension))
        input_ids.extend(batch['input_ids'].tolist())
        attention_mask.extend(batch['attention_mask'].tolist())
        forward_batch_sizes.append(batch['input_ids'].shape[0])
    return torch.cat(vectors), input_ids, attention_mask, forward_batch_sizes


@torch.inference_mode()
def encode(model, tokenizer, texts, input_type, max_tokens, dimension, batch_size=1):
    return encode_with_evidence(model, tokenizer, texts, input_type, max_tokens, dimension, batch_size)[0]


def verify_files(snapshot, pin):
    """Verify upstream LFS SHA256 or Git blob identity, then report content SHA256."""
    hashes = {}
    for name, expected in pin['files'].items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != expected['size']:
            raise ValueError(f'missing or wrong-sized pinned file: {name}')
        sha256 = hashlib.sha256()
        git = hashlib.sha1(f"blob {expected['size']}\0".encode())
        with path.open('rb') as stream:
            while block := stream.read(1024 * 1024):
                sha256.update(block)
                git.update(block)
        digest = sha256.hexdigest()
        if ('sha256' in expected and digest != expected['sha256']) or ('git_sha1' in expected and git.hexdigest() != expected['git_sha1']):
            raise ValueError(f'pinned content mismatch: {name}')
        hashes[name] = digest
    return hashes


def profile(pin, hashes, max_tokens):
    return dict(model=pin['repo'], revision=pin['revision'], files=hashes,
                query_prefix=f'Instruct: {TASK}\nQuery:', document_prefix='',
                special_tokens='tokenizer-add-special-tokens-no-chat-template',
                padding='left', truncation='reject', max_tokens=max_tokens,
                pooling='final-backbone-last-valid-token', normalize='l2-float32',
                dimension=pin['dimension'], format=1)


def load_reference(snapshot, pin, dtype):
    from transformers import AutoModel, AutoTokenizer
    hashes = verify_files(snapshot, pin)
    # Only pinned files are visible to HF: stray special_tokens_map.json or model code must not
    # change preprocessing. Weights stay in place; the temporary view contains symlinks, not copies.
    with tempfile.TemporaryDirectory(prefix='qwen-embedding-reference-') as tmp:
        view = Path(tmp)
        for name in pin['files']:
            dest = view / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to((snapshot / name).resolve())
        tokenizer = AutoTokenizer.from_pretrained(view, local_files_only=True, trust_remote_code=False,
                                                  use_fast=True, padding_side='left')
        model = AutoModel.from_pretrained(view, local_files_only=True, trust_remote_code=False,
                                         use_safetensors=True, dtype=dtype,
                                         attn_implementation='eager').cpu().eval()
    if model.config.hidden_size != pin['dimension'] or model.config.model_type != 'qwen3':
        raise ValueError('loaded model does not match the pinned Qwen embedding backbone')
    return model, tokenizer, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['qwen3-embedding-0.6b', 'qwen3-embedding-4b'])
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='JSON string array; kept local')
    parser.add_argument('--output', type=Path, required=True, help='new JSON evidence file; never overwritten')
    parser.add_argument('--input-type', choices=['query', 'document'], default='document')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='strictly positive CPU forward batch size; defaults to bounded single-item forwards')
    args = parser.parse_args()
    try:
        valid_batch_size(args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error('output already exists')
    input_bytes = args.input.read_bytes()
    texts = json.loads(input_bytes)
    prepare_texts(texts, args.input_type)
    pin = model_pin(args.model)
    torch.set_num_threads(2)
    start = time.perf_counter()
    model, tokenizer, hashes = load_reference(args.snapshot, pin, getattr(torch, args.dtype))
    load_s = time.perf_counter() - start
    start = time.perf_counter()
    vectors, input_ids, attention_mask, forward_batch_sizes = encode_with_evidence(
        model, tokenizer, texts, args.input_type, args.max_tokens, pin['dimension'], args.batch_size)
    first_s = time.perf_counter() - start
    start = time.perf_counter()
    repeated = encode(model, tokenizer, texts, args.input_type, args.max_tokens, pin['dimension'], args.batch_size)
    warm_s = time.perf_counter() - start
    import transformers
    from evaluate_embeddings import text_identity
    record = dict(qualification='CPU-reference-only', profile=profile(pin, hashes, args.max_tokens),
                  runtime=dict(device='cpu', dtype=args.dtype, torch=torch.__version__,
                               transformers=transformers.__version__, threads=2,
                               attention='eager', use_cache=False, batch_size=args.batch_size,
                               forward_batch_sizes=forward_batch_sizes),
                  code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ['qwen_embedding.py', 'qwen_embedding_pins.json']},
                  input_type=args.input_type, input_sha256=hashlib.sha256(input_bytes).hexdigest(),
                  texts_sha256=text_identity(texts),
                  input_ids=input_ids, attention_mask=attention_mask,
                  vectors=vectors.tolist(), repeat_max_abs=(vectors - repeated).abs().max().item(),
                  timings=dict(verified_load_s=load_s, first_encode_s=first_s, warm_encode_s=warm_s))
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
