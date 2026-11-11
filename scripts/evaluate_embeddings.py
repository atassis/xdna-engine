#!/usr/bin/env python3
"""Dense retrieval evaluation and corresponding-vector diagnostics; no inference or indexing."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


def matrix(values, rows=None, dimension=None):
    raw = np.asarray(values)
    if raw.dtype.kind not in 'fiu':
        raise ValueError('vectors must contain real numbers')
    arr = raw.astype(np.float64)
    if arr.ndim != 2 or min(arr.shape) == 0 or (rows is not None and arr.shape[0] != rows):
        raise ValueError('incorrect vector count or shape')
    if dimension is not None and arr.shape[1] != dimension:
        raise ValueError('incorrect vector dimension')
    norms = np.linalg.norm(arr, axis=1)
    if not np.isfinite(arr).all() or not np.isfinite(norms).all() or not np.allclose(norms, 1., rtol=0, atol=1e-4):
        raise ValueError('vectors must be finite and L2 normalized within 1e-4')
    return arr


def evaluate(dataset, vectors):
    documents, queries = dataset['documents'], dataset['queries']
    ids = [d['id'] for d in documents]
    query_ids = [q['id'] for q in queries]
    if not ids or len(set(ids)) != len(ids) or not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError('empty or duplicate document/query IDs')
    docs = matrix(vectors['documents'], len(ids))
    qvecs = matrix(vectors['queries'], len(queries), docs.shape[1])
    cases = []
    for query, scores in zip(queries, qvecs @ docs.T):
        relevant = query['relevant']
        negatives = query['hard_negatives']
        scopes = dataset.get('candidate_groups')
        if scopes is not None and query['group'] not in scopes:
            raise ValueError('missing candidate scope for query group')
        candidates = scopes[query['group']] if scopes is not None else ids
        if (not candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= set(ids)
                or not relevant or not query['group'] or len(set(relevant)) != len(relevant)
                or not set(relevant + negatives) <= set(candidates) or set(relevant) & set(negatives)):
            raise ValueError('invalid candidates, relevance or hard-negative annotations')
        candidate_set = set(candidates)
        ranked = [d for d, _ in sorted(zip(ids, scores), key=lambda pair: (-pair[1], pair[0])) if d in candidate_set]
        ranks = {d: i + 1 for i, d in enumerate(ranked)}
        first = min(ranks[d] for d in relevant)
        errors = [d for d in ranked[:first-1] if d in negatives]
        cases.append(dict(id=query['id'], group=query['group'], ranking=ranked, candidate_count=len(ranked),
                          relevant_ranks={d: ranks[d] for d in relevant},
                          **{'recall@5': sum(ranks[d] <= 5 for d in relevant) / len(relevant),
                             'recall@10': sum(ranks[d] <= 10 for d in relevant) / len(relevant),
                             'mrr@10': 1. / first if first <= 10 else 0.,
                             'hard_negative_error_rate': float(bool(errors))},
                          hard_negatives_before_first_relevant=errors))
    def summary(rows):
        return {metric: sum(r[metric] for r in rows) / len(rows)
                for metric in ('recall@5', 'recall@10', 'mrr@10', 'hard_negative_error_rate')}
    return dict(overall=summary(cases), counts=dict(documents=len(ids), queries=len(queries)),
                groups={g: summary([c for c in cases if c['group'] == g]) for g in sorted({c['group'] for c in cases})},
                cases=cases)


def compare(reference, target):
    ref = matrix(reference)
    tgt = matrix(target, *ref.shape)
    error = np.abs(ref - tgt)
    cosine = np.sum(ref * tgt, axis=1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(tgt, axis=1))
    relative_l2 = np.linalg.norm(ref - tgt, axis=1) / np.linalg.norm(ref, axis=1)
    return dict(minimum_cosine=float(cosine.min()), mean_cosine=float(cosine.mean()),
                corresponding_cosines=cosine.tolist(), maximum_absolute_error=float(error.max()),
                maximum_relative_l2_error=float(relative_l2.max()))


def text_identity(texts):
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def load_json(source, strict=False):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'nonfinite JSON number: {value}')

    value = json.loads(source, object_pairs_hook=pairs if strict else dict,
                       parse_constant=constant if strict else float)
    if strict:
        validate_json(value)
    return value


def validate_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite JSON number')
    if isinstance(value, dict):
        for item in value.values():
            validate_json(item)
    elif isinstance(value, list):
        for item in value:
            validate_json(item)


def read_records(paths, dataset, sources=None, strict=False):
    records = {role: load_json(sources[role] if sources is not None else path.read_bytes(), strict)
               for role, path in paths.items()}
    for role, items in [('document', dataset['documents']), ('query', dataset['queries'])]:
        record = records[role]
        if record['input_type'] != role or record['texts_sha256'] != text_identity([item['text'] for item in items]):
            raise ValueError(f'{role} embeddings do not bind the exact ordered dataset texts')
    if records['document']['profile'] != records['query']['profile']:
        raise ValueError('query/document embedding spaces differ')
    return records


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(left, right):
    # Python's ordinary equality equates True with 1, including inside nested objects.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def number(value, name, minimum=None, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), f'{name} must be a finite number (not bool)')
    require(minimum is None or value >= minimum, f'{name} is below minimum')
    require(maximum is None or value <= maximum, f'{name} is above maximum')
    return value


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{64}', value) is not None,
            f'{name} must be a full SHA256 hex digest')


def hash_map(value, name, roles=False):
    require(isinstance(value, dict) and bool(value), f'{name} must be a nonempty hash map')
    if roles:
        require(set(value) == {'document', 'query'}, f'{name} requires exactly document and query')
    for key, item in value.items():
        require(isinstance(key, str) and bool(key), f'{name} has an invalid name')
        digest(item, f'{name}.{key}')


def sha256(source):
    return hashlib.sha256(source).hexdigest()


def canonical_ids(record, count, max_tokens, target, name):
    ids, masks = record['input_ids'], record['attention_mask']
    require(isinstance(ids, list) and isinstance(masks, list) and len(ids) == len(masks) == count,
            f'{name} token/mask row count mismatch')
    result = []
    for row, (tokens, mask) in enumerate(zip(ids, masks)):
        require(isinstance(tokens, list) and isinstance(mask, list) and 0 < len(tokens) == len(mask) <= max_tokens,
                f'{name} row {row} token/mask length or token limit invalid')
        require(all(type(t) is int and 0 <= t <= 2**32 - 1 for t in tokens),
                f'{name} row {row} token IDs must be u32 integers (not bool)')
        require(all(type(m) is int and m in (0, 1) for m in mask) and 1 in mask,
                f'{name} row {row} mask must be binary integers with a nonempty span')
        first = mask.index(1)
        require(mask == [0] * first + [1] * (len(mask) - first),
                f'{name} row {row} requires a contiguous left-padded valid span')
        require(not target or first == 0, f'{name} serial-unpadded target mask must be all ones')
        result.append(tokens[first:])
    return result


def qualification(policy_source, receipt_source, dataset_source, dataset, ref, target, record_sources):
    policy, receipt = load_json(policy_source, True), load_json(receipt_source, True)
    require(isinstance(policy, dict) and isinstance(receipt, dict), 'policy and receipt must be objects')
    require(type(policy['format']) is int and policy['format'] == 1, 'policy format must be integer 1')
    require(type(receipt['format']) is int and receipt['format'] == 1, 'receipt format must be integer 1')
    for key in ('stage', 'run_id'):
        require(isinstance(policy[key], str) and bool(policy[key]), f'policy {key} must be a nonempty string')
        require(exact(policy[key], receipt[key]), f'receipt {key} does not match policy')
    for key in ('dataset_sha256', 'capture_code_sha256', 'capture_binary_sha256'):
        digest(policy[key], f'policy.{key}')
    digest(receipt['capture_binary_sha256'], 'receipt.capture_binary_sha256')
    require(receipt['capture_binary_sha256'] == policy['capture_binary_sha256'], 'receipt capture binary mismatch')
    for key in ('reference_sha256', 'input_sha256'):
        hash_map(policy[key], f'policy.{key}', True)
    hash_map(receipt['records_sha256'], 'receipt.records_sha256', True)
    for obj, name in ((policy, 'policy'), (receipt, 'receipt')):
        hash_map(obj['target_artifacts'], f'{name}.target_artifacts')
        require('decode' in obj['target_artifacts'], f'{name} target_artifacts requires decode')
    require(exact(policy['target_artifacts'], receipt['target_artifacts']), 'receipt full artifact map mismatch')
    window = policy['window']
    require(isinstance(window, dict), 'policy window must be an object')
    start = number(window['start_unix_s'], 'window start')
    deadline = window['deadline_unix_s']
    if deadline is not None:
        deadline = number(deadline, 'window deadline')
    begun = number(receipt['started_unix_s'], 'receipt start')
    finished = number(receipt['finished_unix_s'], 'receipt finish')
    # Explicit null has no overall expiry; execution authorization/command limits belong to the runner.
    require(start <= begun <= finished and (deadline is None or finished < deadline),
            'receipt timestamps outside ordered policy window')
    require(sha256(dataset_source) == policy['dataset_sha256'], 'dataset full-file SHA256 mismatch')
    profile = policy['profile']
    require(isinstance(profile, dict), 'policy profile must be an object')
    dimension, max_tokens = profile['dimension'], profile['max_tokens']
    require(type(dimension) is int and dimension > 0, 'profile dimension must be a positive integer (not bool)')
    require(type(max_tokens) is int and max_tokens > 0, 'profile max_tokens must be a positive integer (not bool)')
    require(profile['padding'] == 'left', 'profile must specify left padding')
    limits = policy['thresholds']
    require(isinstance(limits, dict) and set(limits) == {
        'minimum_cosine', 'maximum_absolute_error', 'maximum_relative_l2_error', 'maximum_repeat_abs'},
        'thresholds must contain exactly the four numerical budgets')
    for key, value in limits.items():
        number(value, key, -1 if key == 'minimum_cosine' else 0, 1 if key == 'minimum_cosine' else None)
    failures, diagnostics, repeats = [], {}, {}
    for arm, records in (('reference', ref), ('target', target)):
        expected_runtime = policy[arm + '_runtime']
        require(isinstance(expected_runtime, dict) and set(expected_runtime) == {'document', 'query'},
                f'{arm}_runtime requires exactly document and query')
        repeats[arm] = {}
        for role, key in (('document', 'documents'), ('query', 'queries')):
            record, name = records[role], f'{arm}.{role}'
            expected_hash = policy['reference_sha256'][role] if arm == 'reference' else receipt['records_sha256'][role]
            require(sha256(record_sources[arm][role]) == expected_hash, f'{name} full-record SHA256 mismatch')
            require(exact(record['profile'], profile), f'{name} semantic profile mismatch')
            require(isinstance(expected_runtime[role], dict) and isinstance(record['runtime'], dict)
                    and exact(record['runtime'], expected_runtime[role]), f'{name} runtime mismatch')
            digest(record['input_sha256'], name + '.input_sha256')
            require(record['input_sha256'] == policy['input_sha256'][role], f'{name} raw input SHA256 mismatch')
            require(isinstance(record['vectors'], list) and all(
                isinstance(row, list) and all(type(v) in (int, float) for v in row)
                for row in record['vectors']), f'{name} vectors must contain real numbers (not bool)')
            matrix(record['vectors'], len(dataset[key]), dimension)
            repeat = number(record['repeat_max_abs'], name + '.repeat_max_abs', 0)
            repeats[arm][role] = repeat
            if repeat > limits['maximum_repeat_abs']:
                failures.append(f'{name}.maximum_repeat_abs exceeds frozen budget')
            ids = canonical_ids(record, len(dataset[key]), max_tokens, arm == 'target', name)
            if arm == 'target':
                require(record['qualification'] == 'capture-only-unqualified', f'{name} must remain capture-only-unqualified')
                runtime = record['runtime']
                require(runtime['qualification'] == 'unqualified' and runtime['index_compatibility'] == 'not-qualified',
                        f'{name} runtime/index must remain unqualified')
                require(runtime['batching'] == 'serial-unpadded', f'{name} runtime must be serial-unpadded')
                digest(record['code_sha256']['qwen_embedding_capture.rs'], name + '.capture code')
                require(record['code_sha256']['qwen_embedding_capture.rs'] == policy['capture_code_sha256'],
                        f'{name} compiled caller source SHA256 mismatch')
                for artifact in ('decode', 'prefill'):
                    full_hash = policy['target_artifacts'].get(artifact)
                    expected_prefix = None if full_hash is None else full_hash[:12]
                    require(runtime[artifact + '_artifact_hash_prefix'] == expected_prefix,
                            f'{name} {artifact} prefix disagrees with full artifact hash')
                require(ids == canonical_ids(ref[role], len(dataset[key]), max_tokens, False, 'reference.' + role),
                        f'{name} prepared token IDs differ from ordered reference')
    for role in ('document', 'query'):
        diagnostics[role] = compare(ref[role]['vectors'], target[role]['vectors'])
        for metric in ('minimum_cosine', 'maximum_absolute_error', 'maximum_relative_l2_error'):
            value = diagnostics[role][metric]
            passed = value >= limits[metric] if metric == 'minimum_cosine' else value <= limits[metric]
            if not passed:
                failures.append(f'{role}.{metric} violates frozen threshold')
    # The receipt is coordinator-supplied execution/admission evidence, not hardware attestation.
    return dict(scope='numerical-parity-only', **{'pass': not failures}, failures=failures,
                policy_sha256=sha256(policy_source), receipt_sha256=sha256(receipt_source),
                stage=policy['stage'], run_id=policy['run_id'], dataset_sha256=policy['dataset_sha256'],
                profile=profile, reference_sha256=policy['reference_sha256'],
                records_sha256=receipt['records_sha256'], input_sha256=policy['input_sha256'],
                reference_runtime=policy['reference_runtime'], target_runtime=policy['target_runtime'],
                capture_code_sha256=policy['capture_code_sha256'], capture_binary_sha256=policy['capture_binary_sha256'],
                target_artifacts=policy['target_artifacts'], thresholds=limits,
                comparison=diagnostics, repeat_max_abs=repeats,
                execution_evidence='coordinator-receipt-not-hardware-attestation')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--documents-ref', type=Path, required=True)
    parser.add_argument('--queries-ref', type=Path, required=True)
    parser.add_argument('--documents-target', type=Path)
    parser.add_argument('--queries-target', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualification-policy', type=Path)
    parser.add_argument('--target-receipt', type=Path)
    args = parser.parse_args()
    if bool(args.qualification_policy) != bool(args.target_receipt):
        parser.error('both --qualification-policy and --target-receipt are required')
    strict = args.qualification_policy is not None
    if strict and not (args.documents_target and args.queries_target):
        parser.error('qualification requires both target vector records')
    if args.output.exists():
        parser.error('output already exists')
    if bool(args.documents_target) != bool(args.queries_target):
        parser.error('both target vector records are required')
    source = args.dataset.read_bytes()
    dataset = load_json(source, strict)
    refpaths = {'document': args.documents_ref, 'query': args.queries_ref}
    refsources = {r: p.read_bytes() for r, p in refpaths.items()}
    ref = read_records(refpaths, dataset, refsources, strict)
    refvec = dict(documents=ref['document']['vectors'], queries=ref['query']['vectors'])
    report = dict(dataset_sha256=hashlib.sha256(source).hexdigest(), reference=evaluate(dataset, refvec),
                  profile=ref['document']['profile'], reference_runtime={k: v['runtime'] for k, v in ref.items()},
                  qualification='unqualified-descriptive-bounded-dataset-only', source_claim=dataset.get('scope', 'unspecified'))
    if args.documents_target:
        targetpaths = {'document': args.documents_target, 'query': args.queries_target}
        targetsources = {r: p.read_bytes() for r, p in targetpaths.items()}
        target = read_records(targetpaths, dataset, targetsources, strict)
        if target['document']['profile'] != report['profile']:
            raise ValueError('target and reference semantic profiles differ')
        targetvec = dict(documents=target['document']['vectors'], queries=target['query']['vectors'])
        report['target'] = evaluate(dataset, targetvec)
        report['target_runtime'] = {k: v['runtime'] for k, v in target.items()}
        report['comparison'] = {k: compare(refvec[k], targetvec[k]) for k in refvec}
    if strict:
        report['numerical_qualification'] = qualification(
            args.qualification_policy.read_bytes(), args.target_receipt.read_bytes(), source, dataset, ref, target,
            {'reference': refsources, 'target': targetsources})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    return 1 if strict and not report['numerical_qualification']['pass'] else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        raise SystemExit(f'evaluate_embeddings: error: {exc}')
