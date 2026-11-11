import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import evaluate_embeddings as ev


class EvaluationTests(unittest.TestCase):
    def fixture(self):
        data = {'documents': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}],
                'queries': [{'id': 'q', 'group': 'ru-en', 'relevant': ['a', 'b'], 'hard_negatives': ['c']} ]}
        vectors = {'documents': [[1., 0.], [0., 1.], [-1., 0.]], 'queries': [[1., 0.]]}
        return data, vectors

    def test_recall_is_fraction_of_all_relevant_and_mrr_counts_misses(self):
        data, vectors = self.fixture()
        report = ev.evaluate(data, vectors)
        self.assertEqual(report['overall']['recall@5'], 1.)
        self.assertEqual(report['overall']['mrr@10'], 1.)
        self.assertEqual(report['groups']['ru-en'], report['overall'])
        self.assertEqual(report['cases'][0]['ranking'], ['a', 'b', 'c'])
        data['documents'] = [{'id': str(i)} for i in range(12)]
        data['queries'][0]['relevant'] = ['11']
        data['queries'][0]['hard_negatives'] = ['0']
        vectors['documents'] = [[1., 0.]] * 12
        report = ev.evaluate(data, vectors)
        # Stable ties are document IDs, not an implementation-dependent topk order.
        self.assertEqual(report['cases'][0]['ranking'][:4], ['0', '1', '10', '11'])
        self.assertEqual(report['overall']['mrr@10'], .25)
        data['queries'][0]['relevant'] = ['9']
        self.assertEqual(ev.evaluate(data, vectors)['overall']['mrr@10'], 0.)
        self.assertEqual(ev.evaluate(data, vectors)['overall']['recall@10'], 0.)

    def test_group_scope_is_explicit_and_cannot_drop_relevant_documents(self):
        data, vectors = self.fixture()
        data['queries'][0]['hard_negatives'] = []
        data['candidate_groups'] = {'ru-en': ['a', 'b']}
        report = ev.evaluate(data, vectors)
        self.assertEqual(report['cases'][0]['ranking'], ['a', 'b'])
        self.assertEqual(report['cases'][0]['candidate_count'], 2)
        data['candidate_groups']['ru-en'] = ['a']
        with self.assertRaises(ValueError):
            ev.evaluate(data, vectors)

    def test_multisource_recall_and_hard_negative_errors(self):
        data, vectors = self.fixture()
        data['queries'][0]['relevant'] = ['b']
        data['queries'][0]['hard_negatives'] = ['a']
        report = ev.evaluate(data, vectors)
        self.assertEqual(report['cases'][0]['hard_negatives_before_first_relevant'], ['a'])
        self.assertEqual(report['overall']['hard_negative_error_rate'], 1.)

    def test_invalid_labels_universe_and_vectors_reject(self):
        data, vectors = self.fixture()
        bad_data = []
        for key, value in [('relevant', []), ('relevant', ['missing']), ('hard_negatives', ['a'])]:
            changed = copy.deepcopy(data)
            changed['queries'][0][key] = value
            bad_data.append(changed)
        changed = copy.deepcopy(data)
        changed['documents'][1]['id'] = 'a'
        bad_data.append(changed)
        for changed in bad_data:
            with self.subTest(data=changed), self.assertRaises(ValueError):
                ev.evaluate(changed, vectors)
        for replacement in [[], [[1., 0.]], [[float('nan'), 0.]] * 3, [[0., 0.]] * 3, [[2., 0.]] * 3]:
            with self.subTest(vectors=replacement), self.assertRaises(ValueError):
                ev.evaluate(data, dict(vectors, documents=replacement))

    def test_cli_binds_text_order_profiles_and_never_overwrites(self):
        data, vectors = self.fixture()
        for item in data['documents'] + data['queries']:
            item['text'] = item['id']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'dataset.json').write_text(json.dumps(data))
            for role, key in [('document', 'documents'), ('query', 'queries')]:
                record = dict(input_type=role, profile={'model': 'fixture'}, runtime={'device': 'fixture'},
                              texts_sha256=ev.text_identity([r['text'] for r in data[key]]), vectors=vectors[key])
                (root/f'{role}.json').write_text(json.dumps(record))
            command = [sys.executable, str(Path(ev.__file__)), '--dataset', str(root/'dataset.json'),
                       '--documents-ref', str(root/'document.json'), '--queries-ref', str(root/'query.json'),
                       '--output', str(root/'result.json')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = (root/'result.json').read_bytes()
            self.assertEqual(json.loads(before)['reference']['overall']['mrr@10'], 1.)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(before, (root/'result.json').read_bytes())
            record['texts_sha256'] = 'changed'
            (root/'query.json').write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                ev.read_records({'document': root/'document.json', 'query': root/'query.json'}, data)

    def test_synthetic_fixture_has_all_language_groups_and_explicit_scopes(self):
        path = Path(__file__).resolve().parents[1]/'tests/data/qwen_embedding_synthetic.json'
        data = json.loads(path.read_text())
        vectors = {key: [[1., 0.]] * len(data[key]) for key in ['documents', 'queries']}
        report = ev.evaluate(data, vectors)
        self.assertEqual(set(report['groups']), {'en-en', 'ru-ru', 'ru-en', 'en-ru'})
        self.assertEqual(report['counts'], {'documents': 22, 'queries': 14})
        self.assertTrue(any(len(q['relevant']) > 1 for q in data['queries']))

    def test_corresponding_vector_comparison_detects_wrong_pooling(self):
        ref = [[1., 0.], [0., 1.]]
        same = ev.compare(ref, ref)
        self.assertEqual(same['minimum_cosine'], 1.)
        self.assertEqual(same['maximum_absolute_error'], 0.)
        wrong = ev.compare(ref, [[0., 1.], [1., 0.]])
        self.assertEqual(wrong['minimum_cosine'], 0.)
        self.assertEqual(wrong['maximum_absolute_error'], 1.)
        for bad in [[], [[1., 0.]], [[float('inf'), 0.]] * 2]:
            with self.assertRaises(ValueError):
                ev.compare(ref, bad)


class QualificationGateTests(unittest.TestCase):
    """Fabricated two-dimensional records, not model or execution evidence."""

    def setUp(self):
        self.reset_fixture()

    def reset_fixture(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = {
            'documents': [{'id': 'a', 'text': 'alpha'}, {'id': 'b', 'text': 'beta'}],
            'queries': [{'id': 'q', 'text': 'question', 'group': 'fixture',
                         'relevant': ['a'], 'hard_negatives': ['b']}]}
        self.profile = dict(dimension=2, max_tokens=4, padding='left', pooling='last',
                            model='fabricated', revision='fixture', files={'tokenizer': 'fixture'})
        self.artifacts = {'decode': 'a' * 64, 'prefill': 'b' * 64, 'packed_weights': 'c' * 64}
        runtime = dict(qualification='unqualified', index_compatibility='not-qualified',
                       batching='serial-unpadded', decode_artifact_hash_prefix='a' * 12,
                       prefill_artifact_hash_prefix='b' * 12, dtype='fixture', packing='fixture')
        self.records = {}
        for arm in ('reference', 'target'):
            self.records[arm] = {}
            for role, key in [('document', 'documents'), ('query', 'queries')]:
                count = len(self.data[key])
                self.records[arm][role] = dict(
                    qualification='capture-only-unqualified' if arm == 'target' else 'CPU-reference-only',
                    input_type=role, texts_sha256=ev.text_identity([x['text'] for x in self.data[key]]),
                    input_sha256=('d' if role == 'document' else 'e') * 64,
                    profile=copy.deepcopy(self.profile), runtime=copy.deepcopy(runtime) if arm == 'target' else {'device': 'fixture', 'threads': 2},
                    code_sha256={'qwen_embedding_capture.rs': 'f' * 64},
                    vectors=[[1., 0.]] * count, input_ids=[[11 + i] for i in range(count)],
                    attention_mask=[[1] for _ in range(count)], repeat_max_abs=0.)
        # Reference left padding is removed only for cross-arm ID comparison.
        self.records['reference']['document']['input_ids'][0] = [0, 11]
        self.records['reference']['document']['attention_mask'][0] = [0, 1]
        self.policy = dict(format=1, stage='fabricated-numerical-parity', run_id='fixture-run',
                           window={'start_unix_s': 100, 'deadline_unix_s': 200},
                           dataset_sha256='', profile=copy.deepcopy(self.profile),
                           reference_sha256={}, input_sha256={r: v['input_sha256'] for r, v in self.records['reference'].items()},
                           reference_runtime={r: copy.deepcopy(v['runtime']) for r, v in self.records['reference'].items()},
                           target_runtime={r: copy.deepcopy(v['runtime']) for r, v in self.records['target'].items()},
                           capture_code_sha256='f' * 64, capture_binary_sha256='1' * 64,
                           target_artifacts=copy.deepcopy(self.artifacts),
                           thresholds=dict(minimum_cosine=.99, maximum_absolute_error=.01,
                                           maximum_relative_l2_error=.01, maximum_repeat_abs=.001))
        self.receipt = dict(format=1, stage=self.policy['stage'], run_id='fixture-run',
                            started_unix_s=110, finished_unix_s=120, capture_binary_sha256='1' * 64,
                            target_artifacts=copy.deepcopy(self.artifacts), records_sha256={})
        self.write_inputs()

    def put(self, name, value):
        path = self.root / (name + '.json')
        path.write_text(json.dumps(value))
        return ev.hashlib.sha256(path.read_bytes()).hexdigest()

    def write_inputs(self):
        self.policy['dataset_sha256'] = self.put('dataset', self.data)
        for arm in self.records:
            for role, record in self.records[arm].items():
                digest = self.put(f'{arm}-{role}', record)
                if arm == 'reference':
                    self.policy['reference_sha256'][role] = digest
                else:
                    self.receipt['records_sha256'][role] = digest
        self.put('policy', self.policy)
        self.put('receipt', self.receipt)

    def cli(self, flags=None):
        names = {'dataset': 'dataset', 'documents-ref': 'reference-document',
                 'queries-ref': 'reference-query', 'documents-target': 'target-document',
                 'queries-target': 'target-query', 'output': 'result'}
        if flags is None:
            names.update({'qualification-policy': 'policy', 'target-receipt': 'receipt'})
        else:
            names.update(flags)
        command = [sys.executable, ev.__file__]
        for flag, name in names.items():
            command.extend(['--' + flag, str(self.root / (name + '.json'))])
        return subprocess.run(command, capture_output=True, text=True)

    def test_exact_numerical_boundaries_pass_and_one_ulp_tighter_budgets_fail(self):
        metrics = ('minimum_cosine', 'maximum_absolute_error', 'maximum_relative_l2_error')
        for tightened in (None, *metrics, 'maximum_repeat_abs'):
            with self.subTest(tightened=tightened):
                self.reset_fixture()
                self.records['target']['document']['vectors'][0] = [.6, .8]
                observed = ev.compare(self.records['reference']['document']['vectors'],
                                      self.records['target']['document']['vectors'])
                self.policy['thresholds'].update({key: observed[key] for key in metrics})
                for records in self.records.values():
                    for record in records.values():
                        record['repeat_max_abs'] = self.policy['thresholds']['maximum_repeat_abs']
                if tightened:
                    value = self.policy['thresholds'][tightened]
                    direction = float('inf') if tightened == 'minimum_cosine' else -float('inf')
                    self.policy['thresholds'][tightened] = float(ev.np.nextafter(value, direction))
                self.write_inputs()
                result = self.cli()
                self.assertEqual(result.returncode, 0 if tightened is None else 1, result.stderr)
                report = json.loads((self.root / 'result.json').read_text())
                self.assertEqual(report['numerical_qualification']['pass'], tightened is None)

    def test_null_deadline_keeps_finite_ordered_receipts_and_numerical_gates(self):
        self.policy['window']['deadline_unix_s'] = None
        self.receipt.update(started_unix_s=1000, finished_unix_s=1010)
        self.write_inputs()
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads((self.root / 'result.json').read_text())['numerical_qualification']['pass'])
        for changes, message in [({'started_unix_s': 99}, 'receipt timestamps'),
                                 ({'finished_unix_s': 109}, 'receipt timestamps'),
                                 ({'started_unix_s': True}, 'receipt start must be'),
                                 ({'finished_unix_s': float('nan')}, 'nonfinite JSON')]:
            with self.subTest(changes=changes):
                self.reset_fixture()
                self.policy['window']['deadline_unix_s'] = None
                self.receipt.update(changes)
                self.write_inputs()
                result = self.cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
        self.reset_fixture()
        self.policy['window']['deadline_unix_s'] = None
        self.records['target']['document']['vectors'][0] = [-1., 0.]
        self.write_inputs()
        result = self.cli()
        self.assertEqual(result.returncode, 1)
        gate = json.loads((self.root / 'result.json').read_text())['numerical_qualification']
        self.assertFalse(gate['pass'])
        self.assertIn('document.minimum_cosine violates frozen threshold', gate['failures'])

    def test_cli_valid_bound_report_and_no_overwrite(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        path = self.root / 'result.json'
        before = path.read_bytes()
        gate = json.loads(before)['numerical_qualification']
        self.assertTrue(gate['pass'])
        self.assertEqual(gate['scope'], 'numerical-parity-only')
        self.assertEqual(gate['stage'], self.policy['stage'])
        for key, name in [('policy_sha256', 'policy'), ('receipt_sha256', 'receipt')]:
            self.assertEqual(gate[key], ev.hashlib.sha256((self.root / (name + '.json')).read_bytes()).hexdigest())
        self.assertEqual(gate['target_artifacts'], self.artifacts)
        self.assertNotEqual(self.cli().returncode, 0)
        self.assertEqual(path.read_bytes(), before)

    def test_cli_without_policy_is_unqualified_and_partial_flags_reject(self):
        result = self.cli({})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('unqualified', json.loads((self.root / 'result.json').read_text())['qualification'])
        for flags in ({'qualification-policy': 'policy'}, {'target-receipt': 'receipt'},
                      {'qualification-polic': 'policy'}):
            with self.subTest(flags=flags):
                result = self.cli(flags)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('error', result.stderr)

    def test_numerical_failure_retains_bound_report_for_each_role_and_metric(self):
        for role in ('document', 'query'):
            for metric in self.policy['thresholds']:
                with self.subTest(role=role, metric=metric):
                    record = self.records['target'][role]
                    if metric == 'maximum_repeat_abs':
                        record['repeat_max_abs'] = .1
                    else:
                        record['vectors'][0] = [0., 1.]
                        # Isolate each numerical threshold from the other two.
                        self.policy['thresholds'].update(minimum_cosine=-1., maximum_absolute_error=2., maximum_relative_l2_error=2.)
                        self.policy['thresholds'][metric] = .99 if metric == 'minimum_cosine' else .01
                    self.write_inputs()
                    result = self.cli()
                    self.assertNotEqual(result.returncode, 0)
                    path = self.root / 'result.json'
                    self.assertTrue(path.exists(), result.stderr)
                    gate = json.loads(path.read_text())['numerical_qualification']
                    self.assertFalse(gate['pass'])
                    self.assertTrue(any(role in f and metric in f for f in gate['failures']))
                    # Each iteration gets fresh independent fabricated files.
                    self.reset_fixture()

    def test_integrity_mutations_reject_without_report(self):
        def set_record(arm, role, key, value):
            self.records[arm][role][key] = value
        mutations = []
        for arm in ('reference', 'target'):
            for role in ('document', 'query'):
                for key, value in [
                    ('input_type', 'wrong'), ('texts_sha256', '0' * 64), ('input_sha256', '0' * 64),
                    ('vectors', []), ('vectors', [[1., 0.]] * 3), ('vectors', [[1., 0., 0.]]),
                    ('vectors', [[float('nan'), 0.]]), ('vectors', [[0., 0.]]),
                    ('vectors', [[2., 0.]]), ('vectors', [[True, False]]), ('vectors', [[1., False]]),
                    ('input_ids', []), ('attention_mask', []),
                    ('repeat_max_abs', -1), ('repeat_max_abs', True), ('repeat_max_abs', float('inf'))]:
                    mutations.append((f'{arm}/{role}/{key}/{value}', lambda a=arm, r=role, k=key, v=value: set_record(a, r, k, v)))
                for field in ('pooling', 'revision', 'files', 'model'):
                    mutations.append((f'{arm}/{role}/profile/{field}', lambda a=arm, r=role, f=field: self.records[a][r]['profile'].update({f: 'changed'})))
                for key in ('input_sha256', 'code_sha256', 'repeat_max_abs'):
                    if key != 'code_sha256' or arm == 'target':
                        mutations.append((f'missing/{arm}/{role}/{key}', lambda a=arm, r=role, k=key: self.records[a][r].pop(k)))
        for ids, mask in [([True], [1]), ([-1], [1]), ([2**32], [1]),
                          ([11], [True]), ([11], [2]), ([11], [0]),
                          ([11, 12], [1]), ([11, 12, 13], [1, 0, 1]),
                          ([11, 12], [1, 0]), ([11] * 5, [1] * 5),
                          ([99], [1])]:
            mutations.append((f'tokens/{ids}/{mask}', lambda i=ids, m=mask: self.records['target']['query'].update(input_ids=[i], attention_mask=[m])))
        mutations += [
            ('ref-right-padding', lambda: self.records['reference']['query'].update(input_ids=[[11, 0]], attention_mask=[[1, 0]])),
            ('prepared-order', lambda: self.records['target']['document'].update(input_ids=[[12], [11]])),
            ('target-code', lambda: self.records['target']['query']['code_sha256'].update({'qwen_embedding_capture.rs': '0' * 64})),
            ('runtime-packing', lambda: self.records['target']['query']['runtime'].update(packing='changed')),
            ('runtime-bool', lambda: self.records['reference']['query']['runtime'].update(threads=True)),
            ('runtime-promote', lambda: self.records['target']['query']['runtime'].update(qualification='qualified')),
            ('top-promote', lambda: self.records['target']['query'].update(qualification='qualified')),
        ]
        for name, mutate in mutations:
            with self.subTest(mutation=name):
                self.reset_fixture()
                mutate()
                self.write_inputs()  # Rebind file hashes: exercise content checks, not just hashes.
                result = self.cli()
                self.assertNotEqual(result.returncode, 0, name)
                self.assertFalse((self.root / 'result.json').exists(), result.stderr)
                self.assertIn('error', result.stderr.lower())

    def test_policy_receipt_schema_and_identity_mutations_reject(self):
        mutations = []
        for field, value in [('format', True), ('stage', ''), ('run_id', None),
                             ('capture_binary_sha256', 'bad'), ('capture_code_sha256', 'bad'),
                             ('window', []), ('profile', []), ('thresholds', []),
                             ('reference_runtime', {}), ('target_runtime', {}),
                             ('dataset_sha256', '0' * 64), ('reference_sha256', {}),
                             ('input_sha256', {}), ('target_artifacts', {'decode': 'bad'})]:
            mutations.append(('policy/' + field, lambda f=field, v=value: self.policy.update({f: v})))
        for field, value in [('format', True), ('stage', 'other'), ('run_id', 'old'),
                             ('capture_binary_sha256', '0' * 64), ('target_artifacts', {'decode': 'a' * 64}),
                             ('records_sha256', {'document': '0' * 64, 'query': '0' * 64}),
                             ('started_unix_s', 99), ('finished_unix_s', 200),
                             ('started_unix_s', 121), ('finished_unix_s', float('nan'))]:
            mutations.append(('receipt/' + field, lambda f=field, v=value: self.receipt.update({f: v})))
        for key in self.policy['thresholds']:
            for value in (True, '0', float('nan'), float('inf'), -2, 2):
                if value == 2 and key != 'minimum_cosine':
                    continue
                mutations.append((key + str(value), lambda k=key, v=value: self.policy['thresholds'].update({k: v})))
        mutations += [
            ('dimension-bool', lambda: self.policy['profile'].update(dimension=True)),
            ('max-tokens-bool', lambda: self.policy['profile'].update(max_tokens=True)),
            ('window-bool', lambda: self.policy['window'].update(start_unix_s=True)),
            ('window-reversed', lambda: self.policy['window'].update(start_unix_s=201)),
            ('no-prefill', lambda: self.policy['target_artifacts'].pop('prefill')),
            ('prefix', lambda: self.policy['target_runtime']['query'].update(decode_artifact_hash_prefix='0' * 12)),
        ]
        for name, mutate in mutations:
            with self.subTest(mutation=name):
                self.reset_fixture()
                mutate()
                self.put('policy', self.policy)
                self.put('receipt', self.receipt)
                result = self.cli()
                self.assertNotEqual(result.returncode, 0, name)
                self.assertFalse((self.root / 'result.json').exists(), result.stderr)

    def test_dataset_and_record_full_file_admission(self):
        for name in ('dataset', 'reference-document', 'reference-query', 'target-document', 'target-query'):
            with self.subTest(file=name):
                self.reset_fixture()
                path = self.root / (name + '.json')
                path.write_bytes(path.read_bytes() + b'\n')
                result = self.cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / 'result.json').exists())
        for key in ('documents', 'queries'):
            for mutation in ('missing', 'extra', 'duplicate'):
                with self.subTest(key=key, mutation=mutation):
                    self.reset_fixture()
                    if mutation == 'missing':
                        self.data[key].pop()
                    else:
                        row = copy.deepcopy(self.data[key][0])
                        if mutation == 'extra':
                            row['id'] = 'new'
                        self.data[key].append(row)
                    self.write_inputs()
                    self.assertNotEqual(self.cli().returncode, 0)
                    self.assertFalse((self.root / 'result.json').exists())


    def test_optional_prefill_absent_is_valid_only_with_null_runtime_prefix(self):
        self.policy['target_artifacts'].pop('prefill')
        self.receipt['target_artifacts'].pop('prefill')
        for role in ('document', 'query'):
            self.policy['target_runtime'][role]['prefill_artifact_hash_prefix'] = None
            self.records['target'][role]['runtime']['prefill_artifact_hash_prefix'] = None
        self.write_inputs()
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads((self.root / 'result.json').read_text())['numerical_qualification']['pass'])

    def test_reference_repeat_budget_is_also_numerical_failure(self):
        for role in ('document', 'query'):
            with self.subTest(role=role):
                self.reset_fixture()
                self.records['reference'][role]['repeat_max_abs'] = .1
                self.write_inputs()
                result = self.cli()
                self.assertEqual(result.returncode, 1, result.stderr)
                gate = json.loads((self.root / 'result.json').read_text())['numerical_qualification']
                self.assertFalse(gate['pass'])
                self.assertIn(f'reference.{role}.maximum_repeat_abs exceeds frozen budget', gate['failures'])
                self.assertEqual(gate['policy_sha256'], ev.hashlib.sha256((self.root / 'policy.json').read_bytes()).hexdigest())

    def test_runtime_prefix_and_promotion_checks_cannot_be_bypassed_by_frozen_runtime(self):
        for field, value in [('decode_artifact_hash_prefix', 'a'),
                             ('prefill_artifact_hash_prefix', None), ('qualification', 'qualified'),
                             ('index_compatibility', 'qualified'), ('batching', 'padded')]:
            with self.subTest(field=field):
                self.reset_fixture()
                for obj in (self.policy['target_runtime']['query'], self.records['target']['query']['runtime']):
                    obj[field] = value
                self.write_inputs()
                result = self.cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / 'result.json').exists())

    def test_all_required_policy_and_receipt_fields_are_required(self):
        for name in ('policy', 'receipt'):
            fields = list(getattr(self, name))
            for field in fields:
                with self.subTest(name=name, field=field):
                    self.reset_fixture()
                    getattr(self, name).pop(field)
                    self.put(name, getattr(self, name))
                    result = self.cli()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(field, result.stderr)
                    self.assertFalse((self.root / 'result.json').exists())

    def test_strict_json_duplicate_keys_and_overflowing_numbers_reject(self):
        for name in ('policy', 'receipt', 'target-query', 'reference-query', 'dataset'):
            for suffix in ('"duplicate":1,"duplicate":2', '"overflow":1e999'):
                with self.subTest(name=name, suffix=suffix):
                    self.reset_fixture()
                    path = self.root / (name + '.json')
                    source = path.read_text()
                    path.write_text(source[:-1] + ',' + suffix + '}')
                    result = self.cli()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse((self.root / 'result.json').exists())

    def test_gate_requires_both_target_roles_before_reading_inputs(self):
        command = [sys.executable, ev.__file__, '--dataset', 'missing', '--documents-ref', 'missing',
                   '--queries-ref', 'missing', '--qualification-policy', 'missing',
                   '--target-receipt', 'missing', '--output', str(self.root / 'result.json')]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('both target', result.stderr)
        self.assertNotIn('No such file', result.stderr)


    def test_exact_profiles_and_runtimes_do_not_equate_bool_and_integer(self):
        for field in ('profile', 'runtime'):
            with self.subTest(field=field):
                self.reset_fixture()
                if field == 'profile':
                    self.policy['profile']['test_fact'] = 1
                    for records in self.records.values():
                        for record in records.values():
                            record['profile']['test_fact'] = 1
                else:
                    self.policy['reference_runtime']['query']['test_fact'] = 1
                    self.records['reference']['query']['runtime']['test_fact'] = 1
                self.records['reference']['query'][field]['test_fact'] = True
                self.write_inputs()
                result = self.cli()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('mismatch', result.stderr)
                self.assertFalse((self.root / 'result.json').exists())

    def test_correct_row_counts_do_not_hide_wrong_full_dimension_or_extra_token_rows(self):
        for arm in ('reference', 'target'):
            for role, key in (('document', 'documents'), ('query', 'queries')):
                for field in ('vectors', 'input_ids', 'attention_mask'):
                    with self.subTest(arm=arm, role=role, field=field):
                        self.reset_fixture()
                        record = self.records[arm][role]
                        if field == 'vectors':
                            record[field] = [[1., 0., 0.]] * len(self.data[key])
                        else:
                            record[field].append([1])
                        self.write_inputs()
                        result = self.cli()
                        self.assertNotEqual(result.returncode, 0)
                        self.assertFalse((self.root / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
