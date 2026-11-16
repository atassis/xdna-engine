#!/usr/bin/env python3
"""CPU-only tests for the isolated Gemma4 source witness."""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verified_gemma4_data as g4
import verified_gemma_data as g3

def npy(shape, seed=1):
    text = "{'descr': '|u1', 'fortran_order': False, 'shape': %s, }" % (tuple(shape),)
    raw = (text + ' ' * ((-10 - len(text) - 1) % 64) + '\n').encode()
    count = 1
    for axis in shape: count *= axis
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(raw)) + raw + bytes((seed + i) % 251 for i in range(count))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Gemma4WitnessTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix='.g4-witness-tests-'))
        self.addCleanup(lambda: shutil.rmtree(self.work, ignore_errors=True))
        self.owner = self.work / 'accepted'
        self.root = self.owner / 'gemma4-12b'
        self.make_source()

    def receipt(self, directory, extra=()):
        files = {}
        for item in sorted(directory.rglob('*')):
            if item.is_file() and item.name != '.recipe-manifest.json':
                files[item.relative_to(directory).as_posix()] = sha(item)
        for name in extra: files[name] = '0' * 64
        (directory / '.recipe-manifest.json').write_text(json.dumps({'command': 'historical', 'created': 'old', 'inputs': {'source': '/retired'}, 'stage': 'old', 'files': files}))

    def make_source(self):
        if self.root.exists(): shutil.rmtree(self.root)
        store = self.root / 'store'; rf = self.root / 'rf_stack'; hf = self.root / 'hf_config'; tok = self.root / 'tokenizer'
        for directory in (store, rf, hf, tok): directory.mkdir(parents=True)
        manifest = {'model': 'gemma4-12b', 'created': 'old', 'source_dir': '/retired', 'num_layers': 48,
                    'hidden_size': 3840, 'intermediate_size': 15360,
                    'full_attention_layers': [5, 11, 17, 23, 29, 35, 41, 47],
                    'layers': {}, 'embedding': {}, 'final_norm': {}, 'towers': {}}
        blobs = store / 'blobs'
        blobs.mkdir()
        for index in range(11):
            blob = ('%064x' % index)
            (blobs / (blob + '.bin')).write_bytes(b'\0\0\0\0')
            manifest['towers']['tower%d' % index] = {'blob': blob, 'offset': 0, 'length': 4,
                                                      'layout': 'raw_native', 'shape': [1], 'dtype': 'float32'}
        shared_blob = '%064x' % 0
        manifest['embedding'] = {'blob': shared_blob}
        manifest['final_norm'] = {'blob': shared_blob}
        full = {5, 11, 17, 23, 29, 35, 41, 47}
        for index in range(48):
            matrix_names = ['mlp_stream', 'attn_in_stream', 'attn_out_stream']
            if index in full:
                matrix_names = ['mlp_stream', 'q_proj', 'k_proj', 'o_proj', 'input_layernorm',
                                'post_attention_layernorm', 'q_norm', 'k_norm']
            manifest['layers'][str(index)] = {
                'attn_type': 'full' if index in full else 'sliding',
                'matrices': {name: {'blob': shared_blob} for name in matrix_names}
            }
        (store / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
        for i in range(48): (rf / ('w%d.npy' % i)).write_bytes(npy((2,), i))
        (rf / 'w_head.npy').write_bytes(npy((1, 1, 1, 1, 2), 50))
        config = {'model_type': 'gemma4_unified_text', 'hidden_size': 3840, 'intermediate_size': 15360,
                  'num_hidden_layers': 48, 'num_attention_heads': 16, 'num_key_value_heads': 8,
                  'head_dim': 256, 'vocab_size': 262144}
        raw = json.dumps({'text_config': config}, separators=(',', ':'))
        (hf / 'config.json').write_text(raw); (tok / 'config.json').write_text(raw)
        (tok / 'tokenizer_config.json').write_text('{"chat_template":"template"}')
        (tok / 'tokenizer.json').write_text('{"large":"streamed"}')
        (tok / 'chat_template.jinja').write_text('template')
        self.receipt(store); self.receipt(rf)

    def witness(self):
        return g4.build_witness(self.root, accepted_data_root=self.owner, provenance='retained review', legacy={'run': 1})

    def test_positive_build_load_and_verify(self):
        witness = self.witness()
        self.assertEqual(set(witness['roles']), set(g4.ROLE_PATHS))
        self.assertEqual(witness['payload_digest'], g3.canonical_digest(witness['roles']))
        path = self.work / 'witness.json'; path.write_text(json.dumps(witness))
        self.assertEqual(g4.load_witness(path)['model'], 'gemma4-12b')
        self.assertIsNone(g4.verify_source(self.root, witness, accepted_data_root=self.owner))

    def _rebind_embedded_receipt(self, witness, receipt):
        raw = json.dumps(receipt, separators=(',', ':'))
        witness['derivations']['portable_store']['source_receipt_json'] = raw
        member = witness['roles']['retained-store']['members']['.recipe-manifest.json']
        member['sha256'] = hashlib.sha256(raw.encode()).hexdigest()
        member['bytes'] = len(raw.encode())
        witness['payload_digest'] = g3.canonical_digest(witness['roles'])

    def test_load_rejects_embedded_receipt_map_mismatch(self):
        witness = self.witness()
        receipt = json.loads(witness['derivations']['portable_store']['source_receipt_json'])
        receipt['files'].pop(next(iter(receipt['files'])))
        self._rebind_embedded_receipt(witness, receipt)
        with self.assertRaisesRegex(g4.VerifiedDataError, 'embedded store receipt'):
            g4._validate_witness(witness)

    def test_load_rejects_negative_rf_shape_and_embedded_byte_count(self):
        with self.subTest('negative shape'):
            witness = self.witness()
            witness['roles']['retained-rf-stack']['members']['w0.npy']['header']['shape'] = [-1]
            witness['payload_digest'] = g3.canonical_digest(witness['roles'])
            with self.assertRaisesRegex(g4.VerifiedDataError, 'invalid numpy header'):
                g4._validate_witness(witness)
        with self.subTest('embedded byte count'):
            witness = self.witness()
            witness['roles']['retained-store']['members']['manifest.json']['bytes'] = 0
            witness['payload_digest'] = g3.canonical_digest(witness['roles'])
            with self.assertRaisesRegex(g4.VerifiedDataError, 'embedded store metadata'):
                g4._validate_witness(witness)

    def test_reserved_tokenizer_sidecar_and_missing_store_essentials_refuse(self):
        witness = self.witness()
        (self.root / 'tokenizer' / g4.SIDECAR).write_text('{}')
        with self.assertRaisesRegex(g4.VerifiedDataError, 'reserved'):
            g4.verify_source(self.root, witness, accepted_data_root=self.owner)
        self.make_source()
        manifest_path = self.root / 'store' / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['embedding'] = {}
        manifest['final_norm'] = {}
        manifest['layers'] = {}
        manifest_path.write_text(json.dumps(manifest))
        self.receipt(self.root / 'store')
        with self.assertRaisesRegex(g4.VerifiedDataError, 'embedding'):
            self.witness()

    def test_missing_extra_changed_and_old_raw_are_handled(self):
        (self.root / 'store' / 'blobs' / ('%064x.bin' % 0)).unlink()
        with self.assertRaises(g4.VerifiedDataError): self.witness()
        self.make_source()  # rebuild receipts after restoration
        (self.root / 'rf_stack' / 'extra').write_bytes(b'x')
        with self.assertRaises(g4.VerifiedDataError): self.witness()
        (self.root / 'old_raw').mkdir(); (self.root / 'old_raw' / 'ignored').write_bytes(b'old')
        self.receipt(self.root / 'rf_stack')
        witness = self.witness(); (self.root / 'store' / 'blobs' / ('%064x.bin' % 0)).write_bytes(b'changed')
        with self.assertRaises(g4.VerifiedDataError): g4.verify_source(self.root, witness, accepted_data_root=self.owner)

    def test_path_receipt_and_witness_forgeries_refuse(self):
        with self.assertRaises(g4.VerifiedDataError): g4.build_witness(self.root, accepted_data_root=self.owner / '..', provenance='x', legacy={})
        (self.root / 'store' / '.recipe-manifest.json').write_text('{"files":{"../escape":{"sha256":"0","bytes":0}}}')
        with self.assertRaises(g4.VerifiedDataError): self.witness()
        self.make_source(); (self.root / 'rf_stack' / 'link').symlink_to(self.root / 'rf_stack' / 'w0.npy')
        with self.assertRaises(g4.VerifiedDataError): self.witness()
        self.make_source(); witness = self.witness(); witness['model'] = 'forged'
        with self.assertRaises(g4.VerifiedDataError): g4.verify_source(self.root, witness, accepted_data_root=self.owner)
        witness = self.witness(); witness['roles'].pop('build-config'); witness['payload_digest'] = g3.canonical_digest(witness['roles'])
        path = self.work / 'bad.json'; path.write_text(json.dumps(witness))
        with self.assertRaises(g4.VerifiedDataError): g4.load_witness(path)
        witness = self.witness(); witness['roles']['retained-store']['members']['manifest.json']['bytes'] += 1
        path.write_text(json.dumps(witness))
        with self.assertRaises(g4.VerifiedDataError): g4.load_witness(path)
        self.make_source(); receipt = self.root / 'store' / '.recipe-manifest.json'
        forged = json.loads(receipt.read_text()); forged['files']['manifest.json'] = '0' * 64
        receipt.write_text(json.dumps(forged))
        with self.assertRaises(g4.VerifiedDataError): self.witness()

    def test_direct_role_aliases_are_refused_but_unselected_paths_are_ignored(self):
        for role in g4.ROLE_PATHS.values():
            with self.subTest(role=role):
                self.make_source()
                original = self.root / (role + '-real')
                (self.root / role).rename(original)
                (self.root / role).symlink_to(original, target_is_directory=True)
                with self.assertRaisesRegex(g4.VerifiedDataError, 'symlink'):
                    self.witness()
        self.make_source()
        (self.root / 'README').write_text('not selected')
        nested = self.root / 'tokenizer' / 'unconsumed'
        nested.mkdir()
        (nested / 'README').write_text('not selected')
        self.assertIsNotNone(self.witness())

    def test_tokenizer_config_template_is_required_exact_and_nonempty(self):
        token_config = self.root / 'tokenizer' / 'tokenizer_config.json'
        for value in ('{}', '{"chat_template":"different"}', '{"chat_template":""}'):
            with self.subTest(value=value):
                token_config.write_text(value)
                with self.assertRaisesRegex(g4.VerifiedDataError, 'chat_template'):
                    self.witness()
        token_config.write_text('{"chat_template":"template"}')
        self.assertIsNotNone(self.witness())

    def test_load_rejects_embedded_hash_and_verify_binds_origin(self):
        witness = self.witness()
        witness['derivations']['portable_store']['source_manifest_json'] = '{}'
        path = self.work / 'forged.json'
        path.write_text(json.dumps(witness))
        with self.assertRaisesRegex(g4.VerifiedDataError, 'embedded store metadata'):
            g4.load_witness(path)
        witness = self.witness()
        witness['origin']['source_root'] = '/historical/gemma4-12b'
        self.assertIs(g4._validate_witness(witness), witness)
        with self.assertRaisesRegex(g4.VerifiedDataError, 'origin source_root'):
            g4.verify_source(self.root, witness, accepted_data_root=self.owner)
        witness['origin']['source_root'] = 'relative/gemma4-12b'
        with self.assertRaisesRegex(g4.VerifiedDataError, 'lexical absolute'):
            g4._validate_witness(witness)

    def test_store_blob_reference_must_be_receipt_owned(self):
        manifest_path = self.root / 'store' / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['towers']['tower0']['blob'] = 'f' * 64
        manifest_path.write_text(json.dumps(manifest))
        self.receipt(self.root / 'store')
        with self.assertRaisesRegex(g4.VerifiedDataError, 'receipt-owned'):
            self.witness()

    def test_geometry_config_tokenizer_and_embedded_metadata_refuse(self):
        config = json.loads((self.root / 'hf_config' / 'config.json').read_text()); config['text_config']['hidden_size'] = 1
        (self.root / 'hf_config' / 'config.json').write_text(json.dumps(config))
        with self.assertRaises(g4.VerifiedDataError): self.witness()
        self.make_source(); (self.root / 'tokenizer' / 'config.json').write_text('{}')
        with self.assertRaises(g4.VerifiedDataError): self.witness()
        self.make_source(); witness = self.witness(); witness['derivations']['portable_store']['source_manifest_json'] = '{}'
        with self.assertRaises(g4.VerifiedDataError): g4.verify_source(self.root, witness, accepted_data_root=self.owner)


class Gemma4ResidentDataTests(Gemma4WitnessTests):
    """Tiny resident layouts exercise the source-only installed-data verifier."""
    def setUp(self):
        super().setUp()
        resident_dir = Path(__file__).resolve().parents[1] / 'designs' / 'resident_forward'
        sys.path.insert(0, str(resident_dir))
        import package_artifact
        self.package_artifact = package_artifact
        self.resident = self.work / 'resident'
        self.rebuild_resident()

    def rebuild_resident(self):
        if self.resident.exists():
            shutil.rmtree(self.resident)
        self.resident.mkdir()
        shutil.copytree(self.root / 'rf_stack', self.resident / 'weights')
        shutil.copy2(self.root / 'rf_stack' / 'w_head.npy', self.resident / 'w_head.npy')
        self.package_artifact.derive_store(self.root / 'store', self.resident / 'store')
        self.witness_document = self.witness()
        self.sidecar_raw = json.dumps(self.witness_document, separators=(',', ':')).encode('utf-8')
        (self.resident / g4.SIDECAR).write_bytes(self.sidecar_raw)
        self.witness_sha = hashlib.sha256(self.sidecar_raw).hexdigest()

    def verify(self):
        return g4.verify_resident_data(self.resident, expected_witness_sha256=self.witness_sha)

    def test_resident_positive_is_independent_of_deleted_source(self):
        shutil.rmtree(self.root)
        self.assertEqual(self.verify()['model'], 'gemma4-12b')

    def test_expected_sidecar_hash_and_model_are_bound(self):
        with self.assertRaisesRegex(g4.VerifiedDataError, 'SHA256 differs'):
            g4.verify_resident_data(self.resident, expected_witness_sha256='0' * 64)
        document = json.loads(self.sidecar_raw)
        document['model'] = 'forged'
        (self.resident / g4.SIDECAR).write_text(json.dumps(document))
        with self.assertRaises(g4.VerifiedDataError):
            self.verify()

    def test_weight_and_store_inventory_refuses_change_missing_extra_and_symlink(self):
        cases = (
            ('changed weight', self.resident / 'weights' / 'w0.npy', lambda p: p.write_bytes(b'changed')),
            ('missing weight', self.resident / 'weights' / 'w1.npy', lambda p: p.unlink()),
            ('extra store', self.resident / 'store' / 'extra', lambda p: p.write_bytes(b'x')),
            ('symlink store', self.resident / 'store' / 'link', lambda p: p.symlink_to(self.resident / 'store' / 'manifest.json')),
        )
        for label, path, mutate in cases:
            with self.subTest(label=label):
                self.rebuild_resident()
                mutate(path)
                with self.assertRaises(g4.VerifiedDataError):
                    self.verify()

    def test_derived_metadata_and_head_alias_are_bound(self):
        cases = (
            ('source_dir left in', lambda: self._rewrite_manifest(lambda value: value.update({'source_dir': '/forged'}))),
            ('derivation hash', lambda: self._rewrite_receipt(lambda value: value['derivation'].update({'source_manifest_sha256': '0' * 64}))),
            ('receipt map', lambda: self._rewrite_receipt(lambda value: value['files'].update({'manifest.json': '0' * 64}))),
            ('head alias mismatch', lambda: (self.resident / 'w_head.npy').write_bytes(npy((1, 1, 1, 1, 2), 99))),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                self.rebuild_resident()
                mutate()
                with self.assertRaises(g4.VerifiedDataError):
                    self.verify()

    def _rewrite_manifest(self, change):
        path = self.resident / 'store' / 'manifest.json'
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def _rewrite_receipt(self, change):
        path = self.resident / 'store' / '.recipe-manifest.json'
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def test_parent_counterexamples_refuse(self):
        cases = (
            ('boolean derivation version', self._forge_boolean_derivation_version),
            ('numeric manifest type', self._forge_float_num_layers),
            ('coherent RF receipt lie', self._forge_rf_receipt_lie),
        )
        for label, forge in cases:
            with self.subTest(label=label):
                self.rebuild_resident()
                forge()
                with self.assertRaises(g4.VerifiedDataError):
                    self.verify()

    def test_sidecar_verification_uses_the_hashed_snapshot(self):
        original_read = g4._read
        replacement = json.loads(self.sidecar_raw)
        replacement['origin']['provenance'] = 'unapproved second document'
        replacement_raw = json.dumps(replacement, separators=(',', ':')).encode('utf-8')
        changed = False

        def read_then_replace(path, limit, label):
            nonlocal changed
            raw = original_read(path, limit, label)
            if Path(path) == self.resident / g4.SIDECAR and not changed:
                changed = True
                (self.resident / g4.SIDECAR).write_bytes(replacement_raw)
            return raw

        with mock.patch.object(g4, '_read', read_then_replace):
            document = self.verify()
        self.assertNotEqual(document['origin']['provenance'], 'unapproved second document')

    def _forge_boolean_derivation_version(self):
        self._rewrite_receipt(lambda value: value['derivation'].update({'version': True}))

    def _forge_float_num_layers(self):
        self._rewrite_manifest(lambda value: value.update({'num_layers': 48.0}))
        self._rewrite_receipt(lambda value: value['files'].update({
            'manifest.json': sha(self.resident / 'store' / 'manifest.json')}))

    def _forge_rf_receipt_lie(self):
        receipt = self.resident / 'weights' / '.recipe-manifest.json'
        value = json.loads(receipt.read_text())
        value['files']['w0.npy'] = '0' * 64
        receipt.write_text(json.dumps(value, separators=(',', ':')))
        document = json.loads(self.sidecar_raw)
        member = document['roles']['retained-rf-stack']['members']['.recipe-manifest.json']
        member['sha256'] = sha(receipt)
        member['bytes'] = receipt.stat().st_size
        document['payload_digest'] = g3.canonical_digest(document['roles'])
        self.sidecar_raw = json.dumps(document, separators=(',', ':')).encode('utf-8')
        (self.resident / g4.SIDECAR).write_bytes(self.sidecar_raw)
        self.witness_sha = hashlib.sha256(self.sidecar_raw).hexdigest()

    def test_already_portable_store_is_copied_without_invented_derivation(self):
        self.make_source()
        manifest = self.root / 'store' / 'manifest.json'
        value = json.loads(manifest.read_text())
        del value['source_dir']
        manifest.write_text(json.dumps(value, separators=(',', ':')))
        self.receipt(self.root / 'store')
        self.rebuild_resident()
        self.assertEqual((self.resident / 'store' / 'manifest.json').read_bytes(),
                         self.witness_document['derivations']['portable_store']['source_manifest_json'].encode())
        self.assertEqual((self.resident / 'store' / '.recipe-manifest.json').read_bytes(),
                         self.witness_document['derivations']['portable_store']['source_receipt_json'].encode())
        self.assertEqual(self.verify()['model'], 'gemma4-12b')


class Gemma4PublishedDataTests(unittest.TestCase):
    """Published sidecars plus manually framed and real-serializer tiny safetensors fixtures."""
    def setUp(self):
        self.fixture = Gemma4ResidentDataTests('run')
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.resident = self.fixture.resident
        self.tokenizer = self.fixture.work / 'published-tokenizer'
        self.checkpoint = self.fixture.work / 'published-checkpoint'
        shutil.copytree(self.fixture.root / 'tokenizer', self.tokenizer)
        shutil.copytree(self.fixture.root / 'tokenizer', self.checkpoint)
        self._write_towers()

    def _write_towers(self):
        # This is the safetensors framing used by package_gemma4's save_file output:
        # a LE header length, JSON header (space padding is legal), then contiguous F32 payloads.
        manifest = json.loads((self.resident / 'store' / 'manifest.json').read_text())
        entries, payload, offset = {}, bytearray(), 0
        for name, entry in sorted(manifest['towers'].items()):
            blob = self.resident / 'store' / 'blobs' / (entry['blob'] + '.bin')
            raw = blob.read_bytes()[entry['offset']:entry['offset'] + entry['length']]
            entries[name] = {'dtype': 'F32', 'shape': entry['shape'],
                             'data_offsets': [offset, offset + len(raw)]}
            payload.extend(raw); offset += len(raw)
        header = json.dumps(entries, separators=(',', ':')).encode()
        header += b' ' * ((-len(header)) % 8)
        (self.checkpoint / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + payload)

    def verify(self):
        return g4.verify_published_data(self.resident, self.tokenizer, self.checkpoint,
                                        expected_witness_sha256=self.fixture.witness_sha)

    def _reset_published(self):
        for path in (self.tokenizer, self.checkpoint):
            shutil.rmtree(path)
            shutil.copytree(self.fixture.root / 'tokenizer', path)
        self._write_towers()

    def _checkpoint_header(self):
        path = self.checkpoint / 'model.safetensors'
        raw = path.read_bytes(); length = struct.unpack('<Q', raw[:8])[0]
        return json.loads(raw[8:8 + length]), raw[8 + length:]

    def _replace_checkpoint(self, header, payload):
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * ((-len(raw)) % 8)
        (self.checkpoint / 'model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + payload)

    def test_published_data_is_source_independent(self):
        shutil.rmtree(self.fixture.root)
        self.assertEqual(self.verify()['model'], 'gemma4-12b')

    def test_real_safetensors_serializer_is_source_independent(self):
        try:
            import numpy as np
            from safetensors.numpy import save_file
        except ImportError:
            self.skipTest('requires numpy and safetensors; run with .venv-export/bin/python3.12')
        manifest = json.loads((self.resident / 'store' / 'manifest.json').read_text())
        tensors = {}
        for name, entry in manifest['towers'].items():
            blob = self.resident / 'store' / 'blobs' / (entry['blob'] + '.bin')
            tensors[name] = np.fromfile(blob, dtype=np.float32, count=entry['length'] // 4,
                                        offset=entry['offset']).reshape(entry['shape'])
        save_file(tensors, str(self.checkpoint / 'model.safetensors'))
        self.assertEqual(self.verify()['model'], 'gemma4-12b')
        shutil.rmtree(self.fixture.root)
        self.assertEqual(self.verify()['model'], 'gemma4-12b')

    def test_tokenizer_and_checkpoint_sidecar_guards(self):
        cases = (
            ('missing tokenizer member', lambda: (self.tokenizer / 'tokenizer.json').unlink()),
            ('extra tokenizer member', lambda: (self.tokenizer / 'extra.json').write_text('{}')),
            ('tampered tokenizer member', lambda: (self.tokenizer / 'tokenizer.json').write_text('{}')),
            ('tampered checkpoint config', lambda: (self.checkpoint / 'config.json').write_text('{}')),
            ('reserved sidecar', lambda: (self.tokenizer / g4.SIDECAR).write_text('{}')),
            ('root alias', self._alias_tokenizer),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                self._reset_published(); mutate()
                with self.assertRaises(g4.VerifiedDataError): self.verify()

    def _alias_tokenizer(self):
        real = self.tokenizer.with_name('tokenizer-real')
        self.tokenizer.rename(real)
        self.tokenizer.symlink_to(real, target_is_directory=True)

    def test_safetensors_tower_guards(self):
        def corrupt_payload():
            path = self.checkpoint / 'model.safetensors'
            raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        def change(which):
            header, payload = self._checkpoint_header()
            name = next(iter(header)); which(header, name)
            self._replace_checkpoint(header, payload)
        cases = (
            ('wrong name', lambda: change(lambda h, n: h.__setitem__('wrong', h.pop(n)))),
            ('wrong dtype', lambda: change(lambda h, n: h[n].update({'dtype': 'F16'}))),
            ('wrong shape', lambda: change(lambda h, n: h[n].update({'shape': [2]}))),
            ('float shape', lambda: change(lambda h, n: h[n].update({'shape': [1.0]}))),
            ('boolean shape', lambda: change(lambda h, n: h[n].update({'shape': [True]}))),
            ('boolean offset', lambda: change(lambda h, n: h[n].update({'data_offsets': [True, 4]}))),
            ('negative offset', lambda: change(lambda h, n: h[n].update({'data_offsets': [-1, 3]}))),
            ('overlap', lambda: change(lambda h, n: h[list(h)[1]].update({'data_offsets': [0, 4]}))),
            ('gap', lambda: change(lambda h, n: h[n].update({'data_offsets': [4, 8]}))),
            ('payload corruption', corrupt_payload),
            ('truncated', lambda: (self.checkpoint / 'model.safetensors').write_bytes(b'\0' * 7)),
            ('oversized header', lambda: (self.checkpoint / 'model.safetensors').write_bytes(struct.pack('<Q', g4.MAX_METADATA_BYTES + 1))),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                self._reset_published(); mutate()
                with self.assertRaises(g4.VerifiedDataError): self.verify()


class Gemma4ProofProducerTests(Gemma4WitnessTests):
    """The proof sidecar binds an already reviewed source snapshot without copying it."""
    def setUp(self):
        super().setUp()
        self.witness_path = self.work / 'reviewed-witness.json'
        self.raw = json.dumps(self.witness(), indent=2).encode('utf-8') + b'\n'
        self.witness_path.write_bytes(self.raw)
        self.expected = hashlib.sha256(self.raw).hexdigest()
        self.proof = self.work / 'proof'

    def _metadata(self):
        paths = [self.witness_path]
        for item in self.root.rglob('*'):
            if item.is_file():
                paths.append(item)
        return {path: (path.stat().st_ino, path.stat().st_mode, path.stat().st_mtime_ns,
                       path.stat().st_nlink) for path in paths}

    def test_missing_nested_proof_parents_are_created_after_validation(self):
        nested = self.work / 'fresh-build' / 'gemma4-12b' / 'proof'
        with self.assertRaises(g4.VerifiedDataError):
            g4.materialize_proof(self.root, self.witness_path, nested,
                                 accepted_data_root=self.owner, expected_witness_sha256='0' * 64)
        self.assertFalse((self.work / 'fresh-build').exists())
        g4.materialize_proof(self.root, self.witness_path, nested,
                             accepted_data_root=self.owner, expected_witness_sha256=self.expected)
        self.assertEqual((nested / g4.SIDECAR).read_bytes(), self.raw)
        self.assertEqual(list(nested.iterdir()), [nested / g4.SIDECAR])

    def test_materializes_exact_reviewed_bytes_without_changing_inputs(self):
        before = self._metadata()
        result = g4.materialize_proof(self.root, self.witness_path, self.proof,
                                      accepted_data_root=self.owner,
                                      expected_witness_sha256=self.expected)
        sidecar = self.proof / g4.SIDECAR
        self.assertEqual(sidecar.read_bytes(), self.raw)
        self.assertNotEqual(sidecar.stat().st_ino, self.witness_path.stat().st_ino)
        self.assertEqual(list(self.proof.iterdir()), [sidecar])
        self.assertEqual(result['model'], 'gemma4-12b')
        self.assertEqual(result['witness_sha256'], self.expected)
        self.assertEqual(self._metadata(), before)

    def test_refuses_bad_pin_source_and_path_overlap_before_output(self):
        cases = (
            ('wrong expected pin', self.root, self.expected[:-1] + '0', self.proof),
            ('wrong model root', self.owner / 'not-gemma4', self.expected, self.proof),
            ('proof below source', self.root, self.expected, self.root / 'proof'),
            ('proof equals witness', self.root, self.expected, self.witness_path),
        )
        for label, source, expected, proof in cases:
            with self.subTest(label=label):
                with self.assertRaises(g4.VerifiedDataError):
                    g4.materialize_proof(source, self.witness_path, proof,
                                         accepted_data_root=self.owner,
                                         expected_witness_sha256=expected)
                self.assertFalse(proof.exists() if proof != self.witness_path else False)
        changed = self.root / 'rf_stack' / 'w0.npy'
        changed.write_bytes(b'changed')
        with self.assertRaises(g4.VerifiedDataError):
            g4.materialize_proof(self.root, self.witness_path, self.proof,
                                 accepted_data_root=self.owner, expected_witness_sha256=self.expected)
        self.assertFalse(self.proof.exists())
        self.make_source()
        (self.root / 'rf_stack' / 'w1.npy').unlink()
        with self.assertRaises(g4.VerifiedDataError):
            g4.materialize_proof(self.root, self.witness_path, self.proof,
                                 accepted_data_root=self.owner, expected_witness_sha256=self.expected)
        self.assertFalse(self.proof.exists())

    def test_aliases_and_existing_nonempty_proof_refuse_without_writes(self):
        real_witness = self.work / 'real-witness.json'
        self.witness_path.rename(real_witness)
        self.witness_path.symlink_to(real_witness)
        with self.assertRaises(g4.VerifiedDataError):
            g4.materialize_proof(self.root, self.witness_path, self.proof,
                                 accepted_data_root=self.owner, expected_witness_sha256=self.expected)
        self.witness_path.unlink(); real_witness.rename(self.witness_path)
        real_proof = self.work / 'real-proof'
        real_proof.mkdir(); self.proof.symlink_to(real_proof, target_is_directory=True)
        with self.assertRaises(g4.VerifiedDataError):
            g4.materialize_proof(self.root, self.witness_path, self.proof,
                                 accepted_data_root=self.owner, expected_witness_sha256=self.expected)
        self.proof.unlink(); self.proof.mkdir(); sentinel = self.proof / 'sentinel'
        sentinel.write_bytes(b'keep')
        with self.assertRaises(g4.VerifiedDataError):
            g4.materialize_proof(self.root, self.witness_path, self.proof,
                                 accepted_data_root=self.owner, expected_witness_sha256=self.expected)
        self.assertEqual(sentinel.read_bytes(), b'keep')

    def test_first_read_snapshot_is_used_when_witness_changes_during_verification(self):
        original = g4.verify_source
        def mutate_after_snapshot(source, document, *, accepted_data_root):
            self.witness_path.write_bytes(b'{"changed":true}')
            return original(source, document, accepted_data_root=accepted_data_root)
        g4.verify_source = mutate_after_snapshot
        try:
            g4.materialize_proof(self.root, self.witness_path, self.proof,
                                 accepted_data_root=self.owner, expected_witness_sha256=self.expected)
        finally:
            g4.verify_source = original
        self.assertEqual((self.proof / g4.SIDECAR).read_bytes(), self.raw)


if __name__ == '__main__': unittest.main()
