#!/usr/bin/env python3
"""CPU-only contract fixtures for the verified-data witness; no model, toolchain or device work."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verified_gemma_data as data

REPO = Path(__file__).resolve().parents[1]
SPEC = data.spec_for('gemma3-270m')
CONTRACT_IMPORTS = {'argparse', 'ast', 'hashlib', 'json', 'os', 're', 'shutil', 'struct', 'sys',
                    'dataclasses', 'datetime', 'pathlib'}
ITEMSIZES = {'<f4': 4, '|i1': 1}
# A small dump with the real Gemma3 shape: one plain tensor, one fully packed tensor (the dump names
# the tensor itself in `packed`), and one tensor whose packed part is a separate companion array.
TENSORS = {'w.norm': [], 'w.gate': ['w.gate'], 'w.embed': ['w.embed.headpack']}
PACKED = ['w.gate', 'w.embed.headpack']
ARRAYS = {'w.norm.npy': ('<f4', [10], 1), 'w.gate.npy': ('|i1', [40], 0),
          'w.embed.npy': ('<f4', [4, 3], 2), 'w.embed.headpack.npy': ('|i1', [24], 3)}


def npy_bytes(descr, shape, seed) -> bytes:
    header = "{'descr': %r, 'fortran_order': False, 'shape': %s, }" % (descr, tuple(shape))
    pad = (-10 - len(header) - 1) % 64  # numpy aligns 10 + len(header) to a multiple of 64
    raw = (header + ' ' * pad + '\n').encode('latin-1')
    count = ITEMSIZES[descr]
    for axis in shape:
        count *= axis
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(raw)) + raw + bytes(
        (seed + index) % 251 for index in range(count))


def npy_bytes_v2(descr, shape, seed, *, version=(2, 0)) -> bytes:
    """A standard v2/v3 array: magic+version is 8 bytes, the length is a uint32, header starts at 12."""
    header = "{'descr': %r, 'fortran_order': False, 'shape': %s, }" % (descr, tuple(shape))
    pad = (-12 - len(header) - 1) % 64  # numpy aligns 12 + len(header) to a multiple of 64
    raw = (header + ' ' * pad + '\n').encode('latin-1')
    count = ITEMSIZES[descr]
    for axis in shape:
        count *= axis
    return b'\x93NUMPY' + bytes(version) + struct.pack('<I', len(raw)) + raw + bytes(
        (seed + index) % 251 for index in range(count))


def gemma_dump(root: Path, *, quant=None, grid=None, done=None, arrays=None, files=None) -> dict:
    """Write a schema-conforming dump, with named overrides for the negative fixtures."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(SPEC.quant, packed=list(PACKED),
                    note='packed arrays are np.int8 on-wire bytes, NOT values -- never .astype()')
    manifest.update(quant or {})
    axes = {'cols': SPEC.cols, 'layers': None, 'quant': SPEC.quant['dtype'],
            'quant_clip_search': SPEC.quant['clip_search'], 'quant_full_range': SPEC.quant['full_range'],
            'quant_group': SPEC.quant['group_size'], 'quant_layout': SPEC.quant['layout'],
            'quant_leaves': SPEC.leaves, 'quant_scale_dtype': SPEC.quant['scale_dtype']}
    axes.update(grid or {})
    written = {'quant.json': json.dumps(manifest, indent=2),
               data.PROGRESS_RECORD: json.dumps({'fmt': json.dumps(axes, sort_keys=True),
                                                 'done': dict(TENSORS if done is None else done)})}
    for name, value in (arrays or ARRAYS).items():
        written[name] = npy_bytes(*value)
    written.update(files or {})
    for name, value in written.items():
        (root / name).write_bytes(value if isinstance(value, bytes) else value.encode())
    return {name: (root / name) for name in written}


def witness_file(source: Path, target: Path, **legacy) -> Path:
    document = data.build_witness(source, model=SPEC.model,
                                  provenance='retained legacy dump; no quantizer executed here',
                                  legacy=legacy or {'legacy_receipt': 'a' * 64})
    return data.write_witness(target, document)


class WitnessDocumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = self.root / 'data-owner'
        self.source = self.owner / SPEC.model / 'weights_qat_int4g32'
        self.dump = gemma_dump(self.source)

    def witness(self, **legacy):
        return witness_file(self.source, self.root / 'reviewed-witness.json', **legacy)

    def test_record_reads_remain_bounded_if_a_file_grows_after_stat(self):
        record = self.root / 'growing-record.json'
        record.write_bytes(b'12345')
        original = Path.stat

        def grow(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            if path == record and kwargs.get('follow_symlinks', True):
                record.write_bytes(b'x' * 20)
            return info

        with unittest.mock.patch.object(Path, 'stat', grow):
            with self.assertRaisesRegex(data.VerifiedDataError, 'exceeds'):
                data._read_limited(record, 10, 'test record')

    def test_the_contract_needs_only_the_standard_library(self):
        tree = ast.parse((REPO / 'scripts/verified_gemma_data.py').read_text())
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                     for alias in node.names}
        self.assertEqual(imported, CONTRACT_IMPORTS, 'a prepared verifier must not need the build SDK')

    def test_witness_binds_model_format_members_and_origin(self):
        document = data.load_witness(self.witness(legacy_receipt='b' * 64, legacy_output_digest='c' * 64))
        self.assertEqual(document['model'], SPEC.model)
        self.assertEqual(document['variant'], data.VARIANT)
        self.assertEqual(document['format'], SPEC.quant)
        self.assertEqual(set(document['members']), set(self.dump))
        self.assertEqual(document['origin']['source_root'], str(self.source.resolve()))
        self.assertEqual(document['origin']['legacy'], {'legacy_receipt': 'b' * 64,
                                                       'legacy_output_digest': 'c' * 64})
        self.assertEqual(document['payload_digest'], data.canonical_digest(document['members']))
        for name in ARRAYS:
            header = document['members'][name]['header']
            self.assertEqual(header['descr'], ARRAYS[name][0])
            self.assertEqual(header['shape'], ARRAYS[name][1])
            self.assertFalse(header['fortran_order'])

    def test_digest_is_recomputed_by_the_consumer_not_trusted_from_the_file(self):
        path = self.witness()
        document = json.loads(path.read_text())
        next(iter(document['members'].values()))['bytes'] += 1
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(data.VerifiedDataError, 'payload_digest does not bind'):
            data.load_witness(path)

    def test_tampered_member_hash_and_short_document_are_refused(self):
        path = self.witness()
        document = json.loads(path.read_text())
        document['members']['w.norm.npy']['sha256'] = '0' * 64
        document['payload_digest'] = data.canonical_digest(document['members'])
        path.write_text(json.dumps(document))
        # A digest binds the declaration; only the streamed payload can prove the bytes.
        with self.assertRaisesRegex(data.VerifiedDataError, 'member changed'):
            data.check_payload(self.source, data.load_witness(path), spec=SPEC, label='source')
        path.write_text(json.dumps({'version': data.VERSION, 'kind': data.KIND, 'model': SPEC.model,
                                    'variant': data.VARIANT, 'created_utc': 'now', 'origin': {},
                                    'format': {}, 'members': {}}))
        with self.assertRaisesRegex(data.VerifiedDataError, 'fields must be'):
            data.load_witness(path)
        self.dump['quant.json'].write_text('{"a": 1, "a": 2}')
        with self.assertRaisesRegex(data.VerifiedDataError, 'duplicate JSON field'):
            data.build_witness(self.source, model=SPEC.model, provenance='x', legacy={})

    def test_a_second_gemma_never_falls_back_and_non_gemma_is_rejected(self):
        self.assertIs(data.spec_for('gemma3-270m'), SPEC)
        with self.assertRaisesRegex(data.VerifiedDataError, 'gemma4-12b.*not implemented yet'):
            data.spec_for('gemma4-12b')
        with self.assertRaisesRegex(data.VerifiedDataError, 'Gemma-only'):
            data.spec_for('qwen3-0.6b')

    def test_headers_are_checked_without_loading_values(self):
        # A truncated dump is a layout defect, and it is caught without reading array values.
        path = self.source / 'w.gate.npy'
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(data.VerifiedDataError, 'declares .* but holds'):
            data.read_npy_header(path)
        with self.assertRaisesRegex(data.VerifiedDataError, 'declares .* but holds'):
            self.witness()
        path.write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', 64)
                         + b"{'descr': '|V8', 'fortran_order': False, 'shape': (1,), }".ljust(64, b' '))
        with self.assertRaisesRegex(data.VerifiedDataError, 'unsupported numpy dtype'):
            data.read_npy_header(path)


class FormatSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'weights_qat_int4g32'
        self.dump = gemma_dump(self.source)

    def rebuild(self, **overrides):
        shutil.rmtree(self.source)
        self.dump = gemma_dump(self.source, **overrides)

    def test_declared_axes_are_the_served_grid(self):
        self.assertEqual(data.check_format(self.source, SPEC), set(PACKED))
        self.assertIsNone(data.check_layout(self.source, SPEC, set(PACKED)))

    def test_wrong_group_dtype_range_or_clip_refuses(self):
        for axis, value in (('group_size', 64), ('dtype', 'int8'), ('full_range', False),
                            ('clip_search', True), ('scale_dtype', 'f16'), ('layout', 'interleaved')):
            with self.subTest(axis=axis):
                self.rebuild(quant={axis: value})
                with self.assertRaisesRegex(data.VerifiedDataError, 'disagrees with the verified-data spec'):
                    data.check_format(self.source, SPEC)

    def test_dump_grid_and_served_manifest_must_agree(self):
        for axis, value in (('quant_group', 64), ('quant_leaves', 'gate_proj,up_proj'), ('cols', 16)):
            with self.subTest(axis=axis):
                self.rebuild(grid={axis: value})
                with self.assertRaisesRegex(data.VerifiedDataError, f'fmt {axis}'):
                    data.check_format(self.source, SPEC)

    def test_missing_or_extra_member_breaks_the_declared_layout(self):
        for name in ('w.gate.npy', 'w.embed.headpack.npy'):
            with self.subTest(missing=name):
                shutil.rmtree(self.source)
                dump = gemma_dump(self.source)
                dump[name].unlink()
                with self.assertRaisesRegex(data.VerifiedDataError, 'layout disagrees'):
                    data.check_format(self.source, SPEC)
        (self.source / 'w.extra.npy').write_bytes(npy_bytes('|i1', [8], 9))
        with self.assertRaisesRegex(data.VerifiedDataError, 'extra='):
            data.check_format(self.source, SPEC)

    def test_group_membership_and_layout_must_match_the_packed_set(self):
        # Moving one tensor's packed companion changes which arrays are int8 on-wire bytes.
        self.rebuild(quant={'packed': ['w.embed.headpack']})
        with self.assertRaisesRegex(data.VerifiedDataError, 'packed disagrees with'):
            data.check_format(self.source, SPEC)
        self.rebuild(arrays=dict(ARRAYS, **{'w.gate.npy': ('<f4', [40], 0)}))
        with self.assertRaisesRegex(data.VerifiedDataError, 'packed array is <f4'):
            data.check_layout(self.source, SPEC, set(PACKED))
        self.rebuild(arrays=dict(ARRAYS, **{'w.gate.npy': ('|i1', [8, 5], 0)}))
        with self.assertRaisesRegex(data.VerifiedDataError, 'packed array has rank 2'):
            data.check_layout(self.source, SPEC, set(PACKED))

    def test_hostile_array_headers_are_refused(self):
        for name, raw in (('broken.npy', b'not an array at all'),
                          ('oversize.npy', b'\x93NUMPY\x01\x00' + struct.pack('<H', 0xFFFF) + b'{}'),
                          ('short.npy', b'\x93NUMPY\x01\x00\x05\x00{}'),
                          ('exec.npy', b'\x93NUMPY\x01\x00' + struct.pack('<H', 64)
                           + b"__import__('os').system('echo bad')".ljust(64, b' ')),
                          ('fields.npy', b'\x93NUMPY\x01\x00' + struct.pack('<H', 64)
                           + b"{'descr': '<f4'}".ljust(64, b' '))):
            with self.subTest(name=name):
                path = self.source / name
                path.write_bytes(raw)
                with self.assertRaisesRegex(data.VerifiedDataError,
                                            'not a numpy|oversized header|truncated numpy header'
                                            '|unparseable numpy header|lacks descr'):
                    data.read_npy_header(path)
                path.unlink()


class ConfinementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = self.root / 'data-owner'
        self.source = self.owner / SPEC.model / 'weights'
        gemma_dump(self.source)

    def witness(self, source=None):
        return witness_file(source or self.source, self.root / 'reviewed-witness.json')

    def test_source_must_stay_inside_the_accepted_data_owner(self):
        document = data.load_witness(self.witness())
        outside = self.root / 'somewhere-else' / 'weights'
        gemma_dump(outside)
        other = witness_file(outside, self.root / 'other-witness.json')
        for source, witness in ((outside, document), (outside, data.load_witness(other))):
            with self.assertRaisesRegex(data.VerifiedDataError, 'escapes the accepted data root'):
                data.verify_source(source, witness, spec=SPEC, accepted_data_root=self.owner)

    def test_symlinked_source_or_ancestor_is_refused(self):
        document = data.load_witness(self.witness())
        linked = self.owner / 'alias'
        linked.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(data.VerifiedDataError, 'symlink in the data source path'):
            data.verify_source(linked, document, spec=SPEC, accepted_data_root=self.owner)
        moved = self.root / 'real-owner'
        moved.mkdir()
        shutil.move(self.owner / SPEC.model, moved / SPEC.model)
        (self.owner / SPEC.model).symlink_to(moved / SPEC.model, target_is_directory=True)
        with self.assertRaisesRegex(data.VerifiedDataError, 'symlink in the data source path'):
            data.verify_source(self.source, document, spec=SPEC, accepted_data_root=self.owner)
        owner_link = self.root / 'owner-link'
        owner_link.symlink_to(self.owner, target_is_directory=True)
        with self.assertRaisesRegex(data.VerifiedDataError, 'symlink'):
            data.verify_source(self.source, document, spec=SPEC, accepted_data_root=owner_link)

    def test_witness_origin_must_be_the_requested_source(self):
        document = data.load_witness(self.witness())
        sibling = self.owner / 'other-weights'
        gemma_dump(sibling)
        with self.assertRaisesRegex(data.VerifiedDataError, 'witness origin'):
            data.verify_source(sibling, document, spec=SPEC, accepted_data_root=self.owner)

    def test_witness_cannot_name_an_escaping_member(self):
        path = self.witness()
        original = path.read_text()
        document = json.loads(original)
        for name in ('../outside.npy', '/etc/passwd', './w.norm.npy', 'nested/../w.norm.npy',
                     data.SIDECAR, ''):
            with self.subTest(name=name):
                tampered = json.loads(original)
                tampered['members'][name] = {'sha256': '0' * 64, 'bytes': 1}
                tampered['payload_digest'] = data.canonical_digest(tampered['members'])
                path.write_text(json.dumps(tampered))
                with self.assertRaisesRegex(data.VerifiedDataError, 'unsafe witness member|witness sidecar'):
                    data.load_witness(path)
        path.write_text(original)
        for field, value in (('variant', 'source-quantized'), ('version', 99), ('kind', 'other'),
                             ('origin', {'source_root': '/x', 'provenance': '', 'legacy': {}}),
                             ('members', {})):
            with self.subTest(field=field):
                tampered = json.loads(original)
                tampered[field] = value
                tampered['payload_digest'] = data.canonical_digest(tampered['members'])
                path.write_text(json.dumps(tampered))
                with self.assertRaises(data.VerifiedDataError):
                    data.load_witness(path)


class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = self.root / 'data-owner'
        self.source = self.owner / SPEC.model / 'weights'
        self.dump = gemma_dump(self.source)
        self.witness_path = witness_file(self.source, self.root / 'reviewed-witness.json')
        self.reviewed_before = (self.witness_path.stat().st_ino, self.witness_path.stat().st_nlink,
                                self.witness_path.stat().st_mode & 0o777,
                                self.witness_path.stat().st_size, self.witness_path.read_bytes())
        self.target = self.root / 'staged' / 'weights'
        self.before = {name: (path.stat().st_ino, path.stat().st_nlink, path.stat().st_mode,
                              path.stat().st_mtime_ns, path.read_bytes())
                       for name, path in sorted(self.dump.items())}

    def materialize(self):
        return data.materialize(self.source, self.target, self.witness_path, model=SPEC.model,
                                accepted_data_root=self.owner)

    def test_materialize_copies_privately_and_publishes_the_witness(self):
        summary = self.materialize()
        self.assertEqual(summary['members'], len(self.dump))
        published = data.load_witness(self.target / data.SIDECAR)
        self.assertEqual(published['payload_digest'], summary['payload_digest'])
        for name, path in sorted(self.dump.items()):
            self.assertEqual((path.stat().st_ino, path.stat().st_nlink, path.stat().st_mode,
                              path.stat().st_mtime_ns, path.read_bytes()), self.before[name],
                             f'the retained member {name} was rewritten')
            copy = self.target / name
            self.assertNotEqual(copy.stat().st_ino, path.stat().st_ino, name)
            self.assertEqual(copy.stat().st_nlink, 1, name)
            self.assertEqual(copy.stat().st_mode & 0o777, 0o644, name)
            self.assertEqual(copy.read_bytes(), path.read_bytes(), name)
        sidecar = (self.target / data.SIDECAR).stat()
        self.assertNotEqual(sidecar.st_ino, self.reviewed_before[0], 'metadata aliases the reviewed witness')
        self.assertEqual((self.witness_path.stat().st_ino, self.witness_path.stat().st_nlink,
                          self.witness_path.stat().st_mode & 0o777, self.witness_path.stat().st_size,
                          self.witness_path.read_bytes()), self.reviewed_before,
                         'the reviewed witness was rewritten or hardlinked by materialization')

    def test_no_published_member_shares_an_inode_with_the_retained_data(self):
        self.materialize()
        published = {path.name: path.stat().st_ino for path in self.target.iterdir()}
        retained = {path.name: path.stat().st_ino for path in self.source.iterdir()}
        self.assertFalse(set(published.values()) & set(retained.values()),
                         'a published payload member aliases a retained data inode')

    def test_drift_before_and_during_materialization_refuse(self):
        self.dump['w.norm.npy'].write_bytes(npy_bytes('<f4', [10], 77))
        with self.assertRaisesRegex(data.VerifiedDataError, 'source member changed'):
            self.materialize()
        self.assertFalse(self.target.exists(), 'a refused materialization left output behind')
        self.dump['w.norm.npy'].write_bytes(self.before['w.norm.npy'][4])
        original = data.shutil.copyfileobj

        def drift(handle, destination, *args, **kwargs):
            original(handle, destination, *args, **kwargs)
            if Path(destination.name).name == 'w.gate.npy':
                (self.source / 'w.embed.npy').write_bytes(npy_bytes('<f4', [4, 3], 99))

        with unittest.mock.patch.object(data.shutil, 'copyfileobj', drift):
            with self.assertRaisesRegex(data.VerifiedDataError,
                                        'data source changed while materializing'):
                self.materialize()
        with self.assertRaisesRegex(data.VerifiedDataError, 'refusing to materialize'):
            self.materialize()

    def test_target_may_not_alias_the_source_or_an_existing_root(self):
        with self.assertRaisesRegex(data.VerifiedDataError, 'may not overlap the data source'):
            data.materialize(self.source, self.source / 'nested', self.witness_path, model=SPEC.model,
                             accepted_data_root=self.owner)
        self.target.mkdir(parents=True)
        (self.target / 'already-here.json').write_text('{}')
        with self.assertRaisesRegex(data.VerifiedDataError, 'refusing to materialize'):
            self.materialize()
        linked = self.root / 'linked-target'
        linked.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(data.VerifiedDataError, 'refusing to materialize'):
            data.materialize(self.source, linked, self.witness_path, model=SPEC.model,
                             accepted_data_root=self.owner)

    def test_symlinked_member_in_the_source_is_refused(self):
        outside = self.owner / 'elsewhere.npy'
        outside.write_bytes(npy_bytes('|i1', [40], 0))
        self.dump['w.gate.npy'].unlink()
        (self.source / 'w.gate.npy').symlink_to(outside)
        with self.assertRaisesRegex(data.VerifiedDataError, 'symlink'):
            self.materialize()
        self.assertEqual(outside.stat().st_nlink, 1)
        self.assertEqual((self.root / 'reviewed-witness.json').read_bytes(),
                         self.witness_path.read_bytes())

    def test_a_witness_for_another_model_is_refused(self):
        document = json.loads(self.witness_path.read_text())
        document['model'] = 'qwen3-0.6b'
        document['payload_digest'] = data.canonical_digest(document['members'])
        self.witness_path.write_text(json.dumps(document))
        with self.assertRaisesRegex(data.VerifiedDataError, 'witness is for qwen3-0.6b'):
            self.materialize()

    def test_the_published_sidecar_is_the_consumed_witness_byte_for_byte(self):
        """A later receipt can only bind the witness SHA it never had to read again.

        Re-serializing an equivalent document would publish different bytes under the same meaning,
        so the recorded consumed `--witness` input could never be proved to be the sidecar.
        """
        self.materialize()
        sidecar = self.target / data.SIDECAR
        self.assertEqual(sidecar.read_bytes(), self.reviewed_before[4],
                         'the published witness is not the reviewed witness bytes')
        self.assertEqual(hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                         hashlib.sha256(self.reviewed_before[4]).hexdigest())
        self.assertNotEqual(sidecar.stat().st_ino, self.reviewed_before[0])
        self.assertEqual(sidecar.stat().st_nlink, 1)
        self.assertEqual(json.loads(sidecar.read_text()), json.loads(self.reviewed_before[4].decode()))


class DurabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = self.root / 'data-owner'
        self.source = self.owner / SPEC.model / 'weights'
        self.dump = gemma_dump(self.source)
        self.reviewed = witness_file(self.source, self.root / 'reviewed-witness.json')
        self.published = self.root / 'candidate' / SPEC.model / 'weights_qat_int4g32'
        data.materialize(self.source, self.published, self.reviewed, model=SPEC.model,
                         accepted_data_root=self.owner)

    def test_published_verification_survives_losing_the_old_input_and_the_review_copy(self):
        before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(self.published.iterdir())}
        shutil.rmtree(self.root / 'data-owner')
        self.reviewed.unlink()
        self.assertEqual({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(self.published.iterdir())}, before)
        witness = data.load_witness(self.published / data.SIDECAR)
        self.assertEqual(witness['origin']['source_root'], str(self.source.resolve()),
                         'the origin is a historical string, not a live dependency')
        self.assertEqual(data.verify_published(self.published, spec=SPEC)['payload_digest'],
                         witness['payload_digest'])
        self.assertFalse(self.source.exists())
        self.assertFalse(self.reviewed.exists())

    def test_published_drift_or_missing_member_is_refused(self):
        self.assertEqual(data.verify_published(self.published, spec=SPEC)['payload_digest'],
                         data.load_witness(self.published / data.SIDECAR)['payload_digest'])
        for harm in ('hash', 'missing', 'extra', 'symlink', 'header', 'schema', 'metadata-symlink'):
            with self.subTest(harm=harm):
                copy = self.root / f'copy-{harm}'
                shutil.copytree(self.published, copy)
                if harm == 'hash':
                    path = copy / 'w.gate.npy'
                    path.write_bytes(path.read_bytes()[:-1] + bytes([path.read_bytes()[-1] ^ 0xFF]))
                elif harm == 'missing':
                    (copy / 'w.embed.headpack.npy').unlink()
                elif harm == 'extra':
                    (copy / 'not-witnessed.npy').write_bytes(npy_bytes('|i1', [8], 5))
                elif harm == 'symlink':
                    outside = self.root / 'elsewhere-external.npy'
                    outside.write_bytes((copy / 'w.norm.npy').read_bytes())
                    (copy / 'w.norm.npy').unlink()
                    (copy / 'w.norm.npy').symlink_to(outside)
                elif harm == 'header':
                    (copy / 'w.embed.npy').write_bytes(npy_bytes('<f4', [12], 2))
                elif harm == 'metadata-symlink':
                    target = copy / data.SIDECAR
                    target.unlink()
                    target.symlink_to(self.root / 'reviewed-witness.json')
                else:
                    manifest = json.loads((copy / 'quant.json').read_text())
                    manifest['group_size'] = 128
                    (copy / 'quant.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(data.VerifiedDataError,
                                            'member set disagrees|member changed|array header changed'
                                            '|regular file|disagrees with the verified-data spec'
                                            '|symlinked dumped array|symlinked data witness'):
                    data.verify_published(copy, spec=SPEC)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = self.root / 'data-owner'
        self.source = self.owner / SPEC.model / 'weights'
        gemma_dump(self.source)

    def run_cli(self, *argv):
        return subprocess.run([sys.executable, '-B', str(REPO / 'scripts/verified_gemma_data.py')]
                              + list(map(str, argv)), capture_output=True, text=True,
                              env=os.environ | {'PYTHONDONTWRITEBYTECODE': '1'})

    def test_inspect_build_materialize_verify_published_round_trip(self):
        inspect = self.run_cli('--model', SPEC.model, '--inspect', '--source', self.source)
        self.assertEqual(inspect.returncode, 0, inspect.stderr)
        self.assertEqual(json.loads(inspect.stdout), {'arrays': 4, 'model': SPEC.model, 'packed': 2})
        witness = self.root / 'reviewed-witness.json'
        built = self.run_cli('--model', SPEC.model, '--build-witness', '--source', self.source,
                             '--witness', witness, '--provenance', 'retained legacy dump',
                             '--legacy-field', 'legacy_receipt=' + 'd' * 64)
        self.assertEqual(built.returncode, 0, built.stderr)
        target = self.root / 'staged' / 'weights'
        done = self.run_cli('--model', SPEC.model, '--materialize', '--source', self.source,
                            '--out', target, '--witness', witness,
                            '--accepted-data-root', self.owner)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue((target / data.SIDECAR).is_file())
        checked = self.run_cli('--model', SPEC.model, '--verify-published', '--out', target)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)['payload_digest'],
                         json.loads(done.stdout)['payload_digest'])

    def test_a_second_gemma_refuses_on_the_command_line(self):
        result = self.run_cli('--model', 'gemma4-12b', '--inspect', '--source', self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not implemented yet', result.stderr)
        self.assertIn('falling back to the source-quantizing recipe', result.stderr)

    def test_materialize_needs_its_full_authority(self):
        result = self.run_cli('--model', SPEC.model, '--materialize', '--source', self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--materialize needs', result.stderr)


class EscapingAndAliasedPathTests(unittest.TestCase):
    """What the helper itself claims about the roots it reads and writes, before any caller's gate."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = self.root / 'data-owner'
        self.source = self.owner / SPEC.model / 'weights'
        gemma_dump(self.source)
        self.witness_path = witness_file(self.source, self.root / 'reviewed-witness.json')

    def test_a_dotdot_component_is_not_inside_the_accepted_data_owner(self):
        outside = self.root / 'outside-owner' / SPEC.model / 'weights'
        gemma_dump(outside)
        document = data.load_witness(witness_file(outside, self.root / 'outside-witness.json'))
        escaping = self.owner / '..' / 'outside-owner' / SPEC.model / 'weights'
        with self.assertRaisesRegex(data.VerifiedDataError, r'\.\.'):
            data.verify_source(escaping, document, spec=SPEC, accepted_data_root=self.owner)
        with self.assertRaisesRegex(data.VerifiedDataError, r'\.\.'):
            data.verify_source(outside, document, spec=SPEC, accepted_data_root=self.root / 'x' / '..')
        with self.assertRaisesRegex(data.VerifiedDataError, r'\.\.'):
            data.build_witness(escaping, model=SPEC.model, provenance='x', legacy={})

    def test_materialize_refuses_a_symlinked_output_ancestor_before_creating_anything(self):
        redirected = self.root / 'redirected'
        redirected.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(redirected, target_is_directory=True)
        with self.assertRaisesRegex(data.VerifiedDataError, 'symlink in the materialization target path'):
            data.materialize(self.source, alias / 'weights', self.witness_path, model=SPEC.model,
                             accepted_data_root=self.owner)
        self.assertEqual(list(redirected.iterdir()), [], 'the refusal wrote through the alias')
        self.assertFalse((self.root / 'alias' / 'weights').exists())

    def test_witness_publication_refuses_a_symlinked_ancestor(self):
        redirected = self.root / 'redirected'
        redirected.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(redirected, target_is_directory=True)
        with self.assertRaisesRegex(data.VerifiedDataError, 'symlink in the witness output path'):
            data.write_witness(alias / 'witness.json', data.load_witness(self.witness_path))
        self.assertEqual(list(redirected.iterdir()), [])

    def test_a_published_root_reached_through_an_alias_is_refused(self):
        published = self.root / 'published'
        data.materialize(self.source, published, self.witness_path, model=SPEC.model,
                         accepted_data_root=self.owner)
        alias = self.root / 'alias'
        alias.symlink_to(published, target_is_directory=True)
        with self.assertRaisesRegex(data.VerifiedDataError, 'symlink in the published path'):
            data.verify_published(alias, spec=SPEC)


class WitnessDeclarationConsistencyTests(unittest.TestCase):
    """A reviewed declaration that contradicts the bytes or the fixed spec is not a contract."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = self.root / 'data-owner'
        self.source = self.owner / SPEC.model / 'weights'
        self.dump = gemma_dump(self.source)
        self.witness_path = witness_file(self.source, self.root / 'reviewed-witness.json')
        self.document = json.loads(self.witness_path.read_text())

    def tampered(self, **overrides):
        document = json.loads(json.dumps(self.document))
        document.update(overrides)
        path = self.root / 'tampered-witness.json'
        path.write_text(json.dumps(document))
        return path

    def test_a_format_that_contradicts_its_own_payload_is_refused_at_every_gate(self):
        declaration = dict(self.document['format'], group_size=64)
        path = self.tampered(format=declaration)
        # payload_digest binds the members, so the file still loads: only the spec comparison catches it.
        document = data.load_witness(path)
        with self.assertRaisesRegex(data.VerifiedDataError, 'witness format disagrees'):
            data.verify_source(self.source, document, spec=SPEC, accepted_data_root=self.owner)
        with self.assertRaisesRegex(data.VerifiedDataError, 'witness format disagrees'):
            data.materialize(self.source, self.root / 'staged', path, model=SPEC.model,
                             accepted_data_root=self.owner)
        published = self.root / 'published'
        data.materialize(self.source, published, self.witness_path, model=SPEC.model,
                         accepted_data_root=self.owner)
        shutil.copytree(published, self.root / 'declared')
        (self.root / 'declared' / data.SIDECAR).write_bytes(path.read_bytes())
        with self.assertRaisesRegex(data.VerifiedDataError, 'witness format disagrees'):
            data.verify_published(self.root / 'declared', spec=SPEC)

    def test_a_witness_for_another_model_is_refused_at_the_source_gate_too(self):
        path = self.tampered(model='qwen3-0.6b')
        document = data.load_witness(path)
        with self.assertRaisesRegex(data.VerifiedDataError, 'witness declares qwen3-0.6b'):
            data.verify_source(self.source, document, spec=SPEC, accepted_data_root=self.owner)

    def test_partial_layer_metadata_is_refused_while_a_full_dump_is_accepted(self):
        # The owner records 18 layers (llm_decode_spec.py GEMMA3_270M); a null fmt.layers means "all".
        self.assertEqual(SPEC.n_layers, 18)
        for layers in (2, 17, 19):
            with self.subTest(layers=layers):
                shutil.rmtree(self.source)
                gemma_dump(self.source, grid={'layers': layers})
                with self.assertRaisesRegex(data.VerifiedDataError, 'fmt layers'):
                    data.check_format(self.source, SPEC)
        for layers in (None, 18):
            with self.subTest(layers=layers):
                shutil.rmtree(self.source)
                gemma_dump(self.source, grid={'layers': layers})
                self.assertTrue(data.check_format(self.source, SPEC))

    def test_the_generator_layout_vocabulary_is_what_the_dump_can_actually_write(self):
        self.assertEqual(data.LAYOUTS, ('header_first', 'row_group_planar'))
        self.assertEqual(SPEC.quant['layout'], 'header_first')
        self.assertIn(SPEC.quant['scale_dtype'], data.SCALE_DTYPES)


class NumpyHeaderGrammarTests(unittest.TestCase):
    """Header-only reads of the standard .npy grammar: supported versions, explicit refusals."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def read(self, raw: bytes, name='probe.npy'):
        path = self.root / name
        path.write_bytes(raw)
        return data.read_npy_header(path)

    def test_supported_versions_are_read_without_loading_values(self):
        for major, raw in ((1, npy_bytes('<f4', [4, 3], 6)),
                           (2, npy_bytes_v2('<f4', [4, 3], 6)),
                           (3, npy_bytes_v2('<f4', [4, 3], 6, version=(3, 0)))):
            with self.subTest(major=major):
                self.assertEqual(self.read(raw),
                                 {'descr': '<f4', 'fortran_order': False, 'shape': [4, 3]})

    def test_unsupported_versions_minors_and_truncated_headers_are_refused(self):
        v1 = npy_bytes('<f4', [3], 4)
        v2 = npy_bytes_v2('<f4', [3], 4)
        for label, raw, why in (('v1-minor', v1[:7] + b'\x01' + v1[8:], 'minor version'),
                                ('v2-minor', v2[:7] + b'\x01' + v2[8:], 'minor version'),
                                ('major-4', b'\x93NUMPY\x04\x00' + v2[8:], 'unsupported numpy magic'),
                                ('major-0', b'\x93NUMPY\x00\x00' + v2[8:], 'unsupported numpy magic'),
                                ('truncated-v2', v2[:20], 'truncated numpy header'),
                                ('not-numpy', b'not an array at all', 'not a numpy'),
                                ('void-dtype', self.void_header(), 'unsupported numpy dtype')):
            with self.subTest(label=label):
                with self.assertRaisesRegex(data.VerifiedDataError, why):
                    self.read(raw)

    def void_header(self) -> bytes:
        raw = b"{'descr': '|V8', 'fortran_order': False, 'shape': (1,), }".ljust(64, b' ')
        return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(raw)) + raw


if __name__ == '__main__':
    unittest.main()
