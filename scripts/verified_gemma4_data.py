# SPDX-License-Identifier: Apache-2.0
"""Source-only CPU witness for the fixed Gemma4 retained-data input."""
import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from verified_gemma_data import (VerifiedDataError, canonical_digest, declared_path,
                                 read_npy_header, safe_member, stream_hash)

KIND = 'xdna-verified-data-witness'
VERSION = 2
MODEL = 'gemma4-12b'
VARIANT = 'verified-data'
SIDECAR = '.verified-data-witness.json'
MAX_WITNESS_BYTES = 8 << 20
MAX_METADATA_BYTES = 1 << 20
ROLE_PATHS = {'retained-store': 'store', 'retained-rf-stack': 'rf_stack',
              'build-config': 'hf_config', 'tokenizer-source': 'tokenizer'}
GEOMETRY = {'model_type': 'gemma4_unified_text', 'hidden_size': 3840,
            'intermediate_size': 15360, 'num_hidden_layers': 48,
            'num_attention_heads': 16, 'num_key_value_heads': 8,
            'head_dim': 256, 'vocab_size': 262144}
STORE_GEOMETRY = {'model': MODEL, 'num_layers': 48, 'hidden_size': 3840,
                  'intermediate_size': 15360,
                  'full_attention_layers': [5, 11, 17, 23, 29, 35, 41, 47]}
HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def _fail(message):
    raise VerifiedDataError('Gemma4 witness: ' + message)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _read(path, limit, label):
    if path.is_symlink() or not path.is_file():
        _fail('%s is not a regular file: %s' % (label, path))
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        _fail('%s exceeds byte limit: %s' % (label, path))
    return raw


def _json(raw, label):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail('duplicate JSON field %r in %s' % (key, label))
            result[key] = value
        return result
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs)
    except (UnicodeDecodeError, ValueError) as error:
        _fail('invalid JSON %s: %s' % (label, error))


def _strict_json_equal(actual, expected):
    """Compare JSON values without Python's bool/int or int/float equivalence."""
    return canonical_digest(actual) == canonical_digest(expected)


def _member(path):
    digest, size = stream_hash(path)
    member = {'sha256': digest, 'bytes': size}
    if path.suffix == '.npy':
        member['header'] = read_npy_header(path)
    return member


def _role_root(root, role):
    checked = declared_path(root, role + ' root')
    if not checked.is_dir():
        _fail(role + ' root is not a directory')
    return checked


def _scan_receipt_files(root):
    found = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            _fail('symlink in selected receipt role: %s' % path)
        if path.is_file():
            found[path.relative_to(root).as_posix()] = path
        elif not path.is_dir():
            _fail('non-regular selected receipt member: %s' % path)
    return found


def _receipt_role(root, name):
    root = _role_root(root, name)
    receipt_path = root / '.recipe-manifest.json'
    receipt_raw = _read(receipt_path, MAX_METADATA_BYTES, name + ' receipt')
    receipt = _json(receipt_raw, str(receipt_path))
    if not isinstance(receipt, dict) or not isinstance(receipt.get('files'), dict):
        _fail(name + ' receipt must contain files map')
    actual = _scan_receipt_files(root)
    actual.pop('.recipe-manifest.json', None)
    claimed = receipt['files']
    if set(claimed) != set(actual):
        _fail(name + ' receipt inventory differs from actual files')
    members = {}
    for relative, path in actual.items():
        safe_member(relative)
        expected = claimed[relative]
        if not isinstance(expected, str) or not HEX64.fullmatch(expected):
            _fail(name + ' receipt hash invalid: ' + relative)
        member = _member(path)
        if expected != member['sha256']:
            _fail(name + ' receipt member changed: ' + relative)
        members[relative] = member
    members['.recipe-manifest.json'] = _member(receipt_path)
    return {'path': ROLE_PATHS[name], 'members': members}, receipt_raw


def _blob_member(blob, members, label):
    if not isinstance(blob, str) or not HEX64.fullmatch(blob):
        _fail(label + ' blob must be lowercase SHA256')
    name = 'blobs/%s.bin' % blob
    if name not in members:
        _fail(label + ' blob is missing or not receipt-owned')
    return members[name]


def _check_store(manifest_raw, members):
    manifest = _json(manifest_raw, 'store manifest')
    if not isinstance(manifest, dict):
        _fail('store manifest must be object')
    for key, expected in STORE_GEOMETRY.items():
        if manifest.get(key) != expected:
            _fail('store manifest %s disagrees with fixed geometry' % key)
    for key in ('layers', 'embedding', 'final_norm', 'towers'):
        if not isinstance(manifest.get(key), dict):
            _fail('store manifest %s must be object' % key)
    for key in ('embedding', 'final_norm'):
        _blob_member(manifest[key].get('blob'), members, key)
    expected_layers = {str(index) for index in range(48)}
    if set(manifest['layers']) != expected_layers:
        _fail('store manifest layers must be keys 0..47')
    full_layers = set(STORE_GEOMETRY['full_attention_layers'])
    for index in range(48):
        layer = manifest['layers'][str(index)]
        if not isinstance(layer, dict) or layer.get('attn_type') != ('full' if index in full_layers else 'sliding'):
            _fail('layer %d attention type invalid' % index)
        matrices = layer.get('matrices')
        if not isinstance(matrices, dict) or not matrices:
            _fail('layer %d matrices must be nonempty object' % index)
        found_blob = False
        def layer_blobs(value):
            nonlocal found_blob
            if isinstance(value, dict):
                for matrix_key, child in value.items():
                    if matrix_key == 'blob':
                        _blob_member(child, members, 'layer %d' % index)
                        found_blob = True
                    else:
                        layer_blobs(child)
            elif isinstance(value, list):
                for child in value:
                    layer_blobs(child)
        layer_blobs(matrices)
        if not found_blob:
            _fail('layer %d matrices lack blob reference' % index)
    towers = manifest['towers']
    if len(towers) != 11:
        _fail('store manifest must have eleven towers')
    for name, entry in towers.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            _fail('tower record invalid')
        if entry.get('layout') != 'raw_native' or entry.get('dtype') != 'float32':
            _fail(name + ' must be raw_native float32')
        blob = _blob_member(entry.get('blob'), members, name)
        offset, length, shape = entry.get('offset'), entry.get('length'), entry.get('shape')
        if not _is_int(offset) or offset < 0 or not _is_int(length) or length < 0:
            _fail(name + ' extent invalid')
        if not isinstance(shape, list) or any(not _is_int(axis) or axis < 0 for axis in shape):
            _fail(name + ' shape invalid')
        elements = 1
        for axis in shape:
            elements *= axis
        if elements * 4 != length or offset + length > blob['bytes']:
            _fail(name + ' extent disagrees with raw_native shape/blob')

    def bind_blobs(value, label):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'blob':
                    _blob_member(child, members, label)
                else:
                    bind_blobs(child, label)
        elif isinstance(value, list):
            for child in value:
                bind_blobs(child, label)
    for key in ('layers', 'embedding', 'final_norm'):
        bind_blobs(manifest[key], key)
    return manifest


def _config_role(root):
    root = _role_root(root, 'build-config')
    path = root / 'config.json'
    raw = _read(path, MAX_METADATA_BYTES, 'HF config')
    config = _json(raw, str(path))
    text = config.get('text_config') if isinstance(config, dict) else None
    if not isinstance(text, dict) or any(text.get(k) != v for k, v in GEOMETRY.items()):
        _fail('HF text_config geometry mismatch')
    return {'path': ROLE_PATHS['build-config'], 'members': {'config.json': _member(path)}}, raw


def _tokenizer_role(root, config_raw):
    root = _role_root(root, 'tokenizer-source')
    selected = {}
    for path in root.iterdir():
        if path.name == SIDECAR:
            _fail('reserved tokenizer sidecar is not an input member')
        if path.suffix in ('.json', '.jinja', '.model'):
            if path.is_symlink() or not path.is_file():
                _fail('tokenizer selected member is not regular: ' + path.name)
            selected[path.name] = path
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja'}
    if not required <= set(selected):
        _fail('tokenizer required files missing')
    if _read(selected['config.json'], MAX_METADATA_BYTES, 'tokenizer config') != config_raw:
        _fail('tokenizer config.json is not byte-equal to HF config.json')
    template = _read(selected['chat_template.jinja'], MAX_METADATA_BYTES, 'chat template').decode('utf-8')
    token_config = _json(_read(selected['tokenizer_config.json'], MAX_METADATA_BYTES, 'tokenizer config record'),
                         'tokenizer_config.json')
    if not isinstance(token_config, dict) or not isinstance(token_config.get('chat_template'), str) or not template or token_config['chat_template'] != template:
        _fail('tokenizer chat_template must exactly equal nonempty jinja')
    return {'path': ROLE_PATHS['tokenizer-source'],
            'members': {name: _member(path) for name, path in selected.items()}}


def _rf_role(root):
    role, unused = _receipt_role(root, 'retained-rf-stack')
    required = {'w%d.npy' % index for index in range(48)} | {'w_head.npy'}
    if not required <= set(role['members']):
        _fail('RF stack lacks w0..w47.npy or w_head.npy')
    for name in required:
        header = role['members'][name].get('header', {})
        if header.get('descr') != '|u1' or header.get('fortran_order') is not False:
            _fail('RF array format invalid: ' + name)
    if len(role['members']['w_head.npy']['header']['shape']) != 5:
        _fail('RF head must be 5D')
    for index in range(48):
        if len(role['members']['w%d.npy' % index]['header']['shape']) != 1:
            _fail('RF layers must be 1D')
    return role


def _source_roles(root):
    store, store_receipt = _receipt_role(root / 'store', 'retained-store')
    if 'manifest.json' not in store['members']:
        _fail('store manifest missing from receipt')
    manifest_raw = _read(root / 'store' / 'manifest.json', MAX_METADATA_BYTES, 'store manifest')
    _check_store(manifest_raw, store['members'])
    config, config_raw = _config_role(root / 'hf_config')
    return ({'retained-store': store, 'retained-rf-stack': _rf_role(root / 'rf_stack'),
             'build-config': config, 'tokenizer-source': _tokenizer_role(root / 'tokenizer', config_raw)},
            manifest_raw, store_receipt)


def _root(source_root, accepted_data_root):
    owner = declared_path(accepted_data_root, 'accepted data root')
    source = declared_path(source_root, 'Gemma4 source root')
    if not owner.is_dir() or source != owner / MODEL or not source.is_dir():
        _fail('source root must exactly be accepted_data_root/gemma4-12b')
    return source


def _legacy(value):
    return isinstance(value, (str, int, bool)) and not isinstance(value, float)


def build_witness(source_root, *, accepted_data_root, provenance, legacy):
    root = _root(source_root, accepted_data_root)
    if not isinstance(provenance, str) or not provenance:
        _fail('provenance must be nonempty string')
    if not isinstance(legacy, dict) or any(not isinstance(key, str) or not _legacy(value) for key, value in legacy.items()):
        _fail('legacy must be string-keyed historical scalars')
    roles, manifest, receipt = _source_roles(root)
    document = {'version': VERSION, 'kind': KIND, 'model': MODEL, 'variant': VARIANT,
                'created_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'origin': {'source_root': str(root), 'provenance': provenance, 'legacy': legacy},
                'roles': roles,
                'derivations': {'portable_store': {'scheme': 'portable-store-v1',
                                'source_manifest_json': manifest.decode('utf-8'),
                                'source_receipt_json': receipt.decode('utf-8')}},
                'payload_digest': canonical_digest(roles)}
    return _validate_witness(document)


def _validate_member(name, member):
    safe_member(name)
    if not isinstance(member, dict) or set(member) - {'sha256', 'bytes', 'header'}:
        _fail('invalid member fields: ' + name)
    if not isinstance(member.get('sha256'), str) or not HEX64.fullmatch(member['sha256']):
        _fail('invalid member SHA256: ' + name)
    if not _is_int(member.get('bytes')) or member['bytes'] < 0:
        _fail('invalid member bytes: ' + name)
    if 'header' in member:
        header = member['header']
        if not name.endswith('.npy'):
            _fail('non-npy member has header: ' + name)
        if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'} or not isinstance(header['descr'], str) or not isinstance(header['fortran_order'], bool) or not isinstance(header['shape'], list) or any(not _is_int(axis) or axis < 0 for axis in header['shape']):
            _fail('invalid numpy header: ' + name)


def _validate_witness(document):
    fields = {'version', 'kind', 'model', 'variant', 'created_utc', 'origin', 'roles', 'derivations', 'payload_digest'}
    if not isinstance(document, dict) or set(document) != fields:
        _fail('witness fields must be exact')
    if not _is_int(document['version']) or document['version'] != VERSION or document['kind'] != KIND or document['model'] != MODEL or document['variant'] != VARIANT:
        _fail('witness model/version/kind invalid')
    try:
        stamp = datetime.fromisoformat(document['created_utc'].replace('Z', '+00:00'))
    except (AttributeError, ValueError):
        _fail('invalid created_utc')
    if stamp.tzinfo is None or stamp.utcoffset() != timezone.utc.utcoffset(stamp):
        _fail('created_utc must be aware UTC')
    origin = document['origin']
    if not isinstance(origin, dict) or set(origin) != {'source_root', 'provenance', 'legacy'} or not isinstance(origin['provenance'], str) or not origin['provenance'] or not isinstance(origin['legacy'], dict):
        _fail('invalid origin')
    if not isinstance(origin.get('source_root'), str):
        _fail('origin source_root must be string')
    historical = Path(origin['source_root'])
    if not historical.is_absolute() or historical.name != MODEL or '.' in historical.parts or '..' in historical.parts:
        _fail('origin source_root is not lexical absolute Gemma4 root')
    if any(not isinstance(key, str) or not _legacy(value) for key, value in origin['legacy'].items()):
        _fail('invalid historical legacy values')
    roles = document['roles']
    if not isinstance(roles, dict) or set(roles) != set(ROLE_PATHS):
        _fail('invalid fixed roles')
    for role_name, fixed_path in ROLE_PATHS.items():
        role = roles[role_name]
        if not isinstance(role, dict) or set(role) != {'path', 'members'} or role['path'] != fixed_path or not isinstance(role['members'], dict) or not role['members']:
            _fail('invalid role ' + role_name)
        for name, member in role['members'].items():
            _validate_member(name, member)
    if set(roles['build-config']['members']) != {'config.json'}:
        _fail('build-config must contain config.json only')
    tokenizer = roles['tokenizer-source']['members']
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja'}
    if not required <= set(tokenizer) or any('/' in name or Path(name).suffix not in ('.json', '.jinja', '.model') for name in tokenizer):
        _fail('invalid tokenizer selected view')
    rf = roles['retained-rf-stack']['members']
    required_rf = {'w%d.npy' % index for index in range(48)} | {'w_head.npy'}
    if not required_rf <= set(rf):
        _fail('invalid RF role members')
    for name in required_rf:
        header = rf[name].get('header')
        if not isinstance(header, dict) or set(header) != {'descr', 'fortran_order', 'shape'} or header.get('descr') != '|u1' or header.get('fortran_order') is not False or not isinstance(header.get('shape'), list) or any(not _is_int(axis) or axis <= 0 for axis in header['shape']):
            _fail('invalid RF witness header: ' + name)
    if len(rf['w_head.npy']['header']['shape']) != 5 or any(len(rf['w%d.npy' % index]['header']['shape']) != 1 for index in range(48)):
        _fail('invalid RF witness shape')
    if document['payload_digest'] != canonical_digest(roles):
        _fail('payload_digest does not bind roles')
    derivation = document['derivations']
    if not isinstance(derivation, dict) or set(derivation) != {'portable_store'}:
        _fail('invalid derivations')
    portable = derivation['portable_store']
    if not isinstance(portable, dict) or set(portable) != {'scheme', 'source_manifest_json', 'source_receipt_json'} or portable['scheme'] != 'portable-store-v1' or not all(isinstance(portable.get(key), str) for key in ('source_manifest_json', 'source_receipt_json')):
        _fail('invalid portable store derivation')
    manifest_raw = portable['source_manifest_json'].encode('utf-8')
    receipt_raw = portable['source_receipt_json'].encode('utf-8')
    if len(manifest_raw) > MAX_METADATA_BYTES or len(receipt_raw) > MAX_METADATA_BYTES:
        _fail('embedded store metadata exceeds byte limit')
    store = roles['retained-store']['members']
    manifest_member = store.get('manifest.json', {})
    receipt_member = store.get('.recipe-manifest.json', {})
    if (manifest_member.get('sha256') != stream_hash_bytes(manifest_raw)
            or manifest_member.get('bytes') != len(manifest_raw)
            or receipt_member.get('sha256') != stream_hash_bytes(receipt_raw)
            or receipt_member.get('bytes') != len(receipt_raw)):
        _fail('embedded store metadata does not match role hashes')
    _check_store(manifest_raw, store)
    embedded_receipt = _json(receipt_raw, 'embedded store receipt')
    if not isinstance(embedded_receipt, dict) or not isinstance(embedded_receipt.get('files'), dict):
        _fail('embedded store receipt invalid')
    receipt_files = embedded_receipt['files']
    expected_files = set(store) - {'.recipe-manifest.json'}
    if set(receipt_files) != expected_files:
        _fail('embedded store receipt inventory differs from role')
    for name, digest in receipt_files.items():
        safe_member(name)
        if not isinstance(digest, str) or not HEX64.fullmatch(digest) or digest != store[name]['sha256']:
            _fail('embedded store receipt member digest invalid: ' + name)
    return document


def stream_hash_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def load_witness(path):
    return _validate_witness(_json(_read(Path(path), MAX_WITNESS_BYTES, 'witness'), str(path)))


def _resident_file_inventory(root, label):
    """Return every regular file below a declared resident data directory."""
    checked = declared_path(root, label)
    if not checked.is_dir():
        _fail(label + ' is not a directory')
    found = {}
    for path in checked.rglob('*'):
        if path.is_symlink():
            _fail('symlink in ' + label + ': ' + str(path))
        if path.is_file():
            name = path.relative_to(checked).as_posix()
            safe_member(name)
            found[name] = path
        elif not path.is_dir():
            _fail('non-regular member in ' + label + ': ' + str(path))
    return found


def _resident_members(root, expected, label):
    actual = _resident_file_inventory(root, label)
    if set(actual) != set(expected):
        _fail(label + ' inventory differs from witness')
    members = {}
    for name, path in actual.items():
        member = _member(path)
        wanted = expected[name]
        if member['sha256'] != wanted['sha256'] or member['bytes'] != wanted['bytes']:
            _fail(label + ' member differs from witness: ' + name)
        if 'header' in wanted and member.get('header') != wanted['header']:
            _fail(label + ' member header differs from witness: ' + name)
        members[name] = member
    return members


def _resident_rf_receipt(root, members):
    receipt_path = declared_path(root / '.recipe-manifest.json', 'resident RF receipt')
    receipt = _json(_read(receipt_path, MAX_METADATA_BYTES, 'resident RF receipt'), str(receipt_path))
    files = receipt.get('files') if isinstance(receipt, dict) else None
    expected_names = set(members) - {'.recipe-manifest.json'}
    if not isinstance(files, dict) or set(files) != expected_names:
        _fail('resident RF receipt inventory differs from actual files')
    for name in expected_names:
        digest = files[name]
        if not isinstance(digest, str) or not HEX64.fullmatch(digest) or digest != members[name]['sha256']:
            _fail('resident RF receipt member changed: ' + name)


def _resident_store(root, source_members, portable):
    actual = _resident_file_inventory(root, 'resident store')
    if set(actual) != set(source_members):
        _fail('resident store inventory differs from witness')
    members = {name: _member(path) for name, path in actual.items()}
    manifest_path = actual.get('manifest.json')
    receipt_path = actual.get('.recipe-manifest.json')
    if manifest_path is None or receipt_path is None:
        _fail('resident store metadata is missing')
    manifest_raw = _read(manifest_path, MAX_METADATA_BYTES, 'resident store manifest')
    receipt_raw = _read(receipt_path, MAX_METADATA_BYTES, 'resident store receipt')
    manifest = _json(manifest_raw, str(manifest_path))
    receipt = _json(receipt_raw, str(receipt_path))
    if not isinstance(manifest, dict) or not isinstance(receipt, dict) or not isinstance(receipt.get('files'), dict):
        _fail('resident store metadata is invalid')

    source_manifest_raw = portable['source_manifest_json'].encode('utf-8')
    source_receipt_raw = portable['source_receipt_json'].encode('utf-8')
    source_manifest = _json(source_manifest_raw, 'embedded source store manifest')
    source_receipt = _json(source_receipt_raw, 'embedded source store receipt')
    if not isinstance(source_manifest, dict) or not isinstance(source_receipt, dict) or not isinstance(source_receipt.get('files'), dict):
        _fail('embedded source store metadata is invalid')
    source_files = source_receipt['files']
    if set(source_files) != set(source_members) - {'.recipe-manifest.json'}:
        _fail('embedded source receipt inventory differs from witness')

    if 'source_dir' in source_manifest:
        original_source_dir = source_manifest['source_dir']
        if not isinstance(original_source_dir, str):
            _fail('embedded source_dir is invalid')
        expected_manifest = dict(source_manifest)
        del expected_manifest['source_dir']
        if not _strict_json_equal(manifest, expected_manifest):
            _fail('derived resident manifest differs from source manifest')
        expected_files = dict(source_files)
        expected_files['manifest.json'] = stream_hash_bytes(manifest_raw)
        expected_derivation = {
            'version': 1,
            'source_manifest_sha256': source_members['manifest.json']['sha256'],
            'source_receipt_sha256': source_members['.recipe-manifest.json']['sha256'],
            'removed_fields': ['source_dir'],
            'source_dir_sha256': stream_hash_bytes(original_source_dir.encode('utf-8')),
        }
        expected_receipt = {
            'command': 'package_artifact.py --derive-store (portable-store-v1)',
            'derivation': expected_derivation,
            'files': expected_files,
        }
        if not _strict_json_equal(receipt, expected_receipt):
            _fail('derived resident store receipt differs from portable-store-v1')
    else:
        if manifest_raw != source_manifest_raw or receipt_raw != source_receipt_raw:
            _fail('portable resident store metadata differs from witness')
        expected_files = source_files

    if set(receipt['files']) != set(actual) - {'.recipe-manifest.json'}:
        _fail('resident store receipt inventory differs from actual files')
    for name, member in members.items():
        if name == '.recipe-manifest.json':
            continue
        if receipt['files'].get(name) != member['sha256']:
            _fail('resident store receipt member changed: ' + name)
        if name != 'manifest.json':
            wanted = source_members[name]
            if member['sha256'] != wanted['sha256'] or member['bytes'] != wanted['bytes']:
                _fail('resident store member differs from witness: ' + name)
            if 'header' in wanted and member.get('header') != wanted['header']:
                _fail('resident store member header differs from witness: ' + name)


def verify_resident_data(resident_root, *, expected_witness_sha256: str):
    """Verify only resident Gemma4 data files, not native code, proofs, or model numerics."""
    if not isinstance(expected_witness_sha256, str) or not HEX64.fullmatch(expected_witness_sha256):
        _fail('expected witness SHA256 must be lowercase SHA256')
    root = declared_path(resident_root, 'resident root')
    if not root.is_dir():
        _fail('resident root is not a directory')
    sidecar = declared_path(root / SIDECAR, 'resident witness sidecar')
    raw = _read(sidecar, MAX_WITNESS_BYTES, 'resident witness sidecar')
    if stream_hash_bytes(raw) != expected_witness_sha256:
        _fail('resident witness sidecar SHA256 differs from expected value')
    witness = _validate_witness(_json(raw, str(sidecar)))
    roles = witness['roles']
    weights = _resident_members(root / 'weights', roles['retained-rf-stack']['members'], 'resident weights')
    _resident_rf_receipt(root / 'weights', weights)
    head = declared_path(root / 'w_head.npy', 'resident w_head')
    head_member = _member(head)
    expected_head = roles['retained-rf-stack']['members']['w_head.npy']
    if head_member != expected_head:
        _fail('resident w_head differs from RF witness member')
    _resident_store(root / 'store', roles['retained-store']['members'], witness['derivations']['portable_store'])
    return witness


def _published_roots(resident_root, tokenizer_root, checkpoint_root):
    roots = tuple(declared_path(path, label) for path, label in (
        (resident_root, 'resident root'), (tokenizer_root, 'published tokenizer root'),
        (checkpoint_root, 'published checkpoint root')))
    if any(not root.is_dir() for root in roots):
        _fail('published roots must be directories')
    for index, left in enumerate(roots):
        for right in roots[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                _fail('published roots must be pairwise disjoint')
    return roots


def _published_members(root, expected, label, extra_members=()):
    actual = _resident_file_inventory(root, label)
    if set(actual) != set(expected) | set(extra_members):
        _fail(label + ' inventory differs from witness')
    members = {}
    for name, wanted in expected.items():
        member = _member(actual[name])
        if member != wanted:
            _fail(label + ' member differs from witness: ' + name)
        members[name] = member
    return actual, members


def _published_config(raw, label):
    config = _json(raw, label)
    text = config.get('text_config') if isinstance(config, dict) else None
    if not isinstance(text, dict) or any(text.get(key) != value for key, value in GEOMETRY.items()):
        _fail(label + ' geometry mismatch')


def _checkpoint_header(path):
    if path.is_symlink() or not path.is_file():
        _fail('checkpoint model.safetensors is not a regular file')
    size = path.stat().st_size
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            _fail('checkpoint safetensors header is truncated')
        header_size = int.from_bytes(prefix, 'little')
        if header_size > MAX_METADATA_BYTES or header_size > size - 8:
            _fail('checkpoint safetensors header exceeds limit or file')
        raw = handle.read(header_size)
    if len(raw) != header_size:
        _fail('checkpoint safetensors header is truncated')
    header = _json(raw, 'checkpoint safetensors header')
    if not isinstance(header, dict):
        _fail('checkpoint safetensors header must be object')
    return header, 8 + header_size, size


def _verify_checkpoint_towers(resident_root, checkpoint, witness):
    header, payload_start, size = _checkpoint_header(checkpoint / 'model.safetensors')
    manifest = _json(witness['derivations']['portable_store']['source_manifest_json'].encode(),
                     'embedded store manifest')
    towers = manifest['towers']
    if set(header) != set(towers):
        _fail('checkpoint tower names differ from store manifest')
    regions = []
    for name, source in towers.items():
        entry = header[name]
        if not isinstance(entry, dict) or set(entry) != {'dtype', 'shape', 'data_offsets'}:
            _fail(name + ' checkpoint tensor metadata invalid')
        shape = entry['shape']
        if (not isinstance(shape, list)
                or any(not _is_int(axis) or axis < 0 for axis in shape)):
            _fail(name + ' checkpoint tensor shape invalid')
        if entry['dtype'] != 'F32' or shape != source['shape']:
            _fail(name + ' checkpoint tensor dtype or shape differs')
        offsets = entry['data_offsets']
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(not _is_int(value) or value < 0 for value in offsets)):
            _fail(name + ' checkpoint offsets invalid')
        start, end = offsets
        if end - start != source['length']:
            _fail(name + ' checkpoint extent differs from tower')
        regions.append((start, end, name, source))
    regions.sort()
    cursor = 0
    for start, end, name, source in regions:
        if start != cursor:
            _fail('checkpoint payload has overlap or gap')
        cursor = end
        blob = source['blob']
        blob_path = declared_path(resident_root / 'store' / 'blobs' / (blob + '.bin'),
                                  name + ' resident tower blob')
        if not blob_path.is_file() or blob_path.is_symlink():
            _fail(name + ' resident tower blob is not regular')
        if source['offset'] + source['length'] > blob_path.stat().st_size:
            _fail(name + ' resident tower source extent invalid')
        # The checkpoint payload and raw-native source have the same byte layout for F32.
        with (checkpoint / 'model.safetensors').open('rb') as output:
            output.seek(payload_start + start)
            digest = hashlib.sha256()
            remaining = end - start
            while remaining:
                chunk = output.read(min(1024 * 1024, remaining))
                if not chunk:
                    _fail(name + ' checkpoint payload is truncated')
                digest.update(chunk)
                remaining -= len(chunk)
        with blob_path.open('rb') as input_file:
            input_file.seek(source['offset'])
            source_digest = hashlib.sha256()
            remaining = source['length']
            while remaining:
                chunk = input_file.read(min(1024 * 1024, remaining))
                if not chunk:
                    _fail(name + ' resident tower source is truncated')
                source_digest.update(chunk)
                remaining -= len(chunk)
        if digest.digest() != source_digest.digest():
            _fail(name + ' checkpoint payload differs from resident tower bytes')
    if size != payload_start + cursor:
        _fail('checkpoint safetensors has trailing or unbound payload')


def verify_published_data(resident_root, tokenizer_root, checkpoint_root, *, expected_witness_sha256):
    """Verify published tokenizer/checkpoint bytes against one resident witness snapshot only."""
    resident, tokenizer, checkpoint = _published_roots(resident_root, tokenizer_root, checkpoint_root)
    witness = verify_resident_data(resident, expected_witness_sha256=expected_witness_sha256)
    token_members = witness['roles']['tokenizer-source']['members']
    token_paths, token_actual_members = _published_members(tokenizer, token_members, 'published tokenizer')
    checkpoint_paths, checkpoint_actual_members = _published_members(
        checkpoint, token_members, 'published checkpoint', ('model.safetensors',))
    config_member = witness['roles']['build-config']['members']['config.json']
    for label, paths, actual_members in (
            ('published tokenizer config', token_paths, token_actual_members),
            ('published checkpoint config', checkpoint_paths, checkpoint_actual_members)):
        raw = _read(paths['config.json'], MAX_METADATA_BYTES, label)
        if actual_members['config.json'] != config_member:
            _fail(label + ' differs from build-config witness member')
        _published_config(raw, label)
    template = _read(token_paths['chat_template.jinja'], MAX_METADATA_BYTES, 'published chat template').decode('utf-8')
    token_config = _json(_read(token_paths['tokenizer_config.json'], MAX_METADATA_BYTES,
                               'published tokenizer config'), 'published tokenizer config')
    if not template or not isinstance(token_config, dict) or token_config.get('chat_template') != template:
        _fail('published tokenizer chat_template must exactly equal nonempty jinja')
    _verify_checkpoint_towers(resident, checkpoint, witness)
    return witness


def verify_source(source_root, witness, *, accepted_data_root):
    document = _validate_witness(witness)
    root = _root(source_root, accepted_data_root)
    if document['origin']['source_root'] != str(root):
        _fail('witness origin source_root differs from explicit root')
    roles, manifest, receipt = _source_roles(root)
    if roles != document['roles']:
        _fail('source members do not match witness')
    portable = document['derivations']['portable_store']
    if portable['source_manifest_json'].encode('utf-8') != manifest or portable['source_receipt_json'].encode('utf-8') != receipt:
        _fail('embedded store metadata does not match source')


def _proof_root(path, source, witness):
    proof = declared_path(path, 'proof root')
    if proof == witness:
        _fail('proof root must not be the input witness file')
    if proof == source or proof in source.parents or source in proof.parents:
        _fail('proof root must be disjoint from the entire source root')
    if proof.exists():
        if not proof.is_dir():
            _fail('proof root is not a directory')
        if any(proof.iterdir()):
            _fail('proof root must be absent or empty')
    return proof


def _write_proof_sidecar(proof, raw):
    """Atomically publish only bytes already pinned and checked by the caller."""
    proof = declared_path(proof, 'proof root')
    if not proof.is_dir() or any(proof.iterdir()):
        _fail('proof root changed before sidecar publication')
    target = declared_path(proof / SIDECAR, 'proof sidecar')
    if target.exists():
        _fail('proof sidecar already exists')
    temporary = proof / ('.' + SIDECAR + '.' + secrets.token_hex(16) + '.tmp')
    created = False
    temporary_identity = None
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        temporary_stat = os.fstat(descriptor)
        temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
        created = True
        with os.fdopen(descriptor, 'wb') as handle:
            descriptor = None
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        # Re-declare both parents and target immediately before replacement: this is a freshness
        # check, not a claim to provide a generic filesystem race sandbox.
        proof = declared_path(proof, 'proof root')
        target = declared_path(proof / SIDECAR, 'proof sidecar')
        if not proof.is_dir() or target.exists():
            _fail('proof target changed before sidecar publication')
        os.replace(temporary, target)
        created = False
        directory = os.open(proof, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        _fail('cannot write proof sidecar: %s' % error)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                current = os.lstat(temporary)
                if (current.st_dev, current.st_ino) == temporary_identity:
                    os.unlink(temporary)
            except FileNotFoundError:
                pass


def materialize_proof(source_root, witness_path, proof_root, *, accepted_data_root,
                      expected_witness_sha256) -> dict:
    """Verify one pinned source witness and atomically materialize its exact sidecar bytes.

    This makes no model payload copy, link, package, or native admission claim.
    """
    source = _root(source_root, accepted_data_root)
    witness = declared_path(witness_path, 'input witness')
    proof = _proof_root(proof_root, source, witness)
    if not isinstance(expected_witness_sha256, str) or not HEX64.fullmatch(expected_witness_sha256):
        _fail('expected witness SHA256 must be lowercase SHA256')
    raw = _read(witness, MAX_WITNESS_BYTES, 'input witness')
    digest = stream_hash_bytes(raw)
    if digest != expected_witness_sha256:
        _fail('input witness SHA256 differs from expected value')
    document = _validate_witness(_json(raw, str(witness)))
    verify_source(source, document, accepted_data_root=accepted_data_root)
    # No directory is made until all input paths, bytes, document fields, and source members pass.
    if not proof.exists():
        try:
            proof.mkdir(parents=True)
        except OSError as error:
            _fail('cannot create proof root: %s' % error)
    _write_proof_sidecar(proof, raw)
    filecount = sum(len(role['members']) for role in document['roles'].values())
    return {'model': document['model'], 'variant': document['variant'],
            'witness_sha256': digest, 'payload_digest': document['payload_digest'],
            'members': filecount, 'filecount': filecount,
            'proofpath': str(proof / SIDECAR)}
