# SPDX-License-Identifier: Apache-2.0
"""Versioned witness for compiling against retained verified model data.

A `verified-data` recipe variant consumes bytes that already exist and were already reviewed. This
module is the single contract for those bytes: it builds and checks a witness that binds the model,
the dump's own format records, every payload member's hash, and the honest historical origin. It
never runs a quantizer, never loads a tensor, and imports only the standard library so the prepared
installer can verify a published closure without the build SDK.

Three deliberate limits:
- a witness is an operator-reviewed declaration, not new provenance. Its origin fields are historical
  strings for operators; nothing here may dereference them at verification time, so installed
  verification does not depend on a retired legacy tree;
- the source path comes from the operator's command line, confined to an explicitly accepted data
  owner directory. A witness cannot select its own input root, output root or code path;
- `.npy` files are read as headers only. A dtype/order/rank check is what a data consumer needs;
  loading model values here would defeat the no-dense-load rule and slow every check.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

KIND = 'xdna-verified-data-witness'
VERSION = 1
VARIANT = 'verified-data'
# The published closure carries its own witness so installed/reinstall verification never has to
# follow a path back into the preparation directory or a retired legacy tree.
SIDECAR = '.verified-data-witness.json'
QUANT_RECORD = 'quant.json'
PROGRESS_RECORD = '_dump_progress.json'
MAX_WITNESS_BYTES = 8 << 20
MAX_RECORD_BYTES = 1 << 20
NPY_MAGIC = b'\x93NUMPY'
# numpy pads a v1 header into the first 64-byte aligned block; anything larger is not a dump header.
NPY_HEADER_LIMIT = 4096
# The versions the grammar below reads: a 6-byte magic, a 2-byte version, then a uint16 (v1) or
# uint32 (v2/v3) header length, so the header body starts at byte 10 or 12 -- not at 14.
NPY_VERSIONS = (1, 2, 3)
# The layouts scripts/dump_llm_weights.py can actually write (its --quant-layout choices).
LAYOUTS = ('header_first', 'row_group_planar')
SCALE_DTYPES = ('f32', 'f16')
SHA256_HEX = 64
ORIGIN_KEYS = ('source_root', 'provenance', 'legacy')


class VerifiedDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class DataSpec:
    """The fixed format a model's served consumer reads; the variant has no other mode."""
    model: str
    quant: dict
    leaves: str
    cols: int
    packed_descr: str
    plain_descr: str
    # The backbone depth the served decode compiles. A dump is only usable whole: `_dump_progress.json`
    # records `layers: null` for a full dump, and any explicit number must already be this one.
    n_layers: int
    sidecar: str = SIDECAR


GEMMA3_270M = DataSpec(
    model='gemma3-270m',
    # gen_llm_decode reads quant.json itself, and the served Gemma3 decode root name carries this grid.
    quant={'dtype': 'int4', 'group_size': 32, 'layout': 'header_first', 'scale_dtype': 'f32',
           'full_range': True, 'clip_search': False},
    leaves='gate_proj,up_proj,down_proj,o_proj,embed_tokens', cols=8, packed_descr='|i1', plain_descr='<f4',
    n_layers=18)   # designs/decode_fused/llm_decode_spec.py::GEMMA3_270M n_layers

DATA_SPECS = {spec.model: spec for spec in (GEMMA3_270M,)}
# Named so an operator asking for the second case gets a refusal instead of a silent fallback to the
# source-quantizing recipe. Adding a Gemma4 spec is a separate, priced change.
UNIMPLEMENTED = ('gemma4-12b',)


def spec_for(model: str) -> DataSpec:
    if model in DATA_SPECS:
        return DATA_SPECS[model]
    if model in UNIMPLEMENTED:
        raise VerifiedDataError(f'{model}: the {VARIANT} variant is not implemented yet; its declared '
                                'prepared-data inputs and served roots still need their own witness '
                                'spec, and falling back to the source-quantizing recipe is not equivalent')
    raise VerifiedDataError(f'{model}: the {VARIANT} variant is Gemma-only')


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _read_limited(path: Path, limit: int, what: str) -> bytes:
    if path.is_symlink():
        raise VerifiedDataError(f'symlinked {what}: {path}')
    size = path.stat().st_size
    if size > limit:
        raise VerifiedDataError(f'{what} exceeds the {limit} byte limit: {path} ({size})')
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise VerifiedDataError(f'{what} exceeds the {limit} byte limit: {path}')
    return raw


def reject_aliased_ancestors(path: Path, label: str) -> None:
    """Refuse a path that reaches its parent through a link, before anything is read or created."""
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise VerifiedDataError(f'symlink in the {label}: {parent}')


def declared_path(raw, label: str) -> Path:
    """A canonical absolute path from a root claim: no `.`/`..` component, no aliased ancestor.

    `Path.resolve()` cannot do this job, because it erases `..` and follows links before anything can
    object; the returned path needs no resolution because the walk proved it has no link to follow.
    """
    path = Path(raw)
    if not path.is_absolute():
        raise VerifiedDataError(f'{label} must be an absolute path: {path}')
    if '..' in path.parts or '.' in path.parts:
        raise VerifiedDataError(f'{label} may not contain . or .. path components: {path}')
    reject_aliased_ancestors(path, f'{label} path')
    return Path(os.path.abspath(path))


def _bounded_bytes(path: Path, limit: int, what: str) -> bytes:
    try:
        return _read_limited(path, limit, what)
    except OSError as error:
        raise VerifiedDataError(f'cannot read {what} {path}: {error}') from error


def _parse_json(raw: bytes, path: Path, what: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise VerifiedDataError(f'duplicate JSON field {key} in {path}')
            result[key] = value
        return result
    try:
        return json.loads(raw.decode(), object_pairs_hook=pairs)
    except (ValueError, UnicodeDecodeError) as error:
        raise VerifiedDataError(f'cannot read {what} {path}: {error}') from error


def _load_json(path: Path, limit: int, what: str):
    return _parse_json(_bounded_bytes(path, limit, what), path, what)


def stream_hash(path: Path) -> tuple[str, int]:
    if path.is_symlink():
        raise VerifiedDataError(f'symlinked data member: {path}')
    if not path.is_file():
        raise VerifiedDataError(f'data member is not a regular file: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _itemsize(descr: str) -> int:
    match = re.fullmatch(r'[<>|=]?([iufb?])(\d+)', descr)
    if not match:
        raise VerifiedDataError(f'unsupported numpy dtype descriptor: {descr!r}')
    return int(match.group(2))


def read_npy_header(path: Path) -> dict:
    """Header fields only: version, the literal header and the declared byte count.

    `ast.literal_eval` cannot execute anything and the read is bounded by NPY_HEADER_LIMIT, so a
    hostile header cannot make this stream the file. Array values are never loaded; the file size is
    compared against the declared shape, which is what catches a truncated or padded dump.
    """
    with path.open('rb') as handle:
        head = handle.read(8)
        if len(head) < 8 or not head.startswith(NPY_MAGIC):
            raise VerifiedDataError(f'not a numpy v1/v2/v3 array: {path}')
        major, minor = head[len(NPY_MAGIC)], head[len(NPY_MAGIC) + 1]
        if major not in NPY_VERSIONS:
            raise VerifiedDataError(f'unsupported numpy magic version {major}: {path}')
        if minor != 0:
            raise VerifiedDataError(f'unsupported numpy header minor version {major}.{minor}: {path}')
        count = 2 if major == 1 else 4
        length_bytes = handle.read(count)
        if len(length_bytes) != count:
            raise VerifiedDataError(f'truncated numpy header: {path}')
        length = struct.unpack('<H' if major == 1 else '<I', length_bytes)[0]
        offset = 8 + count + length  # where the declared array values start
        if length > NPY_HEADER_LIMIT:
            raise VerifiedDataError(f'numpy header declares an oversized header ({length}): {path}')
        raw = handle.read(length)
    if len(raw) != length:
        raise VerifiedDataError(f'truncated numpy header: {path}')
    try:
        # Only v3 requires the header text itself to be UTF-8; v1/v2 headers are latin-1 bytes.
        header = ast.literal_eval(raw.decode('utf-8' if major == 3 else 'latin-1').strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise VerifiedDataError(f'unparseable numpy header in {path}: {error}') from error
    if not isinstance(header, dict) or not {'descr', 'fortran_order', 'shape'} <= set(header):
        raise VerifiedDataError(f'numpy header lacks descr/fortran_order/shape: {path}')
    if not isinstance(header['shape'], tuple) or not all(isinstance(axis, int) and axis >= 0
                                                         for axis in header['shape']):
        raise VerifiedDataError(f'numpy header shape is not a tuple of integers: {path}')
    declared = _itemsize(str(header['descr']))
    for axis in header['shape']:
        declared *= axis
    if path.stat().st_size - offset != declared:
        raise VerifiedDataError(f'{path.name}: declares {header["shape"]} {header["descr"]} '
                                f'({declared} bytes) but holds {path.stat().st_size - offset}')
    return {'descr': str(header['descr']), 'fortran_order': bool(header['fortran_order']),
            'shape': list(header['shape'])}


def safe_member(name: str) -> str:
    member = Path(name)
    if not name or member.is_absolute() or '..' in member.parts or '.' in member.parts \
            or any(not part for part in member.parts) or name != member.as_posix():
        raise VerifiedDataError(f'unsafe witness member name: {name!r}')
    if member.name == SIDECAR:
        raise VerifiedDataError(f'the published witness sidecar is not a payload member: {name!r}')
    return name


def _confine(source: Path, accepted: Path) -> Path:
    owner, wanted = declared_path(accepted, 'accepted data root'), declared_path(source, 'data source')
    if not owner.is_dir():
        raise VerifiedDataError(f'accepted data root is not a directory: {owner}')
    if wanted != owner and owner not in wanted.parents:
        raise VerifiedDataError(f'data source escapes the accepted data root {owner}: {wanted}')
    if not wanted.is_dir():
        raise VerifiedDataError(f'data source is not a directory: {wanted}')
    return wanted


def _records(root: Path):
    quant = _load_json(root / QUANT_RECORD, MAX_RECORD_BYTES, 'quantization record')
    progress = _load_json(root / PROGRESS_RECORD, MAX_RECORD_BYTES, 'dump progress record')
    if not isinstance(quant, dict) or not isinstance(progress, dict):
        raise VerifiedDataError(f'{root}: {QUANT_RECORD}/{PROGRESS_RECORD} must be JSON objects')
    return quant, progress


def check_format(root: Path, spec: DataSpec, *, members: set[str] | None = None) -> set[str]:
    """Schema/layout check against the dump's own records; returns the packed tensor stems."""
    quant, progress = _records(root)
    if 'note' not in quant:
        raise VerifiedDataError(f'{root}: {QUANT_RECORD} does not declare how packed arrays are stored')
    for axis, expected in spec.quant.items():
        if quant.get(axis) != expected:
            raise VerifiedDataError(f'{spec.model}: {QUANT_RECORD} {axis}={quant.get(axis)!r} '
                                    f'disagrees with the {VARIANT} spec {expected!r}')
    if quant.get('layout') not in LAYOUTS or quant.get('scale_dtype') not in SCALE_DTYPES:
        raise VerifiedDataError(f'{spec.model}: unsupported layout/scale in {QUANT_RECORD}')
    packed = quant.get('packed')
    if not isinstance(packed, list) or not all(isinstance(name, str) for name in packed) \
            or len(packed) != len(set(packed)):
        raise VerifiedDataError(f'{spec.model}: {QUANT_RECORD}.packed must be a list of unique names')
    fmt = progress.get('fmt')
    if not isinstance(fmt, str):
        raise VerifiedDataError(f'{spec.model}: {PROGRESS_RECORD}.fmt must record the dump grid')
    try:
        grid = json.loads(fmt)
    except ValueError as error:
        raise VerifiedDataError(f'{spec.model}: {PROGRESS_RECORD}.fmt is not JSON: {error}') from error
    axes = {'quant': spec.quant['dtype'], 'quant_group': spec.quant['group_size'],
            'quant_layout': spec.quant['layout'], 'quant_scale_dtype': spec.quant['scale_dtype'],
            'quant_full_range': spec.quant['full_range'], 'quant_clip_search': spec.quant['clip_search'],
            'quant_leaves': spec.leaves, 'cols': spec.cols}
    for axis, expected in axes.items():
        if grid.get(axis) != expected:
            raise VerifiedDataError(f'{spec.model}: {PROGRESS_RECORD}.fmt {axis}={grid.get(axis)!r} '
                                    f'disagrees with the {VARIANT} spec {expected!r}')
    # `layers` is the one recorded axis that may be null, because the generator writes null when the
    # dump covers the whole backbone. A number is a claim about a partial dump, and a partial dump is
    # not the model the served decode reads.
    if 'layers' not in grid or grid['layers'] not in (None, spec.n_layers):
        raise VerifiedDataError(f'{spec.model}: {PROGRESS_RECORD}.fmt layers={grid.get("layers")!r} '
                                f'describes a partial dump; the {VARIANT} variant serves all '
                                f'{spec.n_layers} layers (or records null for "all")')
    done = progress.get('done')
    if not isinstance(done, dict) or not done:
        raise VerifiedDataError(f'{spec.model}: {PROGRESS_RECORD}.done must record the dumped tensors')
    derived: set[str] = set()
    for name, extras in done.items():
        derived.add(f'{safe_member(name)}.npy')
        if not isinstance(extras, list) or not all(isinstance(extra, str) for extra in extras):
            raise VerifiedDataError(f'{spec.model}: {PROGRESS_RECORD}.done[{name}] '
                                    'must list the companion arrays')
        for extra in extras:
            derived.add(f'{safe_member(extra)}.npy')
    if members is None:
        members = {safe_member(path.name) for path in root.iterdir() if path.suffix == '.npy'}
    if members != derived:
        missing, extra = sorted(derived - members), sorted(members - derived)
        raise VerifiedDataError(f'{spec.model}: dump layout disagrees: missing={missing[:4]}'
                                f'{"..." if len(missing) > 4 else ""} extra={extra[:4]}'
                                f'{"..." if len(extra) > 4 else ""}')
    declared = {extra for extras in done.values() for extra in extras}
    if set(packed) != declared:
        raise VerifiedDataError(f'{spec.model}: {QUANT_RECORD}.packed disagrees with '
                                f'{PROGRESS_RECORD}.done: {sorted(set(packed) ^ declared)[:4]}')
    return set(packed)


def check_layout(root: Path, spec: DataSpec, packed: set[str]) -> None:
    """Every dumped array is declared by dtype/order/rank: int8 on-wire bytes or plain values."""
    for path in sorted(root.iterdir()):
        if path.suffix != '.npy':
            continue
        if path.is_symlink():
            raise VerifiedDataError(f'symlinked dumped array: {path}')
        header = read_npy_header(path)
        is_packed = path.name[:-len('.npy')] in packed
        expected = spec.packed_descr if is_packed else spec.plain_descr
        if header['descr'] != expected or header['fortran_order']:
            kind = 'packed' if is_packed else 'unpacked'
            raise VerifiedDataError(f'{path.name}: {kind} array is {header["descr"]} '
                                    f'fortran_order={header["fortran_order"]}, expected '
                                    f'{expected} fortran_order=False')
        if is_packed and len(header['shape']) != 1:
            raise VerifiedDataError(f'{path.name}: packed array has rank {len(header["shape"])}, expected 1')


def _validate_document(document):
    if not isinstance(document, dict):
        raise VerifiedDataError('data witness is not a JSON object')
    fields = {'version', 'kind', 'model', 'variant', 'created_utc', 'origin', 'format', 'members',
              'payload_digest'}
    if set(document) != fields:
        raise VerifiedDataError(f'data witness fields must be {sorted(fields)}: {sorted(document)}')
    if document['version'] != VERSION or document['kind'] != KIND:
        raise VerifiedDataError(f'unsupported data witness version/kind: '
                                f'{document.get("version")}/{document.get("kind")}')
    if not isinstance(document['model'], str) or not document['model']:
        raise VerifiedDataError('data witness needs its model')
    if document['variant'] != VARIANT:
        raise VerifiedDataError(f'data witness variant {document.get("variant")!r} is not {VARIANT!r}')
    if not isinstance(document['created_utc'], str) or not document['created_utc']:
        raise VerifiedDataError('data witness needs a creation timestamp')
    origin = document['origin']
    if not isinstance(origin, dict) or set(origin) != set(ORIGIN_KEYS) \
            or not all(isinstance(origin[name], str) and origin[name] for name in ('source_root', 'provenance')) \
            or not isinstance(origin['legacy'], dict) \
            or not all(isinstance(key, str) and isinstance(value, (str, int, bool))
                       for key, value in origin['legacy'].items()):
        raise VerifiedDataError(f'data witness origin needs {list(ORIGIN_KEYS)} with non-empty '
                                'source_root/provenance and a string-keyed legacy field set')
    if not isinstance(document['format'], dict) or not document['format']:
        raise VerifiedDataError('data witness needs the declared format axes')
    members = document['members']
    if not isinstance(members, dict) or not members:
        raise VerifiedDataError('data witness needs at least one payload member')
    for name, entry in members.items():
        safe_member(name)
        if not isinstance(entry, dict) or not {'sha256', 'bytes'} <= set(entry) \
                or set(entry) - {'sha256', 'bytes', 'header'}:
            raise VerifiedDataError(f'data witness member {name} needs sha256/bytes and optional header')
        if not isinstance(entry['sha256'], str) or len(entry['sha256']) != SHA256_HEX \
                or any(char not in '0123456789abcdef' for char in entry['sha256']):
            raise VerifiedDataError(f'data witness member {name} needs a lowercase sha256 hex digest')
        if not isinstance(entry['bytes'], int) or isinstance(entry['bytes'], bool) or entry['bytes'] < 0:
            raise VerifiedDataError(f'data witness member {name} needs a non-negative byte count')
        if entry.get('header') is not None and not isinstance(entry['header'], dict):
            raise VerifiedDataError(f'data witness member {name} has a non-object header')
    if document['payload_digest'] != canonical_digest(members):
        raise VerifiedDataError('data witness payload_digest does not bind its members')
    return document


def load_witness(path: Path):
    return _validate_document(_load_json(Path(path), MAX_WITNESS_BYTES, 'data witness'))


def build_witness(source: Path, *, model: str, provenance: str, legacy: dict) -> dict:
    """Operator-reviewed witness over an existing dump. This declares bytes; it does not produce them."""
    spec = spec_for(model)
    # Reading a dump in order to witness it is a root claim too: the same grammar and alias rules apply
    # before any byte is read, so `--build-witness` cannot be pointed at a tree by a path that later
    # collapses onto something else.
    source = _confine(Path(source), Path(source))
    if not provenance:
        raise VerifiedDataError('a data witness needs a provenance statement naming the real origin')
    quant, _progress = _records(source)
    check_layout(source, spec, check_format(source, spec))
    members = {}
    for path in sorted(source.iterdir()):
        if path.name == spec.sidecar:
            continue
        safe_member(path.name)
        if path.is_symlink() or not path.is_file():
            raise VerifiedDataError(f'witness source contains a non-regular member: {path}')
        sha, size = stream_hash(path)
        entry = {'sha256': sha, 'bytes': size}
        if path.suffix == '.npy':
            entry['header'] = read_npy_header(path)
        members[path.name] = entry
    if not any(name.endswith('.npy') for name in members):
        raise VerifiedDataError(f'{model}: a data witness needs dumped arrays, not only metadata')
    document = {'version': VERSION, 'kind': KIND, 'model': spec.model, 'variant': VARIANT,
                'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'origin': {'source_root': str(source.resolve()), 'provenance': provenance,
                           'legacy': dict(legacy)},
                'format': {axis: quant[axis] for axis in spec.quant},
                'members': members, 'payload_digest': ''}
    document['payload_digest'] = canonical_digest(document['members'])
    return _validate_document(document)


def check_payload(root: Path, witness, *, spec: DataSpec, label: str) -> None:
    """Verify payload bytes against the witness. Needs no access to the original source root."""
    root = Path(root)
    if witness['model'] != spec.model:
        raise VerifiedDataError(f'{spec.model}: {label} witness declares {witness["model"]}')
    if witness['format'] != spec.quant:
        raise VerifiedDataError(f'{spec.model}: {label} witness format disagrees with the '
                                f'{VARIANT} spec: {witness["format"]}')
    # The same root policy the source gate uses: a published closure reached through a planted link is
    # refused for the alias, instead of being verified through the very hop it should be refused for.
    reject_aliased_ancestors(root, f'{label} path')
    if not root.is_dir():
        raise VerifiedDataError(f'{spec.model}: {label} root is not a directory: {root}')
    members = witness['members']
    actual = {safe_member(path.name) for path in root.iterdir() if path.name != spec.sidecar}
    if actual != set(members):
        missing, extra = sorted(set(members) - actual), sorted(actual - set(members))
        raise VerifiedDataError(f'{spec.model}: {label} member set disagrees with the witness: '
                                f'missing={missing[:4]} extra={extra[:4]}')
    check_layout(root, spec, check_format(root, spec,
                                          members={name for name in actual if name.endswith('.npy')}))
    for name, entry in sorted(members.items()):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise VerifiedDataError(f'{spec.model}: {label} member is not a regular file: {path}')
        sha, size = stream_hash(path)
        if size != entry['bytes'] or sha != entry['sha256']:
            raise VerifiedDataError(f'{spec.model}: {label} member changed: {name} '
                                    f'size {size} != {entry["bytes"]} or sha256 mismatch')
        if entry.get('header') is not None:
            header = read_npy_header(path)
            if header != entry['header']:
                raise VerifiedDataError(f'{spec.model}: {label} array header changed: {name} '
                                        f'{header} != witnessed {entry["header"]}')


def verify_published(root: Path, *, spec: DataSpec, label: str = 'published'):
    """Verify a published weights root with its own witness. No source root, no P/E, no legacy tree."""
    root = Path(root)
    witness = load_witness(root / spec.sidecar)
    if witness['model'] != spec.model:
        raise VerifiedDataError(f'{root}: the published witness is for {witness["model"]}, not {spec.model}')
    check_payload(root, witness, spec=spec, label=label)
    return witness


def verify_source(source: Path, witness, *, spec: DataSpec, accepted_data_root: Path) -> None:
    source = _confine(Path(source), Path(accepted_data_root))
    if witness['origin']['source_root'] != str(source.resolve()):
        raise VerifiedDataError(f'{spec.model}: witness origin {witness["origin"]["source_root"]} is not '
                                f'the requested data source; restate the witness or fix --source')
    check_payload(source, witness, spec=spec, label='source')


def write_witness(target: Path, witness) -> Path:
    """An independent copy of the reviewed document; metadata never aliases the source inode."""
    raw = json.dumps(_validate_document(witness), indent=2, sort_keys=True, allow_nan=False) + '\n'
    return _write_atomic(Path(target), raw.encode())


def publish_witness(reviewed: Path, target: Path) -> Path:
    """Publish the exact bytes of the witness a stage consumed, validated and on a fresh inode.

    Re-serializing an equivalent document would publish different bytes under the same meaning, and a
    published receipt could then never prove that its recorded consumed `--witness` input is this
    sidecar. The published copy is still written independently: no link, no copy, no stat carry-over.
    """
    reviewed = Path(reviewed)
    raw = _bounded_bytes(reviewed, MAX_WITNESS_BYTES, 'reviewed data witness')
    _validate_document(_parse_json(raw, reviewed, 'reviewed data witness'))
    return _write_atomic(Path(target), raw)


def _write_atomic(target: Path, raw: bytes) -> Path:
    if target.is_symlink():
        raise VerifiedDataError(f'symlinked witness output: {target}')
    reject_aliased_ancestors(target.parent, 'witness output path')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'.{target.name}.tmp-{os.getpid()}')
    with temporary.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return target


def _source_identity(source: Path) -> list:
    """Cheap pre/post tripwire; the authoritative check is the published payload hash."""
    return sorted((path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in source.iterdir())


def materialize(source: Path, out: Path, witness_path: Path, *, model: str,
                accepted_data_root: Path) -> dict:
    """Copy witnessed bytes privately and publish the witness with them. No quantizer runs here."""
    spec = spec_for(model)
    source, target = Path(source), Path(out)
    witness = load_witness(witness_path)
    if witness['model'] != spec.model:
        raise VerifiedDataError(f'witness is for {witness["model"]}, not {spec.model}')
    reject_aliased_ancestors(target.parent, 'materialization target path')
    _confine(source, Path(accepted_data_root))
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise VerifiedDataError(f'{spec.model}: refusing to materialize into {target}; '
                                'it is not an empty directory')
    source_root, target_root = Path(os.path.abspath(source)), Path(os.path.abspath(target))
    if source_root == target_root or target_root.is_relative_to(source_root) \
            or source_root.is_relative_to(target_root):
        raise VerifiedDataError(f'{spec.model}: the materialization target may not overlap the data source')
    before = _source_identity(source)
    verify_source(source, witness, spec=spec, accepted_data_root=accepted_data_root)
    target.mkdir(parents=True, exist_ok=True)
    for name, entry in sorted(witness['members'].items()):
        destination = target / name
        if destination.is_symlink():
            raise VerifiedDataError(f'symlink inside the materialization target: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (source / name).open('rb') as handle, destination.open('xb') as fresh:
            shutil.copyfileobj(handle, fresh)
        os.chmod(destination, 0o644)
    if _source_identity(source) != before:
        raise VerifiedDataError(f'{spec.model}: data source changed while materializing {source}')
    publish_witness(witness_path, target / spec.sidecar)
    # Re-check the closure exactly as a later consumer does, from the published witness itself.
    verify_published(target, spec=spec)
    return {'model': spec.model, 'source_root': witness['origin']['source_root'],
            'payload_digest': witness['payload_digest'], 'members': len(witness['members']),
            'witness': str(target / spec.sidecar)}


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', type=Path, help='the retained dump this command consumes')
    parser.add_argument('--out', type=Path, help='the weights root this command writes or verifies')
    parser.add_argument('--witness', type=Path)
    parser.add_argument('--accepted-data-root', type=Path)
    parser.add_argument('--inspect', action='store_true', help='report the dump records and layout')
    parser.add_argument('--build-witness', action='store_true')
    parser.add_argument('--materialize', action='store_true')
    parser.add_argument('--verify-published', action='store_true')
    parser.add_argument('--provenance', default='', help='the real origin, stated honestly')
    parser.add_argument('--legacy-field', action='append', default=[], metavar='KEY=VALUE',
                        help='a historical receipt/program field for the witness origin record')
    arguments = parser.parse_args()
    if sum((bool(arguments.inspect), arguments.build_witness, arguments.materialize,
           arguments.verify_published)) != 1:
        parser.error('choose exactly one of --inspect, --build-witness, --materialize, --verify-published')
    spec = spec_for(arguments.model)
    if arguments.inspect:
        if arguments.source is None:
            parser.error('--inspect needs --source')
        packed = check_format(arguments.source, spec)
        check_layout(arguments.source, spec, packed)
        print(json.dumps({'model': spec.model, 'packed': len(packed),
                          'arrays': len([path for path in arguments.source.iterdir()
                                         if path.suffix == '.npy'])}, sort_keys=True))
        return 0
    if arguments.build_witness:
        if arguments.source is None or arguments.witness is None:
            parser.error('--build-witness needs --source and --witness')
        legacy = {}
        for field in arguments.legacy_field:
            key, _, value = field.partition('=')
            if not key or not value or key in legacy:
                parser.error(f'--legacy-field needs a fresh KEY=VALUE, got {field!r}')
            legacy[key] = value
        write_witness(arguments.witness, build_witness(arguments.source, model=spec.model,
                                                      provenance=arguments.provenance, legacy=legacy))
        print(f'{VARIANT} witness {arguments.witness}')
        return 0
    if arguments.verify_published:
        if arguments.out is None:
            parser.error('--verify-published needs --out')
        witness = verify_published(arguments.out, spec=spec)
        print(json.dumps({'model': spec.model, 'published': str(arguments.out),
                          'payload_digest': witness['payload_digest']}, sort_keys=True))
        return 0
    missing = [name for name, value in (('--source', arguments.source), ('--out', arguments.out),
                                        ('--witness', arguments.witness),
                                        ('--accepted-data-root', arguments.accepted_data_root))
               if value is None]
    if missing:
        parser.error(f'--materialize needs {" ".join(missing)}')
    print(json.dumps(materialize(arguments.source, arguments.out, arguments.witness, model=spec.model,
                                 accepted_data_root=arguments.accepted_data_root), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
