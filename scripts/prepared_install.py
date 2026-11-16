#!/usr/bin/env python3
"""Verify an operator-reviewed prepared install manifest without building or publishing.

The manifest is local authorization, not a signature: its hashes must come from the
accepted preparation evidence, not be regenerated to bless changed inputs. Existing
recipe keys remain historical; output digests and provenance are checked independently
of the dispatcher's current, deliberately broad source-authority cache key.
"""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import replace

import model_artifact_dispatch as dispatch
import model_artifact_stages as stage_runtime
import verified_gemma_data as verified
import verified_gemma4_data as gemma4
from check_iron_artifact_provenance import validate as validate_iron
from lib import artifact_manifest
from lib.check_served_artifact_source import check_target
from lib.artifact_manifest import sha256_file
from lib.collect_served_artifacts import drop_covered, scenario_artifact_paths


def verify_receipt(recipe, witness):
    """Verify a published receipt against its explicit historical witness, then return the record."""
    receipt = dispatch.manifest_path(recipe)
    if sha256_file(receipt) != witness['sha256']:
        raise ValueError(f'{recipe.model}: prepared receipt changed')
    record = json.loads(receipt.read_text())
    key = witness['recipe_key']
    if (not re.fullmatch(r'[0-9a-f]{64}', key)
            or record.get('model') != recipe.model or record.get('scenario') != recipe.scenario
            or not dispatch.cache_hit(recipe, key)):
        raise ValueError(f'{recipe.model}: prepared receipt/output verification failed')
    return record


def _spec(model):
    """A manifest that names a model the fixed factory does not implement is a manifest error."""
    try:
        return verified.spec_for(model)
    except verified.VerifiedDataError as error:
        raise ValueError(str(error)) from error


def _g4_path(raw, label):
    """Accept a historical path claim lexically; never resolve retired inputs."""
    if (not isinstance(raw, str) or not raw.startswith('/') or raw.startswith('//')
            or '//' in raw or '/./' in raw or raw.endswith('/.') or any(part == '..' for part in raw.split('/'))):
        raise ValueError(f'gemma4-12b: invalid {label}')
    return raw


def _g4_entry(entry):
    """Validate the fixed v2 Gemma4 declaration without applying G3's schema."""
    if not isinstance(entry, dict) or set(entry) != {'variant', 'recipe_key', 'source_root', 'witness'}:
        raise ValueError('gemma4-12b: verified-data declaration needs variant/recipe_key/source_root/witness')
    if entry['variant'] != gemma4.VARIANT or not isinstance(entry['witness'], dict) \
            or set(entry['witness']) != {'sha256'}:
        raise ValueError('gemma4-12b: invalid verified-data declaration')
    for name, value in (('recipe_key', entry['recipe_key']), ('witness sha256', entry['witness']['sha256'])):
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError(f'gemma4-12b: {name} must be a lowercase sha256')
    source = _g4_path(entry['source_root'], 'source_root')
    if not source.endswith('/gemma4-12b'):
        raise ValueError('gemma4-12b: source_root must end in /gemma4-12b')


def data_variant_gate(data):
    """Reject a manifest whose verified-data bindings are not stated before any live check runs.

    Version 1 stays exactly the manifest it was: no variant may be declared there, so an older
    verifier can never ignore a variant request. Version 2 must declare one, and only for a model the
    fixed factory actually implements. The data owner is a runtime argument, but a variant's relative
    data location is declared by the model, so the manifest cannot widen it.
    """
    version = data['version']
    declared = data.get('data_variants', {})
    if version not in (1, 2):
        raise ValueError('unsupported prepared manifest version')
    if version == 1 and declared:
        raise ValueError(f'prepared manifest version 1 cannot declare data_variants '
                         f'{sorted(declared)}; publish it as version 2')
    if version == 2 and not declared:
        raise ValueError('prepared manifest version 2 needs data_variants')
    for model, entry in sorted(declared.items()):
        if model == gemma4.MODEL:
            if type(version) is not int or version != 2:
                raise ValueError('gemma4-12b: verified-data needs manifest version 2')
            _g4_entry(entry)
            continue
        _spec(model)
        if set(entry) != {'variant', 'recipe_key', 'source_root', 'witness'} \
                or entry['variant'] != verified.VARIANT:
            raise ValueError(f'{model}: unknown {verified.VARIANT} declaration {entry.get("variant")!r}; '
                             'needs variant/recipe_key/source_root/witness')
        if not isinstance(entry['witness'], dict) or set(entry['witness']) != {'sha256'}:
            raise ValueError(f'{model}: {verified.VARIANT} witness needs a sha256 and a source_root')
        # The manifest never selects I/O -- the receipt does that -- but the historical root it binds for
        # audit has to be a location this model could legally read, or the audit string is decoration.
        locations = dispatch.VERIFIED_DATA_INPUTS.get(model, ())
        suffixes = sorted('/' + relative for relative in locations)
        if not isinstance(entry['source_root'], str) or not entry['source_root'].startswith('/') \
                or not any(entry['source_root'].endswith(suffix) for suffix in suffixes):
            raise ValueError(f'{model}: {verified.VARIANT} source_root {entry.get("source_root")!r} is not '
                             f'inside a data root this model declares: {suffixes}')
        for name, value in (('recipe_key', entry['recipe_key']), ('sha256', entry['witness']['sha256'])):
            if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
                raise ValueError(f'{model}: {verified.VARIANT} {name} must be a lowercase sha256')
    return declared


def _producer_command(recipe, record, producer):
    """The receipt's own recorded command for the data stage, or a refusal if the record is not its own."""
    recorded = next((item for item in record.get('commands') or []
                     if item.get('step') == dispatch.VERIFIED_DATA_STEP), None)
    if recorded is None:
        raise ValueError(f'{recipe.model}: the witnessed recipe records no {dispatch.VERIFIED_DATA_STEP} '
                         'command; a stage record alone does not show what was run')
    fields = ('step', 'argv', 'env')
    if {key: recorded.get(key) for key in fields} != {key: (producer.get('command') or {}).get(key)
                                                      for key in fields}:
        raise ValueError(f'{recipe.model}: the {dispatch.VERIFIED_DATA_STEP} stage record is not the command '
                         'the witnessed recipe records')
    for item in record.get('commands') or []:
        if 'dump_llm_weights' in json.dumps(item.get('argv')):
            raise ValueError(f'{recipe.model}: the witnessed recipe still runs the quantizer in '
                             f'{item.get("step")}')
    argv = list(recorded.get('argv') or [])
    contract = recorded.get('stage')
    if not isinstance(contract, dict) or contract.get('dependencies'):
        raise ValueError(f'{recipe.model}: the {dispatch.VERIFIED_DATA_STEP} producer must declare a stage '
                         'contract with no stage dependency')
    return argv


def _consumed_inputs(recipe, producer, argv, document, sidecar):
    """Prove from the published receipt which bytes the recorded stage actually consumed.

    `model_artifact_stages.source_snapshot` records the command and every file of the stage's declared
    input views with its hash, and `producer_source` is the owner digest of that snapshot. The witness
    the stage consumed is therefore provable by hash -- the published sidecar is the exact bytes of that
    file, so one SHA answers for both -- and the payload is provable member by member under the
    historical source root. The root is never read here; it is only the prefix the consumed keys carry.
    """
    consumed = producer.get('inputs')
    if not isinstance(consumed, dict) or not consumed:
        raise ValueError(f'{recipe.model}: the recorded {dispatch.VERIFIED_DATA_STEP} stage consumed no '
                         'inputs; a marker with no consumed data is not a variant binding')
    if len(argv) != 3 + 2 * len(dispatch.VERIFIED_DATA_FLAGS) or argv[2] != '--materialize' \
            or tuple(argv[3::2]) != dispatch.VERIFIED_DATA_FLAGS \
            or Path(argv[1]).parts[-2:] != ('scripts', 'verified_gemma_data.py'):
        raise ValueError(f'{recipe.model}: the recorded {dispatch.VERIFIED_DATA_STEP} command is not the '
                         'data verifier with its fixed --materialize argument grammar')

    def flag(name):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None

    source_root = document['origin']['source_root']
    if flag('--model') != recipe.model or flag('--source') != source_root \
            or flag('--out') != str(sidecar.parent) or flag('--accepted-data-root') is None:
        raise ValueError(f'{recipe.model}: the recorded {dispatch.VERIFIED_DATA_STEP} command binds a model, '
                         'data source, data owner or output other than this subject\'s')
    owner = Path(flag('--accepted-data-root'))
    if not owner.is_absolute() or '..' in owner.parts \
            or source_root not in {str(owner / relative) for relative in dispatch.VERIFIED_DATA_INPUTS[recipe.model]}:
        raise ValueError(f'{recipe.model}: the recorded data owner does not contain the model-declared source')
    prefix = source_root.rstrip('/') + '/'
    for name, member in sorted(document['members'].items()):
        if consumed.get(prefix + name) != member['sha256']:
            raise ValueError(f'{recipe.model}: the recorded {dispatch.VERIFIED_DATA_STEP} inputs do not prove '
                             f'it consumed the witnessed {name}')
    if {key for key in consumed if key.startswith(prefix)} != {prefix + name for name in document['members']}:
        raise ValueError(f'{recipe.model}: the recorded {dispatch.VERIFIED_DATA_STEP} inputs are not exactly '
                         'the witnessed payload members')
    reviewed = flag('--witness')
    if reviewed is None or consumed.get(reviewed) != sha256_file(sidecar):
        raise ValueError(f'{recipe.model}: the recorded {dispatch.VERIFIED_DATA_STEP} inputs do not prove it '
                         'consumed the published data witness')


def verify_data_variant(recipe, record, entry):
    """Bind one published subject to its witnessed data variant, from the published closure alone.

    The witness is the sidecar the variant published inside its declared output scope, the manifest
    binds its hash, and the receipt's own stage record binds the exact bytes the stage consumed. Nothing
    here dereferences the data owner, the preparation directory or a retired legacy tree, and no current
    helper hash or recomputed recipe key is required, because a historical receipt must stay verifiable.
    """
    spec = _spec(recipe.model)
    if entry.get('variant') != verified.VARIANT:
        raise ValueError(f'{recipe.model}: unknown {verified.VARIANT} declaration {entry.get("variant")!r}')
    if entry.get('model', recipe.model) != recipe.model:
        raise ValueError(f'{recipe.model}: variant declaration names {entry["model"]}')
    if entry['recipe_key'] != record.get('recipe_key'):
        raise ValueError(f'{recipe.model}: variant binding does not match the witnessed receipt key')
    stage = (record.get('stages') or {}).get(dispatch.VERIFIED_DATA_STEP)
    if not isinstance(stage, dict):
        raise ValueError(f'{recipe.model}: the witnessed receipt has no {dispatch.VERIFIED_DATA_STEP} '
                         'stage record; a historical receipt is not a variant binding')
    if stage.get('action') not in ('executed', 'verified-reuse'):
        raise ValueError(f'{recipe.model}: {dispatch.VERIFIED_DATA_STEP} stage action '
                         f'{stage.get("action")!r} did not materialize verified data')
    producer = stage.get('producer') or {}
    sources = set(producer.get('sources') or {})
    if 'scripts/verified_gemma_data.py' not in sources or any('dump_llm_weights' in name for name in sources):
        raise ValueError(f'{recipe.model}: the recorded {dispatch.VERIFIED_DATA_STEP} producer is not the '
                         'data verifier; retained bytes are not adopted quantizer output')
    if producer.get('sdk') is not None or producer.get('toolchain') is not None:
        raise ValueError(f'{recipe.model}: a data verifier cannot claim compiler or toolchain provenance')
    if stage.get('producer_source') != dispatch.stage_digest(producer):
        raise ValueError(f'{recipe.model}: the recorded {dispatch.VERIFIED_DATA_STEP} producer_source does '
                         'not bind its producer record')
    argv = _producer_command(recipe, record, producer)
    key = dispatch.stage_digest({'version': dispatch.StageRunner.VERSION, 'source': stage['producer_source'],
                                 'dependencies': {}})
    if stage.get('key') != key:
        raise ValueError(f'{recipe.model}: the recorded {dispatch.VERIFIED_DATA_STEP} stage key does not bind '
                         'that producer source and its declared dependencies')
    roots = [Path(raw) / verified.SIDECAR for raw in recipe.outputs
             if (Path(raw) / verified.SIDECAR).is_file()]
    if len(roots) != 1:
        raise ValueError(f'{recipe.model}: expected exactly one published {verified.SIDECAR} among the '
                         f'declared output roots, found {len(roots)}')
    sidecar = roots[0]
    document = verified.load_witness(sidecar)
    if sha256_file(sidecar) != entry['witness']['sha256']:
        raise ValueError(f'{recipe.model}: published data witness changed')
    if document['model'] != recipe.model or document['variant'] != verified.VARIANT:
        raise ValueError(f'{recipe.model}: published witness declares '
                         f'{document["model"]}/{document["variant"]}')
    if document['origin']['source_root'] != entry['source_root']:
        raise ValueError(f'{recipe.model}: witness origin {document["origin"]["source_root"]} is not the '
                         f'bound source_root {entry["source_root"]}')
    _consumed_inputs(recipe, producer, argv, document, sidecar)
    verified.check_payload(sidecar.parent, document, spec=spec, label='published')


def _strict_equal(left, right):
    """JSON equality which does not treat True as 1."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def _g4_contract(actual, expected, label):
    """Compare JSON stage declarations without bool/int coercion."""
    if not _strict_equal(actual, expected):
        raise ValueError(f'gemma4-12b: {label} stage contract changed')


def _g4_view(root, patterns=(), require_each=False):
    return {'root': root, 'patterns': list(patterns), 'require_each': require_each}


def _g4_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _g4_manifest(stage, label):
    """Reapply the frontier identity invariant without reopening retired outputs."""
    files, produced = stage.get('files'), stage.get('producer_outputs')
    if not isinstance(files, dict) or not files or not isinstance(produced, dict) or set(files) != set(produced):
        raise ValueError(f'gemma4-12b: {label} output manifest is missing or diverges')
    for name, settled in files.items():
        if not isinstance(name, str):
            raise ValueError(f'gemma4-12b: invalid {label} output name')
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or str(path) != name:
            raise ValueError(f'gemma4-12b: invalid {label} output name')
        original = produced[name]
        for fingerprint in (settled, original):
            if (not isinstance(fingerprint, dict) or set(fingerprint) != {'sha256', 'identity', 'executable'}
                    or not _g4_sha(fingerprint['sha256']) or not _g4_sha(fingerprint['identity'])
                    or type(fingerprint['executable']) is not bool
                    or (path.name != 'meta.json' and fingerprint['identity'] != fingerprint['sha256'])):
                raise ValueError(f'gemma4-12b: invalid {label} output fingerprint')
        # Publication can rewrite metadata paths; StageRunner preserves normalized identity.
        if (settled['identity'] != original['identity'] or settled['executable'] != original['executable']
                or (path.name != 'meta.json' and settled['sha256'] != original['sha256'])):
            raise ValueError(f'gemma4-12b: {label} producer output identity changed')
    if stage.get('output_identity') != _g4_output_identity(files):
        raise ValueError(f'gemma4-12b: {label} output identity is unbound')
    return files


def _g4_output_identity(files):
    return dispatch.stage_digest({name: [value['identity'], value['executable']] for name, value in files.items()})


def _g4_published_manifest(recipe, record):
    """Compare settled fingerprints with live serving files, never recorded locations."""
    artifacts = Path(recipe.outputs[0]).parents[1]
    actual = {}
    for raw in recipe.outputs:
        root = gemma4.declared_path(raw, 'published serving root')
        for path in stage_runtime.files(dispatch.View(str(root))):
            if path.is_symlink():
                raise ValueError('gemma4-12b: linked published frontier member')
            name = 'artifacts/' + str(path.relative_to(artifacts))
            if path.name == 'meta.json':
                raw_meta = gemma4._read(path, gemma4.MAX_METADATA_BYTES, 'published program metadata')
                sha = hashlib.sha256(raw_meta).hexdigest()
                identity = dispatch.stage_digest(gemma4._json(raw_meta, str(path)))
            else:
                sha = identity = sha256_file(path)
            actual[name] = {'sha256': sha, 'identity': identity,
                            'executable': bool(path.stat().st_mode & 0o100)}
    if not _strict_equal(actual, record['stages']['package']['files']):
        raise ValueError('gemma4-12b: published files differ from settled package frontier')


def _g4_snapshot(producer, contract, label, *, lock_sha=None):
    """Validate the captured source_snapshot shape without reopening retired source trees."""
    required = {'command', 'executable', 'sources', 'symbols', 'environment', 'sdk', 'toolchain', 'inputs'}
    if set(producer) != required:
        raise ValueError(f'gemma4-12b: {label} producer snapshot shape changed')
    if (not isinstance(producer['inputs'], dict) or not producer['inputs']
            or not all(isinstance(name, str) and _g4_sha(value) for name, value in producer['inputs'].items())):
        raise ValueError(f'gemma4-12b: {label} consumed inputs are missing or malformed')
    executable = producer['executable']
    if (not isinstance(executable, dict) or set(executable) != {'resolved', 'sha256'}
            or not _g4_path(executable['resolved'], f'{label} executable') or not _g4_sha(executable['sha256'])):
        raise ValueError(f'gemma4-12b: {label} executable snapshot changed')
    sources = producer['sources']
    if not isinstance(sources, dict) or not all(isinstance(key, str) and _g4_sha(value) for key, value in sources.items()):
        raise ValueError(f'gemma4-12b: {label} source hashes changed')
    declared = set(contract['sources'])
    directories = {name for name in declared if name in {'scripts/lib', 'aie_kernels', 'third_party/iron'}}
    explicit = declared - directories
    if not explicit <= set(sources) or not all(any(key.startswith(directory + '/') for key in sources)
                                                   for directory in directories):
        raise ValueError(f'gemma4-12b: {label} source coverage changed')
    for key in sources:
        relative = Path(key)
        if relative.is_absolute() or '..' in relative.parts or str(relative) != key:
            raise ValueError(f'gemma4-12b: {label} has noncanonical source key')
        if key in explicit:
            continue
        directory = next((root for root in directories if key.startswith(root + '/')), None)
        if directory is None:
            raise ValueError(f'gemma4-12b: {label} has foreign source key')
        parts = relative.relative_to(directory).parts
        if (any(part.startswith('.') or part in {'__pycache__', 'tests', 'fixtures'} for part in parts)
                or relative.name.startswith('test_') or relative.suffix in {'.pyc', '.pyo', '.md', '.log'}):
            raise ValueError(f'gemma4-12b: {label} source is excluded by source_snapshot')
    if lock_sha is not None and sources.get('toolchain.lock') != lock_sha:
        raise ValueError('gemma4-12b: native source lock differs from accepted lock')
    symbols = producer['symbols']
    if not isinstance(symbols, dict) or set(symbols) != set(contract['symbols']) or not all(isinstance(v, str) and v for v in symbols.values()):
        raise ValueError(f'gemma4-12b: {label} symbols changed')
    environment = producer['environment']
    if not isinstance(environment, dict) or not all(isinstance(name, str) for name in environment):
        raise ValueError(f'gemma4-12b: {label} captured environment changed')
    allowed = {'LD_LIBRARY_PATH', 'PYTHONPATH'}
    for pattern in contract['environment']:
        allowed.update(name for name in environment if fnmatch.fnmatchcase(name, pattern))
    if (not isinstance(environment, dict) or not {'LD_LIBRARY_PATH', 'PYTHONPATH'} <= set(environment)
            or not set(environment) <= allowed or not all(value is None or isinstance(value, str) for value in environment.values())):
        raise ValueError(f'gemma4-12b: {label} captured environment changed')
    for name, value in producer['command']['env'].items():
        if name in allowed and environment.get(name) != value:
            raise ValueError(f'gemma4-12b: {label} command/captured environment disagree')
    argv = producer['command']['argv']
    if (Path(argv[0]).is_absolute() and executable['resolved'] != argv[0]) \
            or (not Path(argv[0]).is_absolute() and Path(executable['resolved']).name != argv[0]):
        raise ValueError(f'gemma4-12b: {label} executable is not its recorded command')
    if label == 'proof':
        if producer['sdk'] is not None or producer['toolchain'] is not None:
            raise ValueError('gemma4-12b: proof SDK/toolchain provenance changed')
    elif label == 'native':
        sdk, toolchain = producer['sdk'], producer['toolchain']
        if (not isinstance(sdk, dict) or set(sdk) != {'python', 'sha256', 'distributions'}
                or not _g4_path(sdk['python'], 'native SDK python') or not _g4_sha(sdk['sha256'])
                or not isinstance(sdk['distributions'], str) or not isinstance(toolchain, dict)
                or set(toolchain) != {'instance', 'peano', 'files'} or not _g4_path(toolchain['instance'], 'toolchain instance')
                or not _g4_path(toolchain['peano'], 'toolchain peano') or not isinstance(toolchain['files'], dict)
                or not toolchain['files']):
            raise ValueError('gemma4-12b: native SDK/toolchain provenance changed')
        try:
            distributions = json.loads(sdk['distributions'])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError('gemma4-12b: native SDK distributions changed') from error
        if (not isinstance(distributions, dict) or set(distributions) != set(contract['sdk_packages'])
                or any(not isinstance(v, dict) or set(v) != {'version', 'record'}
                       or not isinstance(v['version'], str) or not v['version'] or not _g4_sha(v['record'])
                       for v in distributions.values()) or sdk['python'] != contract['sdk_python']):
            raise ValueError('gemma4-12b: native SDK distributions changed')
        venv = str(Path(sdk['python']).parent.parent)
        if toolchain['peano'] != venv + '/lib/python3.14/site-packages/llvm-aie':
            raise ValueError('gemma4-12b: Peano does not belong to the captured SDK')
        for name, expected in {'RF_INST': toolchain['instance'], 'AIECC_PATH': toolchain['instance'] + '/bin/aiecc',
                               'PEANO_INSTALL_DIR': toolchain['peano']}.items():
            if environment.get(name) is not None and environment[name] != expected:
                raise ValueError('gemma4-12b: captured compiler override disagrees with toolchain')
        roots = (toolchain['instance'] + '/bin/', toolchain['instance'] + '/python/',
                 toolchain['instance'] + '/build/include/', toolchain['instance'] + '/build/lib/',
                 toolchain['peano'] + '/bin/', toolchain['peano'] + '/include/', toolchain['peano'] + '/lib/')
        if any(not _g4_path(name, 'toolchain file') or not name.startswith(roots)
               or not isinstance(value, dict) or set(value) != {'sha256'} or not _g4_sha(value['sha256'])
               for name, value in toolchain['files'].items()):
            raise ValueError('gemma4-12b: native toolchain files changed')
    elif label == 'package':
        sdk = producer['sdk']
        if producer['toolchain'] is not None or not isinstance(sdk, dict) or set(sdk) != {'python', 'sha256', 'distributions'} \
                or not _g4_path(sdk['python'], 'package SDK python') or not _g4_sha(sdk['sha256']) or not isinstance(sdk['distributions'], str):
            raise ValueError('gemma4-12b: package SDK provenance changed')
        try:
            distributions = json.loads(sdk['distributions'])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError('gemma4-12b: package SDK distributions changed') from error
        if (not isinstance(distributions, dict) or set(distributions) != set(contract['sdk_packages'])
                or any(not isinstance(v, dict) or set(v) != {'version', 'record'}
                       or not isinstance(v['version'], str) or not v['version'] or not _g4_sha(v['record'])
                       for v in distributions.values()) or sdk['python'] != contract['sdk_python']
                or sdk['python'] != executable['resolved'] or sdk['sha256'] != executable['sha256']):
            raise ValueError('gemma4-12b: package SDK distributions changed')


def _g4_receipt(recipe, witness, entry, accepted_lock_sha256):
    """Check the bounded historical receipt without reading historical source paths."""
    if not _g4_sha(accepted_lock_sha256):
        raise ValueError('gemma4-12b: accepted lock hash is required')
    path = dispatch.manifest_path(recipe)
    with path.open('rb') as handle:
        raw = handle.read(gemma4.MAX_WITNESS_BYTES + 1)
    if len(raw) > gemma4.MAX_WITNESS_BYTES or hashlib.sha256(raw).hexdigest() != witness['sha256']:
        raise ValueError('gemma4-12b: prepared receipt changed or oversized')
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('gemma4-12b: invalid prepared receipt') from error
    if not _strict_equal({key: record.get(key) for key in ('model', 'scenario', 'recipe_key')},
                         {'model': gemma4.MODEL, 'scenario': recipe.scenario, 'recipe_key': entry['recipe_key']}) \
            or not _strict_equal(witness.get('recipe_key'), entry['recipe_key']):
        raise ValueError('gemma4-12b: prepared receipt identity differs from pinned binding')
    commands, stages = record.get('commands'), record.get('stages')
    if not isinstance(commands, list) or len(commands) != 3 or not isinstance(stages, dict) \
            or set(stages) != {'verified-data', 'build', 'package'}:
        raise ValueError('gemma4-12b: receipt is not the fixed three-stage closure')
    order = ('verified-data', 'build', 'package')
    if any(not isinstance(item, dict) or item.get('step') != step for item, step in zip(commands, order)):
        raise ValueError('gemma4-12b: receipt command order changed')
    by_step = dict(zip(order, commands))
    for step, dependencies in zip(order, ((), ('verified-data',), ('verified-data', 'build'))):
        stage, command = stages[step], by_step[step]
        if not isinstance(stage, dict) or not isinstance(stage.get('producer'), dict) \
                or type(stage.get('version')) is not int or stage.get('version') != dispatch.StageRunner.VERSION \
                or stage.get('stage') != step or stage.get('action') not in ('executed', 'verified-reuse'):
            raise ValueError(f'gemma4-12b: invalid {step} stage record')
        contract = command.get('stage')
        if not isinstance(contract, dict) or not _strict_equal(contract.get('dependencies'), list(dependencies)):
            raise ValueError(f'gemma4-12b: {step} stage contract dependencies changed')
        producer = stage['producer']
        if not _strict_equal(stage.get('producer_source'), dispatch.stage_digest(producer)):
            raise ValueError(f'gemma4-12b: {step} producer source changed')
        expected_deps = {dep: {'key': stages[dep].get('key'), 'outputs': stages[dep].get('output_identity')}
                         for dep in dependencies}
        expected_key = dispatch.stage_digest({'version': dispatch.StageRunner.VERSION,
                                              'source': stage['producer_source'], 'dependencies': expected_deps})
        if not _strict_equal(stage.get('key'), expected_key) or not isinstance(stage.get('output_identity'), str):
            raise ValueError(f'gemma4-12b: {step} stage key/output identity changed')
        expected_command = {'step': command.get('step'), 'argv': command.get('argv'), 'env': command.get('env')}
        if not _strict_equal(producer.get('command'), expected_command):
            raise ValueError(f'gemma4-12b: {step} command does not bind its producer')
    data, native, package = (by_step[name] for name in order)
    proof_argv, native_argv, package_argv = data.get('argv'), native.get('argv'), package.get('argv')
    if not all(isinstance(argv, list) and all(isinstance(value, str) for value in argv)
               for argv in (proof_argv, native_argv, package_argv)):
        raise ValueError('gemma4-12b: recorded command arguments are not strings')
    # The helper identifies the historical checkout lexically.  Do not resolve or stat it:
    # this verifier must remain usable after that checkout was retired.
    if len(proof_argv) != 13 or not _g4_path(proof_argv[0], 'proof python') or \
            _g4_path(proof_argv[1], 'proof helper').endswith('/scripts/gemma4_verified_entry.py') is False:
        raise ValueError('gemma4-12b: recorded proof interpreter/helper changed')
    helper = proof_argv[1]
    old_repo = helper[:-len('/scripts/gemma4_verified_entry.py')]
    if not old_repo or helper != old_repo + '/scripts/gemma4_verified_entry.py':
        raise ValueError('gemma4-12b: malformed historical proof helper')
    owner = _g4_path(proof_argv[6], 'accepted data owner')
    source = _g4_path(proof_argv[4], 'proof source')
    witness = _g4_path(proof_argv[8], 'review witness')
    proof_root = _g4_path(proof_argv[12], 'proof output')
    old_build = proof_root[:-len('/gemma4-12b/verified-data')] if proof_root.endswith('/gemma4-12b/verified-data') else ''
    if (proof_argv[2:4] != ['proof', '--source'] or proof_argv[5] != '--accepted-data-root'
            or proof_argv[7] != '--witness' or proof_argv[9:11] != ['--expected-witness-sha256', entry['witness']['sha256']]
            or proof_argv[11] != '--out' or source != owner + '/gemma4-12b' or source != entry['source_root']
            or not old_build or witness == source or witness.startswith(source + '/')
            or witness == old_build or witness.startswith(old_build + '/')):
        raise ValueError('gemma4-12b: recorded proof command contract changed')
    if native_argv != ['bash', old_repo + '/designs/resident_forward/recipes/rf48C.sh']:
        raise ValueError('gemma4-12b: recorded native compiler command changed')
    native_env = native.get('env')
    native_python = _g4_path((native.get('stage') or {}).get('sdk_python'), 'native SDK python')
    native_venv = str(Path(native_python).parent.parent)
    expected_native_env = {'RF_BUILD': old_build + '/gemma4-12b/rf48C',
                           'KEEP_WORK': old_build + '/gemma4-12b/rf48C', 'RF_STORE': source + '/store',
                           'RF_MODEL': gemma4.MODEL, 'IRON': old_repo + '/third_party/iron',
                           'IRON_DIR': old_repo + '/third_party/iron', 'VENV_IRON': native_venv,
                           'PYTHONPATH': old_repo + '/third_party/iron', 'TOOLCHAIN_RESOLVE_ONLY': '1',
                           'RF_HF_CONFIG': source + '/hf_config/config.json'}
    if not _strict_equal(native_env, expected_native_env):
        raise ValueError('gemma4-12b: recorded native environment changed')
    artifacts = _g4_path(package_argv[12], 'artifacts root') if len(package_argv) == 19 else ''
    expected_package = [proof_argv[0], helper, 'package', '--source', source, '--accepted-data-root', owner,
                        '--proof-root', proof_root, '--expected-witness-sha256', entry['witness']['sha256'],
                        '--artifacts-root', artifacts, '--build-dir', expected_native_env['RF_BUILD'], '--out',
                        recipe.outputs[0], '--tokenizer-out', recipe.outputs[1]]
    if (len(package_argv) != 19 or package_argv != expected_package
            or artifacts != str(Path(recipe.outputs[0]).parents[1])
            or tuple(recipe.outputs[:3]) != (package_argv[16], package_argv[18],
                                              artifacts + '/gemma4-12b-qat/checkpoint')):
        raise ValueError('gemma4-12b: recorded package command contract changed')
    proof_contract = {'inputs': [_g4_view(source + '/store'), _g4_view(source + '/rf_stack'),
                                 _g4_view(source + '/hf_config', ('config.json',)),
                                 _g4_view(source + '/tokenizer', ('*.json', '*.jinja', '*.model')),
                                 _g4_view(str(Path(witness).parent), (Path(witness).name,))],
                      'outputs': [_g4_view(proof_root, (gemma4.SIDECAR,), True)], 'dependencies': [],
                      'sources': ['scripts/gemma4_verified_entry.py', 'scripts/verified_gemma4_data.py',
                                  'scripts/verified_gemma_data.py'], 'symbols': [], 'environment': [],
                      'sdk_python': '', 'sdk_packages': [], 'retain': False, 'scratch': [], 'toolchain': False}
    _g4_contract(data.get('stage'), proof_contract, 'proof')
    if not _strict_equal(data.get('env'), {}):
        raise ValueError('gemma4-12b: proof command environment changed')
    native_contract = native.get('stage')
    native_sources = ['toolchain.lock', 'scripts/amd_paths.sh', 'scripts/toolchain_up.sh', 'scripts/toolchain_gc.sh',
                      'scripts/fast_build_env.sh', 'scripts/cache_env.sh', 'scripts/require_disk_backed.sh',
                      'scripts/lib', 'aie_kernels', 'third_party/iron', 'third_party/iron/iron/common/test_utils.py',
                      'designs/resident_forward/bfp16_model.py', 'designs/resident_forward/chain_ref.py',
                      'designs/resident_forward/head_ref.py', 'designs/resident_forward/kvring.py',
                      'designs/resident_forward/m1_design.py', 'designs/resident_forward/m2_design.py',
                      'designs/resident_forward/mlp_ref.py', 'designs/resident_forward/o_ref.py',
                      'designs/resident_forward/rattn2_design.py', 'designs/resident_forward/rattnh.py',
                      'designs/resident_forward/rf_model.py', 'designs/resident_forward/rf_paths.py',
                      'designs/resident_forward/rforward.py', 'designs/resident_forward/rglobal.py',
                      'designs/resident_forward/rhead.py', 'designs/resident_forward/rlayer_design.py',
                      'designs/resident_forward/rmlp_design.py', 'designs/resident_forward/rmlp_ref.py',
                      'designs/resident_forward/rseg_design.py', 'designs/resident_forward/rsplitl.py',
                      'designs/decode_fused/elf_zst.py', 'designs/resident_forward/recipes/rf48C.sh',
                      'designs/resident_forward/fwd_ladder.sh', 'designs/resident_forward/env.sh',
                      'designs/resident_forward/rf_build.py', 'designs/resident_forward/rf_input.py',
                      'designs/resident_forward/fwd_pack.py', 'designs/resident_forward/kobj_cache.py']
    expected_native_contract = {
        'inputs': [_g4_view(source + '/hf_config', ('config.json',)),
                   _g4_view(source + '/store', ('manifest.json',))],
        'outputs': [_g4_view(expected_native_env['RF_BUILD'] + '/rf48C',
                            ('design.elf.zst', 'full_elf_config.json', 'fwd_layout.json', 'pack.json',
                             'gen_args.txt', 'gen_env.txt', 'params.txt'), True),
                    _g4_view(expected_native_env['RF_BUILD'] + '/kobj', ('run-*',), True)],
        'dependencies': ['verified-data'], 'sources': native_sources, 'symbols': [],
        'environment': ['IRON*', 'VENV_IRON', 'PEANO_INSTALL_DIR', 'AIE*', 'MODEL_AIECC', 'RF_*',
                        'PREFILL_*', 'DECODE_*', 'GEN_EXTRA', 'ACT_POLY', 'PRECISION', 'WINDOW_RUNGS',
                        'KV_BLOCK_T', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'],
        'sdk_python': native_python, 'sdk_packages': ['numpy'], 'retain': True,
        'scratch': [expected_native_env['RF_BUILD']], 'toolchain': True}
    _g4_contract(native_contract, expected_native_contract, 'native')
    package_contract = {'inputs': [_g4_view(source + '/store'), _g4_view(source + '/rf_stack'),
                                   _g4_view(source + '/hf_config', ('config.json',)),
                                   _g4_view(source + '/tokenizer', ('*.json', '*.jinja', '*.model'))],
                        'outputs': [_g4_view(raw) for raw in recipe.outputs[:3]],
                        'dependencies': ['verified-data', 'build'],
                        'sources': ['scripts/gemma4_verified_entry.py', 'scripts/verified_gemma4_data.py',
                                    'scripts/verified_gemma_data.py', 'designs/resident_forward/bfp16_model.py',
                                    'designs/resident_forward/chain_ref.py', 'designs/resident_forward/head_ref.py',
                                    'designs/resident_forward/kvring.py', 'designs/resident_forward/m1_design.py',
                                    'designs/resident_forward/m2_design.py', 'designs/resident_forward/mlp_ref.py',
                                    'designs/resident_forward/o_ref.py', 'designs/resident_forward/rattn2_design.py',
                                    'designs/resident_forward/rattnh.py', 'designs/resident_forward/rf_model.py',
                                    'designs/resident_forward/rf_paths.py', 'designs/resident_forward/rforward.py',
                                    'designs/resident_forward/rglobal.py', 'designs/resident_forward/rhead.py',
                                    'designs/resident_forward/rlayer_design.py', 'designs/resident_forward/rmlp_design.py',
                                    'designs/resident_forward/rmlp_ref.py', 'designs/resident_forward/rseg_design.py',
                                    'designs/resident_forward/rsplitl.py', 'third_party/iron/iron/common/quant.py',
                                    'designs/resident_forward/artifact_meta.py',
                                    'designs/resident_forward/package_artifact.py'],
                        'symbols': ['package_gemma4', 'copy_tokenizer'], 'environment': ['RF_*'],
                        'sdk_python': proof_argv[0], 'sdk_packages': ['numpy', 'safetensors'],
                        'retain': False, 'scratch': [], 'toolchain': False}
    package_stage = package.get('stage')
    if not isinstance(package_stage, dict):
        raise ValueError('gemma4-12b: package stage contract changed')
    _g4_contract(package_stage, package_contract, 'package')
    if not _strict_equal(package.get('env'), {'RF_MODEL': gemma4.MODEL, 'RF_HF_CONFIG': source + '/hf_config/config.json',
                                               'RF_STORE': source + '/store'}):
        raise ValueError('gemma4-12b: package environment changed')
    producer = stages['verified-data']['producer']
    if producer.get('sdk') is not None or producer.get('toolchain') is not None \
            or set(producer.get('sources', {})) != {'scripts/gemma4_verified_entry.py',
                                                     'scripts/verified_gemma4_data.py',
                                                     'scripts/verified_gemma_data.py'} \
            or not all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)
                       for value in producer.get('sources', {}).values()):
        raise ValueError('gemma4-12b: invalid proof producer provenance')
    # Snapshot evidence is typed independently of its self-hash.  It remains historical:
    # no source path above is opened by these checks.
    _g4_snapshot(stages['verified-data']['producer'], proof_contract, 'proof')
    _g4_snapshot(stages['build']['producer'], native_contract, 'native', lock_sha=accepted_lock_sha256)
    _g4_snapshot(stages['package']['producer'], package_contract, 'package')
    manifests = {name: _g4_manifest(stages[name], name) for name in order}
    proof_file = 'build/gemma4-12b/verified-data/' + gemma4.SIDECAR
    expected_proof = {proof_file: {'sha256': entry['witness']['sha256'],
                                 'identity': entry['witness']['sha256'], 'executable': False}}
    if not _strict_equal(manifests['verified-data'], expected_proof):
        raise ValueError('gemma4-12b: proof frontier does not bind the reviewed sidecar')
    native_prefix = 'build/gemma4-12b/rf48C/rf48C/'
    program_files = {native_prefix + name for name in native_contract['outputs'][0]['patterns']}
    cache_files = set(manifests['build']) - program_files
    cache_prefix = 'build/gemma4-12b/rf48C/kobj/'
    if not program_files <= set(manifests['build']) or not cache_files:
        raise ValueError('gemma4-12b: native frontier lacks program files or retained run cache')
    for name in cache_files:
        if not name.startswith(cache_prefix):
            raise ValueError('gemma4-12b: unowned native cache file')
        relative = Path(name[len(cache_prefix):]).parts
        if len(relative) < 2 or not relative[0].startswith('run-') or relative[0] == 'run-':
            raise ValueError('gemma4-12b: native cache is not owned by a fresh run')
    package_prefixes = tuple('artifacts/' + str(Path(raw).relative_to(artifacts)) + '/' for raw in recipe.outputs)
    if any(not name.startswith(package_prefixes) for name in manifests['package']):
        raise ValueError('gemma4-12b: package frontier has unowned files')
    # package_artifact copies these compiler outputs unchanged; gen_env has a separate derivation.
    resident_prefix = package_prefixes[0]
    for name in ('design.elf.zst', 'pack.json', 'fwd_layout.json', 'params.txt', 'gen_args.txt'):
        if not _strict_equal(manifests['build'][native_prefix + name],
                             manifests['package'].get(resident_prefix + name)):
            raise ValueError(f'gemma4-12b: packaged compiler output changed: {name}')
    keys = {}
    for command in commands:
        name = command['step']
        dependencies = {dep: {'key': keys[dep], 'outputs': _g4_output_identity(manifests[dep])}
                        for dep in command['stage']['dependencies']}
        keys[name] = dispatch.stage_digest({'version': dispatch.StageRunner.VERSION,
                                           'source': stages[name]['producer_source'], 'dependencies': dependencies})
        if stages[name]['key'] != keys[name]:
            raise ValueError(f'gemma4-12b: {name} dependency frontier is unbound')
    if record.get('output_digest') != dispatch.output_digest(recipe.outputs):
        raise ValueError('gemma4-12b: published output digest changed')
    return record


def _verify_g4_variant(recipe, record, entry):
    roots = tuple(recipe.outputs[:3])
    if len(roots) != 3 or any(not Path(root).is_absolute() for root in roots):
        raise ValueError('gemma4-12b: invalid fixed serving projection')
    # verify_published_data reads the pinned resident sidecar once and returns that parsed snapshot.
    document = gemma4.verify_published_data(*roots, expected_witness_sha256=entry['witness']['sha256'])
    if (document['model'] != gemma4.MODEL or document['variant'] != gemma4.VARIANT
            or document['origin']['source_root'] != entry['source_root']):
        raise ValueError('gemma4-12b: published witness differs from declaration')
    argv = record['commands'][0]['argv']
    source, reviewed = entry['source_root'], argv[8]
    expected = {reviewed: entry['witness']['sha256']}
    for role, relative in gemma4.ROLE_PATHS.items():
        for name, member in document['roles'][role]['members'].items():
            expected[f'{source}/{relative}/{name}'] = member['sha256']
    if not _strict_equal(record['stages']['verified-data']['producer'].get('inputs'), expected):
        raise ValueError('gemma4-12b: proof producer did not consume exactly witnessed roles and review pin')
    _g4_published_manifest(recipe, record)
    native_expected = {f'{source}/hf_config/config.json': document['roles']['build-config']['members']['config.json']['sha256'],
                       f'{source}/store/manifest.json': document['roles']['retained-store']['members']['manifest.json']['sha256']}
    if not _strict_equal(record['stages']['build']['producer'].get('inputs'), native_expected):
        raise ValueError('gemma4-12b: native producer did not consume exactly its pinned roles')
    package_expected = {}
    for role, relative in gemma4.ROLE_PATHS.items():
        package_expected.update({f'{source}/{relative}/{name}': member['sha256']
                                 for name, member in document['roles'][role]['members'].items()})
    if not _strict_equal(record['stages']['package']['producer'].get('inputs'), package_expected):
        raise ValueError('gemma4-12b: package producer did not consume exactly witnessed roles')


def checked_file(entry):
    path = Path(entry['path'])
    if not path.is_absolute() or sha256_file(path) != entry['sha256']:
        raise ValueError(f'prepared file missing or changed: {path}')
    return path


def run(argv, **kwargs):
    subprocess.run(list(map(str, argv)), check=True, stdout=sys.stderr, **kwargs)


def verify(manifest, repo):
    data = json.loads(manifest.read_text())
    variants = data_variant_gate(data)
    files = {name: checked_file(data[name]) for name in ('npu', 'npu_dev', 'onnxruntime', 'config', 'toolchain')}
    if files['onnxruntime'].name == 'libonnxruntime.so.1' or not re.fullmatch(r'libonnxruntime\.so\.1\.[0-9.]+', files['onnxruntime'].name):
        raise ValueError('prepared onnxruntime must be the real versioned .so')
    if files['toolchain'].read_bytes() != (repo / 'toolchain.lock').read_bytes():
        raise ValueError('prepared toolchain differs from repository lock')
    for name in ('npu', 'npu_dev'):
        if not os.access(files[name], os.X_OK):
            raise ValueError(f'prepared executable is not executable: {files[name]}')
    artifacts = Path(data['artifacts'])
    kernels = Path(data['kernels']['path'])
    if not artifacts.is_absolute() or not kernels.is_absolute():
        raise ValueError('prepared roots must be absolute')
    if (not kernels.is_dir() or kernels.is_symlink()
            or any(path.is_symlink() for path in kernels.rglob('*'))
            or dispatch.tree_digest(kernels) != data['kernels']['sha256']):
        raise ValueError('prepared kernel bundle missing, linked, or changed')
    config = files['config']
    recipes = dispatch.plan(config, repo, artifacts, artifacts.parent / 'model-inputs', artifacts.parent / 'build')
    bound = set()
    for recipe in {item.model: item for item in recipes}.values():
        # Gemma4's ordinary plan has eight roots.  Its published closure is deliberately
        # bounded to the three serving roots before receipt/output verification.
        g4_sidecar = recipe.model == gemma4.MODEL and any(
            (Path(raw) / gemma4.SIDECAR).is_file() for raw in recipe.outputs[:3])
        g4_declared = recipe.model == gemma4.MODEL and recipe.model in variants
        # Inspect the pinned receipt before the ordinary cache path so a surviving
        # stage marker cannot downgrade into the eight-root/raw route.
        g4_stage_marker = False
        if recipe.model == gemma4.MODEL and not g4_declared:
            receipt = dispatch.manifest_path(replace(recipe, outputs=recipe.outputs[:3]))
            try:
                with receipt.open('rb') as handle:
                    receipt_raw = handle.read(gemma4.MAX_WITNESS_BYTES + 1)
                pin = data['receipts'][recipe.model]['sha256']
                if len(receipt_raw) <= gemma4.MAX_WITNESS_BYTES and hashlib.sha256(receipt_raw).hexdigest() == pin:
                    g4_stage_marker = dispatch.VERIFIED_DATA_STEP in (json.loads(receipt_raw).get('stages') or {})
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if g4_declared or g4_sidecar or g4_stage_marker:
            if not g4_declared:
                raise ValueError('gemma4-12b: published closure marker lacks a version 2 binding')
            serving = replace(recipe, outputs=recipe.outputs[:3])
            record = _g4_receipt(serving, data['receipts'][recipe.model], variants[recipe.model],
                                 sha256_file(files['toolchain']))
            # A stage-only marker is equally binding: _g4_receipt refuses an ordinary
            # record rather than allowing the old eight-root cache path to run.
            _verify_g4_variant(serving, record, variants[recipe.model])
            bound.add(recipe.model)
            continue
        record = verify_receipt(recipe, data['receipts'][recipe.model])
        # A variant closure or a variant stage record is a request for the new binding, whichever half of
        # the published pair survived: a version 1 manifest (or a version 2 one that omits the model)
        # must not install it while an older verifier's checks silently skip.
        marker = dispatch.VERIFIED_DATA_STEP in (record.get('stages') or {}) \
            or any((Path(raw) / verified.SIDECAR).is_file() for raw in recipe.outputs)
        if marker and recipe.model not in variants:
            raise ValueError(f'{recipe.model}: the published closure carries a {verified.VARIANT} marker the '
                             'manifest does not declare; publish a version 2 manifest with the binding')
        if recipe.model in variants:
            verify_data_variant(recipe, record, variants[recipe.model])
            bound.add(recipe.model)
    if bound != set(variants):
        raise ValueError(f'prepared manifest declares {verified.VARIANT} for models its config does '
                         f'not select: {sorted(set(variants) - bound)}')
    lock = files['toolchain'].read_text()
    normalized = ''.join(line + '\n' for raw in lock.splitlines()
                         if (line := re.sub(r'\s*#.*$', '', raw)).strip())
    pin = hashlib.sha256(normalized.encode()).hexdigest()[:12]
    expected = next(line.split('=', 1)[1].split('#', 1)[0].strip()
                    for line in lock.splitlines() if line.startswith('IRON_SOURCE_COMMIT='))
    families = sorted(p for p in kernels.iterdir() if p.is_dir())
    if not families:
        raise ValueError('prepared kernels have no families')
    for directory in [kernels, *families]:
        if (directory / '.toolchain-stamp').read_text().strip() != pin:
            raise ValueError(f'prepared kernel/toolchain mismatch: {directory}')
    with tempfile.TemporaryDirectory(prefix='npu-prepared-check-') as tmp:
        root = Path(tmp)
        runtime = root / 'lib'
        runtime.mkdir()
        (runtime / 'libonnxruntime.so.1').symlink_to(files['onnxruntime'])
        env = os.environ | {'LD_LIBRARY_PATH': str(runtime)}
        shutil.copytree(repo / 'scenarios', root / 'scenarios')
        shutil.copy2(files['toolchain'], root / 'toolchain.lock')
        (root / 'artifacts').symlink_to(artifacts, target_is_directory=True)
        (root / 'kernels').symlink_to(kernels, target_is_directory=True)
        all_paths = set()
        for row in dispatch.configured_models(config):
            rel = row['scenario']
            if Path(rel).is_absolute() or '..' in Path(rel).parts:
                raise ValueError('prepared scenarios must be repository-relative')
            scenario = root / rel
            if sha256_file(scenario) != data['scenarios'][rel]:
                raise ValueError(f'prepared scenario changed: {rel}')
            parsed = tomllib.loads(scenario.read_text())
            if not parsed['scenario']['kind']:
                raise ValueError(f'missing scenario kind: {rel}')
            paths = scenario_artifact_paths(parsed)
            all_paths.update(paths)
            for path in paths:
                target = root / path
                if not target.exists() or (target.is_dir() and not any(target.iterdir())):
                    raise ValueError(f'prepared scenario artifact missing: {path}')
                # The staged hop is judged, not just its presence: a prepared deployment must not
                # become the second install that quietly serves a snapshot or a lane payload.
                check_target(artifacts, target)
            arts = parsed.get('artifacts', {})
            for key in ('decode', 'prefill', 'resident'):
                if key not in arts:
                    continue
                meta = root / arts[key] / 'meta.json'
                if validate_iron(meta, expected):
                    raise ValueError(f'prepared IRON provenance mismatch: {meta}')
                built = json.loads(meta.read_text()).get('toolchain', {}).get('hash')
                if built and built != pin:
                    raise ValueError(f'prepared artifact/toolchain mismatch: {meta}')
            if arts.get('decode') and arts.get('prefill'):
                run([sys.executable, repo / 'scripts/check_prefill_arena_pairing.py',
                     root / arts['decode'], root / arts['prefill']])
        rendered = artifact_manifest.build(root, artifacts, drop_covered(all_paths))
        artifact_receipt = root / 'artifacts.manifest.json'
        artifact_receipt.write_text(json.dumps(rendered))
        # Keep stdout reserved for shell assignments.
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            if artifact_manifest.verify(root, artifact_receipt):
                raise ValueError('prepared artifact manifest failed')
        run([files['npu_dev'], 'kernels-manifest', '--check-native-contracts-only',
             '--repo-root', repo, '--engine-root', root, '--config', config, *families], env=env)
        run([files['npu_dev'], 'kernels-verify', '--engine-root', root,
             '--config', config, repo, kernels], env=env)
        run([files['npu'], '--config', config, 'model', 'ls', '--output', 'json'],
            env=env | {'XDNA_ENGINE_ROOT': str(root)})
        # Prepared deployment also requires an already usable standalone runtime;
        # do not discover a missing RUNPATH only after replacing the live binary.
        clean_env = {key: value for key, value in os.environ.items() if key != 'LD_LIBRARY_PATH'}
        run([files['npu'], '--config', config, 'model', 'ls', '--output', 'json'],
            env=clean_env | {'XDNA_ENGINE_ROOT': str(root)})
    return {'BUILT_BIN': str(files['npu']), 'DECLARED_KERNELS_BIN': str(files['npu_dev']),
            'ORT_REAL_SO': str(files['onnxruntime']), 'ENGINE_CONFIG': str(config),
            'ENGINE_ARTIFACTS': str(artifacts), 'PREPARED_KERNELS': str(kernels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--repo', required=True, type=Path)
    args = parser.parse_args()
    assignments = verify(args.manifest, args.repo)
    for name, value in assignments.items():
        print(f'{name}={shlex.quote(value)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
