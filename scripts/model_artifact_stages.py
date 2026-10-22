# SPDX-License-Identifier: Apache-2.0
"""Verified stage frontiers for model_artifact_dispatch's existing commands.

Published data is reused by private copy, never as a writable live alias. Only
non-published output views survive in the canonical build state; compiler work
files outside those views remain disposable transaction scratch.
"""
import fnmatch
import hashlib
import inspect
import json
import os
import shutil
import stat
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import uuid
from pathlib import Path


@dataclass(frozen=True)
class View:
    root: str
    patterns: tuple[str, ...] = ()
    require_each: bool = False


@dataclass(frozen=True)
class Stage:
    inputs: tuple[View, ...] = ()
    outputs: tuple[View, ...] = ()
    dependencies: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    sdk_python: str = ''
    sdk_packages: tuple[str, ...] = ()
    retain: bool = False
    scratch: tuple[str, ...] = ()
    toolchain: bool = False


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON field {key} in {path}')
            result[key] = value
        return result
    if path.is_symlink():
        raise RuntimeError(f'symlinked stage record: {path}')
    return json.loads(path.read_text(), object_pairs_hook=pairs)


def write_json(path, value):
    if path.is_symlink():
        raise RuntimeError(f'symlinked stage record: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with temporary.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def files(view, *, outputs=False):
    root = Path(view.root)
    if not root.exists():
        raise FileNotFoundError(f'required stage view is missing: {root}')
    selected = set()
    if view.patterns:
        for pattern in view.patterns:
            if Path(pattern).is_absolute() or '..' in Path(pattern).parts:
                raise ValueError(f'unsafe stage view pattern: {pattern}')
            matches = list(root.glob(pattern))
            if outputs and view.require_each and not matches:
                raise FileNotFoundError(f'required stage output is missing: {root / pattern}')
            for match in matches:
                selected.update(match.rglob('*') if match.is_dir() else (match,))
    else:
        selected.update(root.rglob('*') if root.is_dir() else (root,))
    result = sorted(p for p in selected if p.is_file() and p.name != '.model-recipe.json')
    if not result:
        raise FileNotFoundError(f'empty stage view: {root}, {view.patterns}')
    return result


def sdk_identity(python, packages):
    probe = (
        'import hashlib,json;from importlib.metadata import distribution;'
        f'names={packages!r};'
        "print(json.dumps({name:{'version':(d:=distribution(name)).version,"
        "'record':hashlib.sha256((d.read_text('RECORD') or d.read_text('METADATA') or '').encode()).hexdigest()}"
        ' for name in names},sort_keys=True))'
    )
    try:
        return subprocess.check_output([python, '-c', probe], text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.output.strip() if isinstance(error, subprocess.CalledProcessError) and error.output else str(error)
        raise RuntimeError(f'SDK identity probe failed for {python}: {detail}') from error


def toolchain_identity(item, repo):
    env = os.environ | item.env | {'TOOLCHAIN_RESOLVE_ONLY': '1'}
    try:
        instance = Path(subprocess.check_output(['bash', str(repo / 'scripts/toolchain_up.sh')],
                                               cwd=repo, env=env, text=True, stderr=subprocess.STDOUT).strip())
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.output.strip() if isinstance(error, subprocess.CalledProcessError) and error.output else str(error)
        raise RuntimeError(f'{item.step}: pinned toolchain is not ready (read-only preflight): {detail}') from error
    if not instance.is_absolute() or not instance.is_dir():
        raise RuntimeError(f'{item.step}: invalid resolved toolchain instance: {instance}')
    venv = Path(env['VENV_IRON'])
    peano = venv / 'lib/python3.14/site-packages/llvm-aie'
    for variable, expected in {'RF_INST': instance, 'AIECC_PATH': instance / 'bin/aiecc',
                               'PEANO_INSTALL_DIR': peano}.items():
        if env.get(variable) and Path(env[variable]).resolve() != expected.resolve():
            raise RuntimeError(f'{item.step}: {variable} overrides the declared pinned toolchain: {env[variable]}')
    roots = [instance / name for name in ('bin', 'python', 'build/include', 'build/lib')]
    roots += [peano / name for name in ('bin', 'include', 'lib')]
    manifest, hashed = {}, {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f'{item.step}: missing toolchain input tree: {root}')
        for directory, dirs, names in os.walk(root, followlinks=True):
            parent = Path(directory)
            # SDK trees may alias installed Python sources, but must not contain directory cycles.
            ancestors = {entry.resolve() for entry in (parent, *parent.parents) if entry.is_relative_to(root)}
            dirs[:] = sorted(name for name in dirs if name not in {'.git', '__pycache__', '.pytest_cache'})
            for name in dirs:
                if (parent / name).resolve() in ancestors:
                    raise ValueError(f'toolchain source symlink cycle: {parent / name}')
            for name in sorted(names):
                path = parent / name
                if path.suffix in {'.pyc', '.pyo'} or not path.is_file():
                    continue
                info = path.stat()
                identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                if identity not in hashed:
                    hashed[identity] = file_hash(path)
                manifest[str(path)] = {'sha256': hashed[identity]}
    return {'instance': str(instance), 'peano': str(peano), 'files': manifest}


def source_snapshot(item, repo, *, symbols=None):
    contract = item.stage
    sources = {}
    for raw in contract.sources:
        root = Path(raw)
        if not root.is_absolute():
            root = repo / root
        if not root.exists():
            raise FileNotFoundError(f'{item.step}: missing producer source: {root}')
        candidates = (root,) if root.is_file() else sorted(root.rglob('*'))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if root.is_dir():
                parts = candidate.relative_to(root).parts
                if (any(part.startswith('.') or part in {'__pycache__', 'tests', 'fixtures'} for part in parts)
                        or candidate.name.startswith('test_')
                        or candidate.suffix in {'.pyc', '.pyo', '.md', '.log'}):
                    continue
            label = str(candidate.relative_to(repo)) if candidate.is_relative_to(repo) else str(candidate)
            sources[label] = file_hash(candidate)
    selected_symbols = {name: inspect.getsource(symbols[name]) for name in contract.symbols}
    environment = os.environ | item.env
    # PATH often contains per-shell launcher directories; bind the resolved producer instead.
    names = {'LD_LIBRARY_PATH', 'PYTHONPATH'}
    for pattern in contract.environment:
        names.update(name for name in environment if fnmatch.fnmatchcase(name, pattern))
    sdk = None
    if contract.sdk_python:
        python = shutil.which(contract.sdk_python, path=environment.get('PATH'))
        sdk = {'python': python, 'sha256': file_hash(python) if python else None,
               'distributions': sdk_identity(contract.sdk_python, contract.sdk_packages)}
    inputs = {}
    for view in contract.inputs:
        for candidate in files(view):
            inputs[str(candidate)] = file_hash(candidate)
    executable = shutil.which(item.argv[0], path=environment.get('PATH'))
    executable_identity = {'resolved': executable, 'sha256': file_hash(executable) if executable else None}
    return {'command': {'step': item.step, 'argv': item.argv, 'env': item.env},
            'executable': executable_identity, 'sources': sources, 'symbols': selected_symbols,
            'environment': {name: environment.get(name) for name in sorted(names)},
            'sdk': sdk, 'toolchain': toolchain_identity(item, repo) if contract.toolchain else None,
            'inputs': inputs}


def source_identity(item, repo, *, symbols=None):
    return digest(source_snapshot(item, repo, symbols=symbols))


def scopes_overlap(left, right):
    # Reject ambiguous glob intersections, including files not present yet.
    for a, b in zip(Path(left).parts, Path(right).parts):
        if '**' in (a, b):
            return True
        wild_a, wild_b = any(c in a for c in '*?['), any(c in b for c in '*?[')
        if not wild_a and not wild_b:
            if a != b:
                return False
        elif not wild_a or not wild_b:
            if not fnmatch.fnmatchcase(a, b) and not fnmatch.fnmatchcase(b, a):
                return False
        else:
            prefixes = [value[:min((value.find(c) for c in '*?[' if c in value), default=len(value))] for value in (a, b)]
            if not prefixes[0].startswith(prefixes[1]) and not prefixes[1].startswith(prefixes[0]):
                return False
            suffixes = [value[max(value.rfind(c) for c in '*?[]') + 1:] for value in (a, b)]
            if not suffixes[0].endswith(suffixes[1]) and not suffixes[1].endswith(suffixes[0]):
                return False
    return True


def rebase_stage(contract, relocate):
    if contract is None:
        return None
    def view(value):
        return replace(value, root=relocate(value.root))
    return replace(contract, inputs=tuple(view(v) for v in contract.inputs),
                   outputs=tuple(view(v) for v in contract.outputs),
                   scratch=tuple(relocate(value) for value in contract.scratch))


class StageRunner:
    VERSION = 1

    def __init__(self, recipe, staged, repo, workspace, state, artifacts, build, execute, *, symbols=None, validate=None):
        self.recipe, self.staged, self.repo = recipe, staged, Path(repo)
        self.workspace, self.state = Path(workspace), Path(state)
        self.artifacts, self.build = Path(artifacts), Path(build)
        self.execute, self.symbols = execute, symbols or {}
        self.validate = validate
        self.commands = {item.step: item for item in recipe.commands}
        self.staged_commands = {item.step: item for item in staged.commands}
        self.records, self.identities, self.finished = {}, {}, set()
        self.source_records = {}
        self.actions = {}
        self.pending = None

    def _safe(self, path, root):
        path, root = Path(path), Path(root)
        if '..' in path.parts or not path.is_relative_to(root):
            raise ValueError(f'stage path escapes owned root: {path}')
        for entry in (path, *path.parents):
            if entry.is_symlink():
                raise RuntimeError(f'symlink in stage path: {entry}')
            if entry == root:
                break
        return path

    def _validate(self):
        self._safe(self.workspace, self.build)
        self._safe(self.state, self.build)
        if self.state.is_relative_to(self.workspace) or self.workspace.is_relative_to(self.state):
            raise ValueError('stage state and transaction workspace must be disjoint')
        if len(self.commands) != len(self.recipe.commands):
            raise ValueError('duplicate stage names')
        for name, item in self.commands.items():
            if not name or name in {'.', '..'} or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in name):
                raise ValueError(f'unsafe stage name: {name}')
            if item.stage is None:
                raise ValueError(f'{name}: missing stage contract')
            for dep in item.stage.dependencies:
                if dep not in self.commands:
                    raise ValueError(f'{name}: missing dependency {dep}')
            for view in self.staged_commands[name].stage.outputs:
                self._safe(view.root, self.workspace)
                if Path(view.root) == self.workspace:
                    raise ValueError('output cannot own the entire stage workspace')
            for scratch in self.staged_commands[name].stage.scratch:
                self._safe(scratch, self.workspace)
        visiting, visited = set(), set()
        def walk(name):
            if name in visiting:
                raise ValueError(f'stage dependency cycle: {name}')
            if name in visited:
                return
            visiting.add(name)
            for dep in self.commands[name].stage.dependencies:
                walk(dep)
            visiting.remove(name)
            visited.add(name)
        for name in self.commands:
            walk(name)
        owned = []
        for name, item in self.staged_commands.items():
            for view in item.stage.outputs:
                # A patterned view owns its matched children, not the shared parent.
                scopes = tuple(Path(view.root) / pattern for pattern in view.patterns) if view.patterns else (Path(view.root),)
                for scope in scopes:
                    self._safe(scope, self.workspace)
                    for other, previous in owned:
                        if name != other and scopes_overlap(scope, previous):
                            raise ValueError(f'stage output ownership overlap: {other}, {name}: {scope}')
                    owned.append((name, scope))
        for name, item in self.staged_commands.items():
            for raw in item.stage.scratch:
                scratch = Path(raw)
                if scratch == self.workspace:
                    raise ValueError('scratch cannot own the entire stage workspace')
                for other, scope in owned:
                    if other != name and scopes_overlap(scratch, scope):
                        raise ValueError(f'scratch overlaps another stage output: {name}, {other}')
                for command in self.staged_commands.values():
                    for view in command.stage.inputs:
                        root = Path(view.root)
                        if scratch.is_relative_to(root) or root.is_relative_to(scratch):
                            raise ValueError(f'scratch overlaps source input: {scratch}, {root}')

    def _owner(self):
        return {'version': self.VERSION, 'model': self.recipe.model, 'workspace': str(self.workspace),
                'artifacts': str(self.artifacts), 'build': str(self.build), 'outputs': list(self.recipe.outputs),
                'ownership': {name: [asdict(view) for view in item.stage.outputs] for name, item in self.commands.items()}}

    def _open(self):
        self._validate()
        pending = self.state / 'pending.json'
        if pending.exists() or pending.is_symlink():
            raise RuntimeError(f'publication recovery requires inspection before retry: {pending}')
        owner = json.loads(json.dumps(self._owner()))
        for root in (self.state, self.workspace):
            self._safe(root, root)
            marker = root / 'owner.json'
            if root.exists():
                if not marker.is_file() or read_json(marker) != owner:
                    raise RuntimeError(f'unowned or incompatible stage directory: {root}')
            else:
                root.mkdir(parents=True)
                write_json(marker, owner)
        for name in self.commands:
            path = self.state / f'{name}.json'
            if path.exists() or path.is_symlink():
                record = read_json(path)
                if not isinstance(record, dict) or record.get('version') != self.VERSION or record.get('stage') != name:
                    raise ValueError(f'invalid stage record: {path}')
                self.records[name] = record

    def _normalize(self, value, reverse=False):
        a, b = str(self.workspace / 'artifacts'), str(self.artifacts)
        if reverse:
            a, b = b, a
        if isinstance(value, str):
            return b + value[len(a):] if value == a or value.startswith(a + os.sep) else value
        if isinstance(value, list):
            return [self._normalize(v, reverse) for v in value]
        if isinstance(value, dict):
            return {k: self._normalize(v, reverse) for k, v in value.items()}
        return value

    def _fingerprint(self, path):
        sha = file_hash(path)
        identity = digest(self._normalize(read_json(path))) if path.name == 'meta.json' else sha
        return {'sha256': sha, 'identity': identity, 'executable': bool(path.stat().st_mode & stat.S_IXUSR)}

    def _snapshot(self, name):
        manifest = {}
        for view in self.staged_commands[name].stage.outputs:
            for path in files(view, outputs=True):
                self._safe(path, self.workspace)
                manifest[str(path.relative_to(self.workspace))] = self._fingerprint(path)
        return manifest

    def _output_identity(self, manifest):
        return digest({key: [value['identity'], value['executable']] for key, value in manifest.items()})

    def _candidate(self, name):
        parents = {}
        for dep in self.commands[name].stage.dependencies:
            key = self._candidate(dep)
            record = self.records.get(dep)
            if record is None or key != record['key']:
                return None
            parents[dep] = {'key': key, 'outputs': record['output_identity']}
        if name not in self.identities:
            self.source_records[name] = source_snapshot(self.commands[name], self.repo, symbols=self.symbols)
            self.identities[name] = digest(self.source_records[name])
        return digest({'version': self.VERSION, 'source': self.identities[name], 'dependencies': parents})

    def _owns(self, name, rel):
        target = self.workspace / rel
        for view in self.staged_commands[name].stage.outputs:
            root = Path(view.root)
            if not target.is_relative_to(root):
                continue
            if not view.patterns:
                return True
            parts = target.relative_to(root).parts
            for pattern in view.patterns:
                expected = Path(pattern).parts
                if len(parts) >= len(expected) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts, expected)):
                    return True
        return False

    def _valid(self, record, *, prepared=False):
        name = record.get('stage')
        if name not in self.commands:
            raise ValueError(f'unknown stage record: {name}')
        manifest = record.get('files')
        if not isinstance(manifest, dict) or record.get('output_identity') != self._output_identity(manifest):
            raise ValueError('invalid stage output manifest')
        locations = record.get('locations')
        if not isinstance(locations, dict) or set(locations) != set(manifest):
            raise ValueError('invalid stage output locations')
        if record.get('expired') and manifest:
            if not self._transient(name):
                raise ValueError(f'{name}: retained output cannot be marked expired')
            return False
        for rel, expected in manifest.items():
            if Path(rel).is_absolute() or '..' in Path(rel).parts or not self._owns(name, rel):
                raise ValueError(f'unsafe or unowned stage output: {rel}')
            path = Path(locations[rel])
            allowed = (self.workspace / rel, self.state / 'frontiers' / record['stage'] / rel)
            if rel.startswith('artifacts/'):
                allowed += (self.artifacts / Path(rel).relative_to('artifacts'),)
            if prepared:
                allowed += (self.state / 'pending-frontiers' / name / rel,)
            if path not in allowed:
                raise ValueError(f'stage output location escapes its owner: {path}')
            boundary = self.artifacts if path.is_relative_to(self.artifacts) else self.state if path.is_relative_to(self.state) else self.workspace
            self._safe(path, boundary)
            if not path.is_file() or self._fingerprint(path) != expected:
                return False
        return True

    def _hydrate(self, name):
        record = self.records[name]
        for rel, raw in record['locations'].items():
            source = Path(raw)
            target = self._safe(self.workspace / rel, self.workspace)
            if source == target:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            os.chmod(target, 0o755 if record['files'][rel]['executable'] else 0o644)
            if target.name == 'meta.json':
                original = read_json(target)
                restored = self._normalize(original, reverse=True)
                if restored != original:
                    target.write_text(json.dumps(restored, indent=2) + '\n')
        manifest = self._snapshot(name)
        if self._output_identity(manifest) != record['output_identity']:
            raise RuntimeError(f'{name}: restored frontier identity changed')
        self.records[name] = {**record, 'files': manifest,
                              'locations': {rel: str(self.workspace / rel) for rel in manifest}}
        write_json(self.state / f'{name}.json', self.records[name])

    def _clear(self, name):
        for view in self.staged_commands[name].stage.outputs:
            root = self._safe(view.root, self.workspace)
            targets = [match for pattern in view.patterns for match in root.glob(pattern)] if view.patterns else [root]
            for target in targets:
                self._safe(target, self.workspace)
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()

    def _need(self, name):
        if name in self.finished:
            return
        candidate, record = self._candidate(name), self.records.get(name)
        if record is not None and candidate == record['key'] and self._valid(record):
            self._hydrate(name)
            if self.validate is not None:
                self.validate(self.staged, name)
            self.finished.add(name)
            self.actions[name] = 'verified-reuse'
            return
        item = self.commands[name]
        for dep in item.stage.dependencies:
            self._need(dep)
        # A rebuilt parent may have a different output identity under the same source key.
        candidate = self._candidate(name)
        if candidate is None:
            raise RuntimeError(f'{name}: dependency did not complete')
        before = {dep: self._snapshot(dep) for dep in self.finished}
        for scratch in self.staged_commands[name].stage.scratch:
            target = self._safe(scratch, self.workspace)
            if target.exists():
                shutil.rmtree(target)
        self._clear(name)
        self.execute(replace(self.staged, commands=(self.staged_commands[name],)), self.repo)
        if self.validate is not None:
            self.validate(self.staged, name)
        after_source = source_identity(item, self.repo, symbols=self.symbols)
        if after_source != self.identities[name]:
            raise RuntimeError(f'{name}: producer inputs changed during execution')
        for dep, expected in before.items():
            if self._snapshot(dep) != expected:
                raise RuntimeError(f'{name}: mutated predecessor {dep}')
        manifest = self._snapshot(name)
        record = {'version': self.VERSION, 'stage': name, 'key': candidate,
                  'production': {'id': uuid.uuid4().hex, 'completed_at': datetime.now(timezone.utc).isoformat()},
                  'producer_source': self.identities[name], 'producer': self.source_records[name],
                  'producer_outputs': manifest,
                  'output_identity': self._output_identity(manifest), 'files': manifest,
                  'locations': {rel: str(self.workspace / rel) for rel in manifest}}
        self.records[name] = record
        write_json(self.state / f'{name}.json', record)
        self.finished.add(name)
        self.actions[name] = 'executed'

    def run(self):
        self._open()
        targets = []
        for name, item in self.commands.items():
            for view in item.stage.outputs:
                if any(Path(view.root).is_relative_to(Path(raw)) or Path(raw).is_relative_to(Path(view.root))
                       for raw in self.recipe.outputs):
                    targets.append(name)
                    break
        if not targets:
            raise ValueError('stage contracts do not own any published output')
        for name in targets:
            self._need(name)

    def _transient(self, name):
        contract = self.commands[name].stage
        return not contract.retain and all(not Path(view.root).is_relative_to(self.artifacts) for view in contract.outputs)

    def prepare_settlement(self, key=None):
        pending = {}
        for name, record in self.records.items():
            if name not in self.finished:
                continue
            manifest = self._snapshot(name)
            if self._output_identity(manifest) != record['output_identity']:
                raise RuntimeError(f'{name}: publication changed frontier identity')
            locations, prepared = {}, {}
            for rel in manifest:
                source = self.workspace / rel
                if rel.startswith('artifacts/'):
                    destination = self.artifacts / Path(rel).relative_to('artifacts')
                elif self.commands[name].stage.retain:
                    destination = self._safe(self.state / 'frontiers' / name / rel, self.state)
                    fresh = self._safe(self.state / 'pending-frontiers' / name / rel, self.state)
                    fresh.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, fresh)
                    os.chmod(fresh, 0o755 if manifest[rel]['executable'] else 0o644)
                    prepared[rel] = str(fresh)
                else:
                    destination = None
                locations[rel] = str(destination) if destination is not None else None
            pending[name] = {**record, 'files': manifest, 'locations': locations,
                             'prepared_locations': prepared, 'expired': self._transient(name)}
        self.pending = pending
        if key is not None:
            write_json(self.state / 'pending.json', {'version': self.VERSION, 'recipe_key': key, 'records': pending})

    def resume_settlement(self, key):
        self._validate()
        path = self.state / 'pending.json'
        if not path.exists() and not path.is_symlink():
            return
        self._safe(self.state / 'owner.json', self.state)
        if read_json(self.state / 'owner.json') != json.loads(json.dumps(self._owner())):
            raise RuntimeError(f'unowned stage settlement: {self.state}')
        pending = read_json(path)
        if pending.get('version') != self.VERSION or pending.get('recipe_key') != key:
            raise RuntimeError(f'publication recovery requires inspection: {path}')
        self.pending = pending['records']
        self.settle()

    def publication_record(self):
        return {name: {**{key: value for key, value in record.items() if key != 'prepared_locations'},
                       'action': self.actions.get(name, 'lineage-only')}
                for name, record in (self.records | (self.pending or {})).items()}

    def settle(self):
        self._validate()
        if self.pending is None:
            raise RuntimeError('stage settlement was not prepared before publication')
        for name, record in self.pending.items():
            if name not in self.commands or record.get('stage') != name:
                raise ValueError(f'unknown stage settlement: {name}')
            for rel, source in record.get('prepared_locations', {}).items():
                if not self.commands[name].stage.retain or rel.startswith('artifacts/'):
                    raise ValueError(f'{name}: unowned prepared frontier: {rel}')
                if Path(source) != self.state / 'pending-frontiers' / name / rel:
                    raise ValueError(f'{name}: prepared frontier location escapes its owner: {source}')
                self._safe(source, self.state)
                single = {**record, 'files': {rel: record['files'][rel]},
                          'output_identity': self._output_identity({rel: record['files'][rel]}),
                          'locations': {rel: record['locations'][rel]}}
                if self._valid(single):
                    continue
                staged = {**single, 'locations': {rel: source}}
                if not self._valid(staged, prepared=True):
                    raise RuntimeError(f'{name}: prepared frontier did not verify: {rel}')
                target = self._safe(record['locations'][rel], self.state)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
            if not (record.get('expired') and self._transient(name)) and not self._valid(record):
                raise RuntimeError(f'{name}: published frontier did not verify')
            committed = {key: value for key, value in record.items() if key != 'prepared_locations'}
            write_json(self.state / f'{name}.json', committed)
            self.records[name] = committed
        for name, record in self.pending.items():
            for rel, source in record.get('prepared_locations', {}).items():
                path = self._safe(source, self.state)
                if path.exists():
                    if self._fingerprint(path) != record['files'][rel]:
                        raise RuntimeError(f'{name}: prepared frontier changed before expiry: {path}')
                    path.unlink()
        path = self.state / 'pending.json'
        if path.exists():
            path.unlink()
