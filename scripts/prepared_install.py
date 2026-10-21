#!/usr/bin/env python3
"""Verify an operator-reviewed prepared install manifest without building or publishing.

The manifest is local authorization, not a signature: its hashes must come from the
accepted preparation evidence, not be regenerated to bless changed inputs. Existing
recipe keys remain historical; output digests and provenance are checked independently
of the dispatcher's current, deliberately broad source-authority cache key.
"""
import argparse
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

import model_artifact_dispatch as dispatch
from check_iron_artifact_provenance import validate as validate_iron
from lib import artifact_manifest
from lib.artifact_manifest import sha256_file
from lib.collect_served_artifacts import drop_covered, scenario_artifact_paths


def verify_receipt(recipe, witness):
    receipt = dispatch.manifest_path(recipe)
    if sha256_file(receipt) != witness['sha256']:
        raise ValueError(f'{recipe.model}: prepared receipt changed')
    record = json.loads(receipt.read_text())
    key = witness['recipe_key']
    if (not re.fullmatch(r'[0-9a-f]{64}', key)
            or record.get('model') != recipe.model or record.get('scenario') != recipe.scenario
            or not dispatch.cache_hit(recipe, key)):
        raise ValueError(f'{recipe.model}: prepared receipt/output verification failed')


def checked_file(entry):
    path = Path(entry['path'])
    if not path.is_absolute() or sha256_file(path) != entry['sha256']:
        raise ValueError(f'prepared file missing or changed: {path}')
    return path


def run(argv, **kwargs):
    subprocess.run(list(map(str, argv)), check=True, stdout=sys.stderr, **kwargs)


def verify(manifest, repo):
    data = json.loads(manifest.read_text())
    if data['version'] != 1:
        raise ValueError('unsupported prepared manifest version')
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
    for recipe in {item.model: item for item in recipes}.values():
        verify_receipt(recipe, data['receipts'][recipe.model])
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
