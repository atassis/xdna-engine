#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Command adapter for accepted Gemma4 proof and CPU packaging inputs."""
import argparse
import json
import os
import sys
from pathlib import Path

import verified_gemma4_data as gemma4
from verified_gemma4_data import VerifiedDataError, materialize_proof


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise ValueError(message)


def _parser():
    parser = _Parser(prog='gemma4_verified_entry.py', allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    proof = commands.add_parser('proof', allow_abbrev=False)
    proof.add_argument('--source', type=Path, required=True)
    proof.add_argument('--accepted-data-root', type=Path, required=True)
    proof.add_argument('--witness', type=Path, required=True)
    proof.add_argument('--expected-witness-sha256', required=True)
    proof.add_argument('--out', type=Path, required=True)
    package = commands.add_parser('package', allow_abbrev=False)
    package.add_argument('--source', type=Path, required=True)
    package.add_argument('--accepted-data-root', type=Path, required=True)
    package.add_argument('--proof-root', type=Path, required=True)
    package.add_argument('--expected-witness-sha256', required=True)
    package.add_argument('--artifacts-root', type=Path, required=True)
    package.add_argument('--build-dir', type=Path, required=True)
    package.add_argument('--out', type=Path, required=True)
    package.add_argument('--tokenizer-out', type=Path, required=True)
    return parser


def _duplicate_option(argv):
    seen = set()
    for value in argv:
        if value.startswith('--'):
            name = value.split('=', 1)[0]
            if name in seen:
                return name
            seen.add(name)
    return None


def _disjoint(left, right, label):
    if left == right or left in right.parents or right in left.parents:
        raise VerifiedDataError('%s must be disjoint: %s and %s' % (label, left, right))


def _fresh_destination(path, label):
    checked = gemma4.declared_path(path, label)
    if checked.exists():
        raise VerifiedDataError(label + ' already exists: ' + str(checked))
    return checked


def _write_cached_sidecar(resident, raw):
    target = gemma4.declared_path(resident / gemma4.SIDECAR, 'resident witness sidecar')
    if target.exists():
        raise VerifiedDataError('resident witness sidecar already exists')
    try:
        with target.open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise VerifiedDataError('cannot write resident witness sidecar: %s' % error) from error


def _package(options):
    """Package only a pinned, verified source snapshot; dispatch imports stay branch-local."""
    source = gemma4._root(options.source, options.accepted_data_root)
    proof = gemma4.declared_path(options.proof_root, 'proof root')
    artifacts = gemma4.declared_path(options.artifacts_root, 'artifacts root')
    build = gemma4.declared_path(options.build_dir, 'build directory')
    out = _fresh_destination(options.out, 'resident output')
    tokenizer_out = _fresh_destination(options.tokenizer_out, 'tokenizer output')
    checkpoint = _fresh_destination(artifacts / 'gemma4-12b-qat' / 'checkpoint',
                                    'checkpoint output')
    if not proof.is_dir():
        raise VerifiedDataError('proof root is not a directory')
    if not isinstance(options.expected_witness_sha256, str) or not gemma4.HEX64.fullmatch(options.expected_witness_sha256):
        raise VerifiedDataError('expected witness SHA256 must be lowercase SHA256')
    compiled = gemma4.declared_path(build / 'rf48C', 'compiled rf48C input')
    if not compiled.is_dir():
        raise VerifiedDataError('build directory lacks compiled rf48C input')
    temporary_base = build.parent
    if temporary_base == source or source in temporary_base.parents:
        raise VerifiedDataError('build temporary base must not be inside source root')
    _disjoint(proof, source, 'proof root and source root')
    for index, left in enumerate((checkpoint, out, tokenizer_out)):
        _disjoint(left, source, 'package destination and source root')
        _disjoint(left, proof, 'package destination and proof root')
        _disjoint(left, compiled, 'package destination and compiled input')
        for right in (checkpoint, out, tokenizer_out)[index + 1:]:
            _disjoint(left, right, 'package destinations')

    sidecar = gemma4.declared_path(proof / gemma4.SIDECAR, 'proof sidecar')
    raw = gemma4._read(sidecar, gemma4.MAX_WITNESS_BYTES, 'proof sidecar')
    if gemma4.stream_hash_bytes(raw) != options.expected_witness_sha256:
        raise VerifiedDataError('proof sidecar SHA256 differs from expected value')
    document = gemma4._validate_witness(gemma4._json(raw, str(sidecar)))
    gemma4.verify_source(source, document, accepted_data_root=options.accepted_data_root)

    # model_artifact_dispatch imports packaging dependencies; proof-only use must not import them.
    from model_artifact_dispatch import package_gemma4
    repository = gemma4.declared_path(Path(__file__).parent.parent.absolute(), 'repository root')
    package_gemma4(repository, artifacts, build, out, source / 'tokenizer', tokenizer_out,
                   source_store=source / 'store', source_rf_stack=source / 'rf_stack')
    _write_cached_sidecar(out, raw)
    verified = gemma4.verify_published_data(out, tokenizer_out, checkpoint,
                                            expected_witness_sha256=options.expected_witness_sha256)
    return {'model': verified['model'], 'variant': verified['variant'],
            'witness_sha256': options.expected_witness_sha256,
            'payload_digest': verified['payload_digest'],
            'resident': str(out), 'tokenizer': str(tokenizer_out),
            'checkpoint': str(checkpoint)}


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    duplicate = _duplicate_option(arguments)
    if duplicate:
        print('error: repeated option: ' + duplicate, file=sys.stderr)
        return 2
    parser = _parser()
    try:
        options = parser.parse_args(arguments)
    except ValueError as error:
        print('error: ' + str(error), file=sys.stderr)
        return 2
    try:
        if options.command == 'proof':
            summary = materialize_proof(
                options.source, options.witness, options.out,
                accepted_data_root=options.accepted_data_root,
                expected_witness_sha256=options.expected_witness_sha256)
        else:
            summary = _package(options)
    except (VerifiedDataError, ValueError, OSError) as error:
        print('error: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
