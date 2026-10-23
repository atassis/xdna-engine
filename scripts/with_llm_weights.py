#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a raw-weight consumer with an explicit dump or a local, command-scoped export.

Use --spec MODEL [--checkpoint-dir PATH] -- COMMAND... . The child receives WEIGHTS;
for a Python CLI with --weights, use a shell child to expand "$WEIGHTS" there.
Explicit WEIGHTS is caller-owned and never collected. Otherwise the unchanged
exporter reads the local checkpoint, holding its recipe resource lease, and the
raw export lives under XDNA_BUILD only until the consumer stops. Consumer outputs
must be outside that transaction; keep the checkpoint and exporter for recovery.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from lib.data_root import XDNA_BUILD, XDNA_DATA
from model_artifact_dispatch import resource_locks


REPO = Path(__file__).resolve().parents[1]


def run_child(argv, env):
    # An interrupted wrapper must not collect files while its exporter/consumer still writes.
    process = subprocess.Popen(argv, env=env, start_new_session=True)
    try:
        status = process.wait()
        return status if status >= 0 else 128 - status
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The leader may exit before a descendant that ignored TERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a consumer command is required after --')
    env = os.environ.copy()
    if env.get('WEIGHTS'):
        if not Path(env['WEIGHTS']).is_dir():
            parser.error(f"explicit WEIGHTS directory does not exist: {env['WEIGHTS']}")
        return run_child(command, env)
    checkpoint = args.checkpoint_dir or XDNA_DATA / 'model-inputs' / args.spec / 'checkpoint'
    with resource_locks((checkpoint,)):
        if not checkpoint.is_dir() or not any(checkpoint.glob('*.safetensors')):
            parser.error(f'local checkpoint with safetensors required: {checkpoint}')
        root = XDNA_BUILD / 'llm-weight-transactions'
        status = run_child(['bash', '-c', 'source "$1"; require_disk_backed "$2" "transient LLM weights"',
                            'bash', str(REPO / 'scripts/require_disk_backed.sh'), str(root)], env)
        if status:
            return status
        with tempfile.TemporaryDirectory(prefix='weights-', dir=root) as temporary:
            weights = Path(temporary) / 'weights'
            status = run_child([sys.executable, str(REPO / 'scripts/dump_llm_weights.py'),
                                '--spec', args.spec, '--checkpoint-dir', str(checkpoint),
                                '--out', str(weights)], env)
            if status:
                return status
            env['WEIGHTS'] = str(weights)
            return run_child(command, env)


def terminate(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
