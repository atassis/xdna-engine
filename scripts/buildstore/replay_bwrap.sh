#!/usr/bin/env bash
# scripts/buildstore/replay_bwrap.sh
# SPDX-License-Identifier: Apache-2.0
# Thin wrapper: the mechanism moved to replay.py (bwrap aborts past 9000 argv entries, one
# recorded path short of a real recipe's ~9.6k). See replay.py's module docstring.
# usage: replay_bwrap.sh <manifest.json> <recorded-out-dir> <scratch>
exec python3 "$(dirname "$0")/replay.py" "$@"
