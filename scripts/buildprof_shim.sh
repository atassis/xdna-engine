#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# AIECC_PATH shim for the S0 build profile: runs the real aiecc with --profile and keeps one log
# per invocation. Queries (--version, --help) pass through untouched.
set -uo pipefail
real="${BUILDPROF_REAL_AIECC:?BUILDPROF_REAL_AIECC must name the real aiecc}"
dir="${BUILDPROF_DIR:?BUILDPROF_DIR must name the log directory}"
case " $* " in *" --version "* | *" --aie-version "* | *" --help "*) exec "$real" "$@" ;; esac
mkdir -p "$dir"
log="$dir/aiecc-$(date +%s%N)-$$.log"
{ printf 'argv: %s\n' "$*"; printf 'cwd: %s\n' "$PWD"; } > "$log"
start=$(date +%s%N)
"$real" "$@" --profile --no-progress 2> "$log.err"
rc=$?
cat "$log.err" >&2
cat "$log.err" >> "$log"
rm -f "$log.err"
printf 'wall_ms: %d\nrc: %d\n' $(( ($(date +%s%N) - start) / 1000000 )) "$rc" >> "$log"
exit "$rc"
