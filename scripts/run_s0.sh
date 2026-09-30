#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# S0: cold-build every recipe in a TSV once, serially, with per-stage aiecc profiles.
# usage: run_s0.sh <recipes.tsv> [s0-dir]    (instance from MLIR_AIE_INSTANCE)
#
# REPO defaults to the MAIN xdna-engine checkout, not this worktree: recipes read gitignored
# inputs (weights, .venv-iron, artifacts/) that a worktree does not carry. Override REPO to build
# from a different checkout of the same commit.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$here/../.." && pwd)"
REPO="${REPO:-$WS/xdna-engine}"
recipes="${1:?usage: run_s0.sh <recipes.tsv> [s0-dir]}"
S0="${2:-/mnt/data/xdna/scratch/s0/$(date +%Y%m%d-%H%M)}"
. "$here/amd_paths.sh"
aiecc_resolve "${MLIR_AIE_INSTANCE:?set MLIR_AIE_INSTANCE to the pinned instance}" || exit 1
: "${PEANO_INSTALL_DIR:?set PEANO_INSTALL_DIR explicitly; aiecc falls back to \$PATH otherwise}"
export BUILDPROF_REAL_AIECC="$AIECC_PATH" AIECC_PATH="$here/buildprof_shim.sh" AIE_AIECC_CACHE=0
mkdir -p "$S0"
while IFS=$'\t' read -r name cmd; do
  { [ -z "$name" ] || [ "${name:0:1}" = "#" ]; } && continue
  d="$S0/$name"
  if [ "$cmd" = "NORECIPE" ]; then echo "[s0] $name: NORECIPE, skipped"; mkdir -p "$d"; touch "$d/NORECIPE"; continue; fi
  mkdir -p "$d/work" "$d/out"
  echo "[s0] $name"
  OUT="$d/out" BUILDPROF_DIR="$d/aiecc-logs" KEEP_WORK="$d/work" \
    systemd-run --user --scope --quiet -p MemoryMax=12G \
    nice -n 10 /usr/bin/time -v -o "$d/time.txt" bash -c "cd '$REPO' && $cmd" > "$d/build.log" 2>&1
  echo "rc=$?" >> "$d/time.txt"
done < "$recipes"
python3 "$here/buildprof_summarize.py" "$S0" > "$S0/summary.tsv"
echo "[s0] summary: $S0/summary.tsv"
