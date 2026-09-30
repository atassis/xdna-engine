#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# S0: cold-build every recipe in a TSV once, serially, with per-stage aiecc profiles.
# usage: run_s0.sh <recipes.tsv> [s0-dir]    (instance from MLIR_AIE_INSTANCE)
#
# REPO defaults to this checkout. Recipes read gitignored inputs (weights, .venv-iron,
# artifacts/) that a worktree may not carry -- override REPO to build from a checkout
# that has them.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$here/.." && pwd)}"
. "$here/lib/data_root.sh"   # -> XDNA_SCRATCH, XDNA_DATA
recipes="${1:?usage: run_s0.sh <recipes.tsv> [s0-dir]}"
S0="${2:-$XDNA_SCRATCH/s0/$(date +%Y%m%d-%H%M)}"
xdna_mkdir "$XDNA_DATA"
. "$here/amd_paths.sh"
aiecc_resolve "${MLIR_AIE_INSTANCE:?set MLIR_AIE_INSTANCE to the pinned instance}" || exit 1
: "${PEANO_INSTALL_DIR:?set PEANO_INSTALL_DIR explicitly; aiecc falls back to \$PATH otherwise}"
export BUILDPROF_REAL_AIECC="$AIECC_PATH" AIECC_PATH="$here/buildprof_shim.sh" AIE_AIECC_CACHE=0
mkdir -p "$S0"
while IFS=$'\t' read -r name cmd; do
  { [ -z "$name" ] || [ "${name:0:1}" = "#" ]; } && continue
  d="$S0/$name"
  if [ "$cmd" = "NORECIPE" ]; then echo "[s0] $name: NORECIPE, skipped"; mkdir -p "$d"; touch "$d/NORECIPE"; continue; fi
  avail_kb=$(df -kP "$XDNA_DATA" | awk 'NR==2 {print $4}')
  if [ "${avail_kb:-0}" -lt $((8 * 1024 * 1024)) ]; then
    echo "[s0] $name: skipped, $XDNA_DATA under 8 GB free (${avail_kb} KB)"; continue
  fi
  mkdir -p "$d/work" "$d/out" "$d/npu-cache"
  echo "[s0] $name"
  # Cold: a fresh kernel PCH/object cache per recipe and no ccache; KEEP_WORK already defeats
  # IRON's mtime MLIR cache and AIE_AIECC_CACHE=0 the whole-aiecc cache.
  # KCC_LOG/AIE_KERNEL_COMPILER_LAUNCHER/KCC_NEXT= route every kernel .cc compile through
  # kcc_log_launcher.sh (fleet enumeration, F0.2) instead of running it straight.
  OUT="$d/out" BUILDPROF_DIR="$d/aiecc-logs" KEEP_WORK="$d/work" \
    NPU_CACHE_HOME="$d/npu-cache" CCACHE_DISABLE=1 \
    KCC_LOG="$d/kcc.jsonl" AIE_KERNEL_COMPILER_LAUNCHER="$here/kcc_log_launcher.sh" KCC_NEXT= \
    systemd-run --user --scope --quiet -p MemoryMax=12G \
    nice -n 10 /usr/bin/time -v -o "$d/time.txt" bash -c "cd '$REPO' && $cmd" > "$d/build.log" 2>&1
  echo "rc=$?" >> "$d/time.txt"
  # Keep each design's aiecc input (relative path, so multiple designs per build stay
  # distinct); everything else under work/out/npu-cache/blobs is multi-GB build scratch
  # that has filled $XDNA_DATA before and is not needed after the mlir copy.
  mkdir -p "$d/mlir"
  (cd "$d/work" 2>/dev/null && find . -path '*/aie.mlir' -exec bash -c \
    'mkdir -p "$1/mlir/$(dirname "$0")" && cp "$0" "$1/mlir/$0"' {} "$d" \; )
  rm -rf "$d/work" "$d/out" "$d/npu-cache" "$d/blobs"
done < "$recipes"
python3 "$here/buildprof_summarize.py" "$S0" > "$S0/summary.tsv"
echo "[s0] summary: $S0/summary.tsv"
