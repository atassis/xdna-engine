#!/usr/bin/env bash
# scripts/lib/data_root.sh -- single root for every generated-data path this repo
# writes: downloaded checkpoints, built model artifacts, toolchain/build caches,
# scratch work, buildstore CAS, qlab corpora, logs. SOURCE it (not amd_paths.sh,
# which stays scoped to the AMD upstream checkouts) from any script that reads or
# writes generated data.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/data_root.sh"
#   ... use "$XDNA_MODELS" / "$XDNA_ARTIFACTS" / "$XDNA_CACHE" / ...
#
# Point XDNA_DATA at a bigger disk via config/local.env (gitignored; copy
# config/local.env.example) instead of exporting it in every shell. Each derived
# dir is also individually overridable. Nothing here CREATES a directory --
# call xdna_mkdir first.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
# A linked worktree has no config/local.env or data/ of its own (both gitignored,
# so git worktree never copies them): inherit the main checkout's, which the .git
# FILE (not `worktree list`) names.
_dr_main=""
if [ -f "$REPO/.git" ]; then
  _dr_main="$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
  _dr_main="${_dr_main%/.git}"
fi
if [ -f "$REPO/config/local.env" ]; then
  . "$REPO/config/local.env"
elif [ -n "$_dr_main" ] && [ -f "$_dr_main/config/local.env" ]; then
  . "$_dr_main/config/local.env"
fi
if [ -z "${XDNA_DATA:-}" ] && [ ! -e "$REPO/data" ] && [ -n "$_dr_main" ] && [ -d "$_dr_main/data" ]; then
  XDNA_DATA="$_dr_main/data"
fi
unset _dr_main

export XDNA_DATA="${XDNA_DATA:-$REPO/data}"
export XDNA_MODELS="${XDNA_MODELS:-$XDNA_DATA/models}"
export XDNA_ARTIFACTS="${XDNA_ARTIFACTS:-$XDNA_DATA/artifacts}"
export XDNA_CACHE="${XDNA_CACHE:-$XDNA_DATA/cache}"
export XDNA_BUILD="${XDNA_BUILD:-$XDNA_DATA/build}"
export XDNA_SCRATCH="${XDNA_SCRATCH:-$XDNA_DATA/scratch}"
export XDNA_CAS="${XDNA_CAS:-$XDNA_DATA/cas}"
export XDNA_QLAB="${XDNA_QLAB:-$XDNA_DATA/qlab}"
export XDNA_LOGS="${XDNA_LOGS:-$XDNA_DATA/logs}"

# Names scripts already used before this root existed. Kept as individual
# overrides (some are also read directly by rust/npu-service, see env_flags.rs)
# but their DEFAULT now derives from XDNA_DATA instead of a hardcoded path.
export BUILDSTORE_CAS="${BUILDSTORE_CAS:-$XDNA_CAS}"
export QLAB_WORK="${QLAB_WORK:-$XDNA_QLAB}"
export XDNA_ARTIFACT_STORE="${XDNA_ARTIFACT_STORE:-$XDNA_ARTIFACTS}"
export XDNA_MODEL_STORE="${XDNA_MODEL_STORE:-$XDNA_MODELS}"
export XDNA_BUILD_ROOT="${XDNA_BUILD_ROOT:-$XDNA_BUILD}"

xdna_mkdir() { mkdir -p "$@"; }
