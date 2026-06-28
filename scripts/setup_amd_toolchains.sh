#!/usr/bin/env bash
# =============================================================================
# Single entry point: apply ALL AMD-toolchain patch series onto their targets.
#
#   scripts/setup_amd_toolchains.sh            # apply every configured repo
#   RESET=--reset scripts/setup_amd_toolchains.sh   # un-apply + re-apply each
#
# Finds the repo root from its own location. Target repo paths come from ENV
# (no absolute fork paths are committed to git). Optionally source a gitignored
# `scripts/amd_toolchains.env` (see amd_toolchains.env.example) to set them.
#
# Reports a per-repo + overall summary so ONE run tells you everything applied,
# or exactly what broke. Non-zero exit if any configured repo failed.
#
# Reuses the generic scripts/apply_patches.sh; per-repo order lives in a `series`
# file under route_b_kernels/patches/ (e.g. mlir-aie.series, iron.series).
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$REPO/scripts/amd_toolchains.env" ] && . "$REPO/scripts/amd_toolchains.env"

P="$REPO/route_b_kernels/patches"
APPLY="$REPO/scripts/apply_patches.sh"
RESET="${RESET:-}"

# Target repos. mlir-aie is the in-repo submodule (relative); the rest are
# sibling checkouts whose paths come from ENV with sane fallbacks.
MLIR_AIE_DIR="${MLIR_AIE_DIR:-$REPO/mlir-aie}"
IRON_DIR="${IRON_DIR:-${HOME}/repositories/ns/amd/IRON}"
LLVM_AIE_DIR="${LLVM_AIE_DIR:-}"   # set when we start patching Peano
MLIR_AIR_DIR="${MLIR_AIR_DIR:-}"   # set when we start patching mlir-air

fail=0
SUMMARY=()

apply_one() {  # name  series-file  target-dir
  local name="$1" series="$2" target="$3"
  if [ -z "$target" ];        then SUMMARY+=("SKIP  $name  (no *_DIR set in env)"); return; fi
  if [ ! -f "$series" ];      then SUMMARY+=("SKIP  $name  (no series file: $(basename "$series"))"); return; fi
  if [ ! -e "$target/.git" ]; then SUMMARY+=("SKIP  $name  (target not a git repo: $target)"); return; fi
  echo "============================================================"
  echo "=== $name  ->  $target"
  echo "============================================================"
  if bash "$APPLY" "$series" "$target" "$RESET"; then
    SUMMARY+=("OK    $name")
  else
    SUMMARY+=("FAIL  $name"); fail=1
  fi
}

# mlir-aie + mlir-air + IRON are NO LONGER patched here -- their delta is carried as FORK BRANCHES (commits,
# not .patch): mlir-aie = atassis/mlir-aie:xdna2-asr (setup_route_b.sh checks it out, toolchain_up.sh builds
# it); IRON = atassis/IRON:xdna2-asr (the decode build scripts require amd/IRON checked out on it); mlir-air =
# atassis/mlir-air per-PR branches (#1694, #1695). Only llvm-aie/Peano remains (consumed as a pinned wheel;
# this .series apply is a no-op unless an llvm-aie.series is ever added).
apply_one llvm-aie "$P/llvm-aie.series" "$LLVM_AIE_DIR"

echo
echo "===================== SUMMARY ====================="
printf '  %s\n' "${SUMMARY[@]}"
echo "==================================================="
if [ "$fail" -eq 0 ]; then
  echo "ALL OK"
else
  echo "SOME FAILED -- see FAIL lines above" >&2
  exit 1
fi
