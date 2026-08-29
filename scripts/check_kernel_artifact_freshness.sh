#!/usr/bin/env bash
# Preflight: is every resident kernel build dir still fresh against the CURRENT toolchain.lock?
#
# WHY THIS EXISTS: a toolchain re-pin changes what a rebuild would produce, but does nothing to
# artifacts already sitting on disk from the OLD pin -- ensure_fresh_sandbox (kernel_sandbox.sh)
# only purges a dir the next time its owning build script actually runs. Between a re-pin and the
# next full kernel build, every dir it covers is stamped for a hash that no longer matches
# toolchain.lock, and the shipped service loads whatever bytes are there regardless (a re-pin
# silently broke it for 5 days -- task artifact-preflight-and-fail-loud). This script is the
# read-only check for that gap: run it after a re-pin, before trusting any resident artifact.
#
# Generalizes kernel_registry::check_toolchain_freshness (rust/npu-asr, branch
# feat/repin-artifact-preflight, unmerged/whole_array-only) to every dir
# ensure_fresh_sandbox can purge. Same 5-state verdict, same stamp convention, same hash
# derivation (current_toolchain_hash, kernel_sandbox.sh) -- so this script and that Rust check can
# never silently disagree about what "fresh" means.
#
# Usage: scripts/check_kernel_artifact_freshness.sh
#   Exit 0: every covered dir is OK. Exit 1: at least one dir needs a rebuild (message names the
#   dir, the regen script, and -- for a hash mismatch -- both hashes). Read-only: never builds,
#   purges, or writes a stamp.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/kernel_sandbox.sh"

PE="$REPO/mlir-aie/programming_examples"
MM="$PE/basic/matrix_multiplication/single_core"
MMW="$PE/basic/matrix_multiplication/whole_array"

# dir | regen script (relative to $REPO) -- the two build_*_kernels.sh entry points that cover
# these 5 dirs today (see their own `for _bd in ...` / `ensure_fresh_sandbox` call sites).
DIRS=(
  "$MMW/build|scripts/build_parakeet_kernels.sh"
  "$MM/build|scripts/build_kernels.sh"
  "$PE/ml/dwconv1d/build|scripts/build_kernels.sh"
  "$PE/ml/layernorm/build|scripts/build_kernels.sh"
  "$PE/ml/silu/build|scripts/build_kernels.sh"
)

current="$(current_toolchain_hash "$REPO")"
echo "[check_kernel_artifact_freshness] toolchain.lock sha256(12)=$current"

fail=0
for entry in "${DIRS[@]}"; do
  dir="${entry%%|*}"
  regen="${entry#*|}"
  rel="${dir#"$REPO"/}"

  if [ ! -d "$dir" ]; then
    echo "[check_kernel_artifact_freshness] MISSING   $rel does not exist -- rebuild with $regen" >&2
    fail=1; continue
  fi

  stamp_file="$dir/.toolchain-stamp"
  if [ ! -f "$stamp_file" ]; then
    echo "[check_kernel_artifact_freshness] UNSTAMPED $rel has no .toolchain-stamp -- freshness against toolchain.lock is unknown (built before this convention existed, or by a path that skips ensure_fresh_sandbox); rebuild with $regen" >&2
    fail=1; continue
  fi

  stamp="$(cat "$stamp_file")"
  if [ "$stamp" != "$current" ]; then
    echo "[check_kernel_artifact_freshness] STALE     $rel was built for toolchain.lock=$stamp, but toolchain.lock is now $current -- it was re-pinned and this dir was never rebuilt; rebuild with $regen" >&2
    fail=1; continue
  fi

  if ! find "$dir" -maxdepth 1 -name 'final*.xclbin' -print -quit | grep -q .; then
    echo "[check_kernel_artifact_freshness] EMPTY     $rel is stamped current ($current) but holds no final*.xclbin -- the rebuild after the last purge did not finish; rebuild with $regen" >&2
    fail=1; continue
  fi

  echo "[check_kernel_artifact_freshness] OK        $rel (toolchain.lock=$current)"
done

echo "[check_kernel_artifact_freshness] NOT COVERED: artifacts/relpos, artifacts/conveyor -- no .toolchain-stamp mechanism exists for either family (see [[manifest-preflight-cannot-see-the-relpos-artifacts]]). A clean run above is NOT evidence those two are fresh."

if [ "$fail" -ne 0 ]; then
  echo "[check_kernel_artifact_freshness] one or more resident kernel build dirs are stale or missing against the current toolchain pin -- refusing to call this a clean re-pin." >&2
  exit 1
fi
