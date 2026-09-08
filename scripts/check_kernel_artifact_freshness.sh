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

# dir | regen script (relative to $REPO) -- the build_*_kernels.sh entry points that cover
# these dirs today (see their own `for _bd in ...` / `ensure_fresh_sandbox` call sites).
# ml/softmax400 and ml/mha_decode added 2026-09-08 (task
# artifact-families-with-no-freshness-stamp): both already write the SAME .toolchain-stamp
# convention at build time (build_kernels.sh's sandbox loop and build_mha_decode.sh's own
# ensure_fresh_sandbox call respectively) but neither was ever added here, so a stale rebuild of
# either was invisible to this preflight.
DIRS=(
  "$MMW/build|scripts/build_parakeet_kernels.sh"
  "$MM/build|scripts/build_kernels.sh"
  "$PE/ml/dwconv1d/build|scripts/build_kernels.sh"
  "$PE/ml/layernorm/build|scripts/build_kernels.sh"
  "$PE/ml/silu/build|scripts/build_kernels.sh"
  "$PE/ml/softmax400/build|scripts/build_kernels.sh"
  "$PE/ml/mha_decode/build|scripts/build_mha_decode.sh"
)

# FAMILY roots | regen script -- artifacts/<family>/<subdir>/{final.xclbin,insts.bin}, one
# .toolchain-stamp per SUBDIR (relpos_prebuild.sh writes one per bucket; conveyor{,_bd}_prebuild.sh
# write one for their single leaf dir). Added 2026-09-08, same task: these three were the
# family class check_kernel_artifact_freshness.sh's own "NOT COVERED" line named as an
# acknowledged gap (relpos, conveyor) or didn't name at all (conveyor_bd) -- distinct from DIRS
# because each root fans out over N subdirectories rather than being one build dir itself.
# NOT covered here: the artifacts/relpos.<variant> and artifacts/conveyor_bd_io siblings -- A/B
# experiment dirs with no producer script found in scripts/*.sh (orphaned or built ad hoc); adding
# them needs an owner decision on which are still live, not a mechanical extension of this list.
# LLM DECODE artifacts. A DIFFERENT stamp convention, which is why they were missed: the fused
# decode build writes {"toolchain":{"hash":...}} INTO meta.json and emits decode.elf, so neither
# the .toolchain-stamp file nor the final*.xclbin that check_one looks for is ever present.
# Added 2026-09-09 after a re-pin left `npu generate` refusing to load ("toolchain-stale: built
# against c4fb9caa28b9") while this script reported every dir it knew about as fresh -- the gate
# said green because the artifact was outside its list, not because it was current. Discovered by
# the owner running the command, which is the cheapest possible detector and the wrong one to
# depend on. artifacts/gemma4-12b/decode is deliberately absent: it has no meta.json, so it
# predates the convention and adding it would report UNSTAMPED forever with no producer to fix it.
ELF_DIRS=(
  "$REPO/artifacts/qwen3-0.6b/decode|scripts/build_llm_decode.sh qwen3-0.6b"
)

FAMILY_DIRS=(
  "$REPO/artifacts/relpos|scripts/relpos_prebuild.sh"
  "$REPO/artifacts/conveyor|scripts/conveyor_prebuild.sh"
  "$REPO/artifacts/conveyor_bd|scripts/conveyor_bd_prebuild.sh"
)

current="$(current_toolchain_hash "$REPO")"
echo "[check_kernel_artifact_freshness] toolchain.lock sha256(12)=$current"

fail=0

# check_one <dir> <regen-script> -- the 5-state verdict (MISSING/UNSTAMPED/STALE/EMPTY/OK), shared
# by both DIRS (dir IS the artifact) and FAMILY_DIRS (dir is one subdir of a family). Sets `fail=1`
# on anything but OK; never exits itself, so a family root's later subdirs still get checked and
# reported even after an earlier subdir's warning.
# check_elf_one <dir> <regen> -- same five verdicts as check_one, against the meta.json convention.
check_elf_one() {
  local dir="$1" regen="$2"
  local rel="${dir#"$REPO"/}"

  if [ ! -d "$dir" ]; then
    echo "[check_kernel_artifact_freshness] MISSING   $rel does not exist -- rebuild with $regen" >&2
    fail=1; return
  fi
  local meta="$dir/meta.json"
  if [ ! -f "$meta" ]; then
    echo "[check_kernel_artifact_freshness] UNSTAMPED $rel has no meta.json -- freshness against toolchain.lock is unknown; rebuild with $regen" >&2
    fail=1; return
  fi
  local stamp
  stamp="$(python3 -c 'import json,sys;print((json.load(open(sys.argv[1])).get("toolchain") or {}).get("hash",""))' "$meta" 2>/dev/null)"
  if [ -z "$stamp" ]; then
    echo "[check_kernel_artifact_freshness] UNSTAMPED $rel meta.json carries no toolchain.hash -- rebuild with $regen" >&2
    fail=1; return
  fi
  if [ "$stamp" != "$current" ]; then
    echo "[check_kernel_artifact_freshness] STALE     $rel was built for toolchain.lock=$stamp, but toolchain.lock is now $current -- it was re-pinned and this dir was never rebuilt; rebuild with $regen" >&2
    fail=1; return
  fi
  if [ ! -s "$dir/decode.elf" ]; then
    echo "[check_kernel_artifact_freshness] EMPTY     $rel is stamped current ($current) but holds no decode.elf; rebuild with $regen" >&2
    fail=1; return
  fi
  echo "[check_kernel_artifact_freshness] OK        $rel (toolchain.lock=$current)"
}

check_one() {
  local dir="$1" regen="$2"
  local rel="${dir#"$REPO"/}"

  if [ ! -d "$dir" ]; then
    echo "[check_kernel_artifact_freshness] MISSING   $rel does not exist -- rebuild with $regen" >&2
    fail=1; return
  fi

  local stamp_file="$dir/.toolchain-stamp"
  if [ ! -f "$stamp_file" ]; then
    echo "[check_kernel_artifact_freshness] UNSTAMPED $rel has no .toolchain-stamp -- freshness against toolchain.lock is unknown (built before this convention existed, or by a path that skips it); rebuild with $regen" >&2
    fail=1; return
  fi

  local stamp; stamp="$(cat "$stamp_file")"
  if [ "$stamp" != "$current" ]; then
    echo "[check_kernel_artifact_freshness] STALE     $rel was built for toolchain.lock=$stamp, but toolchain.lock is now $current -- it was re-pinned and this dir was never rebuilt; rebuild with $regen" >&2
    fail=1; return
  fi

  if ! find "$dir" -maxdepth 1 -name 'final*.xclbin' -print -quit | grep -q .; then
    echo "[check_kernel_artifact_freshness] EMPTY     $rel is stamped current ($current) but holds no final*.xclbin -- the rebuild after the last purge did not finish; rebuild with $regen" >&2
    fail=1; return
  fi

  echo "[check_kernel_artifact_freshness] OK        $rel (toolchain.lock=$current)"
}

for entry in "${DIRS[@]}"; do
  check_one "${entry%%|*}" "${entry#*|}"
done

for entry in "${FAMILY_DIRS[@]}"; do
  root="${entry%%|*}"; regen="${entry#*|}"
  rel_root="${root#"$REPO"/}"
  if [ ! -d "$root" ]; then
    echo "[check_kernel_artifact_freshness] MISSING   $rel_root does not exist -- rebuild with $regen" >&2
    fail=1; continue
  fi
  # One subdir per xclbin (bucket_100, single, ...) -- find every immediate child that actually
  # holds a final*.xclbin, so an empty or in-progress sibling subdir is not silently skipped as
  # "not a leaf" (it is caught by check_one's own EMPTY/MISSING states instead).
  found=0
  while IFS= read -r -d '' leaf; do
    found=1
    check_one "$(dirname "$leaf")" "$regen"
  done < <(find "$root" -mindepth 1 -maxdepth 3 -name 'final*.xclbin' -print0)
  if [ "$found" -eq 0 ]; then
    echo "[check_kernel_artifact_freshness] MISSING   $rel_root exists but holds no final*.xclbin anywhere under it -- rebuild with $regen" >&2
    fail=1
  fi
done

for row in "${ELF_DIRS[@]}"; do
  check_elf_one "${row%%|*}" "${row#*|}"
done

if [ "$fail" -ne 0 ]; then
  echo "[check_kernel_artifact_freshness] one or more resident kernel build dirs are stale or missing against the current toolchain pin -- refusing to call this a clean re-pin." >&2
  exit 1
fi
