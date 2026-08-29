#!/usr/bin/env bash
# Kernel build-sandbox freshness stamp. Sourced by the kernel-build scripts (build_kernels.sh,
# build_parakeet_kernels.sh) and by check_kernel_artifact_freshness.sh. Purges a build dir ONLY when
# the toolchain lock-hash changed since the sandbox was last built (a toolchain change), so kernels
# rebuild clean against the new toolchain and the tile-named-object stale-trap cannot fire. No-op on
# a kernel-logic-only change (same lock-hash).
#   ensure_fresh_sandbox <build_dir>
# Env: KERNEL_SANDBOX_GC=0 disables (no purge, no stamp). REPO (repo root) is honored if already set.

# sha256(toolchain.lock) truncated to 12 hex chars -- the ONE derivation of the freshness stamp
# value, so ensure_fresh_sandbox and check_kernel_artifact_freshness.sh can never compute two
# different hashes for the same lock file. Same truncation as kernel_registry::current_toolchain_hash
# (rust/npu-asr, branch feat/repin-artifact-preflight) -- keep both in sync if either changes.
current_toolchain_hash() {
  local repo="$1"
  sha256sum "$repo/toolchain.lock" | cut -c1-12
}

ensure_fresh_sandbox() {
  local bd="$1"
  [ "${KERNEL_SANDBOX_GC:-1}" = "0" ] && return 0
  local repo="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  # path guard: only ever operate under the mlir-aie build tree
  case "$bd" in
    *mlir-aie/programming_examples/*) : ;;
    *) echo "[kernel_sandbox] refuse: '$bd' is not under the mlir-aie build tree" >&2; return 1 ;;
  esac
  local cur; cur="$(current_toolchain_hash "$repo")"
  local stamp="$bd/.toolchain-stamp"
  if [ -d "$bd" ] && { [ ! -f "$stamp" ] || [ "$(cat "$stamp" 2>/dev/null)" != "$cur" ]; }; then
    echo "[kernel_sandbox] toolchain changed -> purge $bd (was $(cat "$stamp" 2>/dev/null || echo none), now $cur)" >&2
    rm -rf "$bd"
  fi
  mkdir -p "$bd"
  echo "$cur" > "$stamp"
}
