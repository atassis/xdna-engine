#!/usr/bin/env bash
# Kernel build-sandbox freshness stamp. Sourced by the kernel-build scripts (build_kernels.sh,
# build_parakeet_kernels.sh). Retires a build dir ONLY when the toolchain lock-hash changed since the
# sandbox was last built (a toolchain change), so kernels rebuild clean against the new toolchain and the
# tile-named-object stale-trap cannot fire. No-op on a kernel-logic-only change (same lock-hash).
#   ensure_fresh_sandbox <build_dir>
# Env: KERNEL_SANDBOX_GC=0 disables (no retire, no stamp). REPO (repo root) is honored if already set.
#      KERNEL_SANDBOX_KEEP_DAYS (default 7) -- age at which retired sandboxes are reaped.
#
# WHY THIS MOVES ASIDE INSTEAD OF DELETING (2026-09-01). These build dirs live in the mlir-aie
# SUBMODULE and are therefore SHARED by every checkout and every concurrent session on this box --
# they are not per-worktree. `rm -rf` here on a lock-hash change destroys whatever any other session
# built, and a toolchain re-pin is exactly when several sessions are most likely to be mid-build. The
# freshness requirement is real (a stale object silently linked against a new toolchain is the trap
# this guard exists for), but it only requires that the stale tree stop being FOUND -- not that it
# stop EXISTING. So retire it to a timestamped sibling and reap by age. Recovery is a `mv` instead of
# a rebuild, and a concurrent build loses its path rather than its output.
#
# The path guard below is load-bearing: it is the only thing standing between a bad argument and an
# rm/mv on an arbitrary directory.

ensure_fresh_sandbox() {
  local bd="$1"
  [ "${KERNEL_SANDBOX_GC:-1}" = "0" ] && return 0
  local repo="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  # path guard: only ever operate under the mlir-aie build tree
  case "$bd" in
    *mlir-aie/programming_examples/*) : ;;
    *) echo "[kernel_sandbox] refuse: '$bd' is not under the mlir-aie build tree" >&2; return 1 ;;
  esac
  # Fail loud on a missing lock. Without this the sha is EMPTY, never matches the stamp, and the
  # sandbox is retired on EVERY call -- a silent rebuild-everything loop that reads as slowness, not
  # as a bug. REPO is honored from the environment, so a stale REPO from an unrelated script is
  # enough to trigger it.
  [ -f "$repo/toolchain.lock" ] || {
    echo "[kernel_sandbox] refuse: no toolchain.lock at '$repo' (REPO=${REPO:-unset})" >&2; return 1; }
  local cur; cur="$(sha256sum "$repo/toolchain.lock" | cut -c1-12)"
  [ -n "$cur" ] || { echo "[kernel_sandbox] refuse: empty lock hash" >&2; return 1; }
  local stamp="$bd/.toolchain-stamp"
  if [ -d "$bd" ] && { [ ! -f "$stamp" ] || [ "$(cat "$stamp" 2>/dev/null)" != "$cur" ]; }; then
    local was; was="$(cat "$stamp" 2>/dev/null || echo none)"
    local retired="${bd}.stale-${was}-$(date +%Y%m%dT%H%M%S)"
    echo "[kernel_sandbox] toolchain changed (was $was, now $cur)" >&2
    echo "[kernel_sandbox] retiring $bd -> $retired  (NOT deleted; this tree is shared across sessions)" >&2
    mv "$bd" "$retired" || { echo "[kernel_sandbox] could not retire $bd -- leaving it alone" >&2; return 1; }
  fi
  # reap retired sandboxes by age, so move-aside does not grow without bound
  local keep="${KERNEL_SANDBOX_KEEP_DAYS:-7}"
  local parent; parent="$(dirname "$bd")"
  local base; base="$(basename "$bd")"
  find "$parent" -maxdepth 1 -type d -name "${base}.stale-*" -mtime "+${keep}" \
       -exec echo "[kernel_sandbox] reaping aged sandbox {}" \; -exec rm -rf {} + 2>/dev/null || true
  mkdir -p "$bd"
  echo "$cur" > "$stamp"
}
