#!/usr/bin/env bash
# check_worktree_toolchain_anchor.sh -- refuse to remove a worktree a built toolchain instance
# is anchored to.
#
# gc_instances (toolchain_gc.sh) already refuses to delete an INSTANCE dir that a compat symlink
# points at. Nothing plays that role on the other side: a toolchain instance's src/ is a git
# worktree of the mlir-aie submodule, and that worktree's admin data can itself live INSIDE the
# git-dir of an outer task worktree (submodule worktrees nest under
# <repo>/.git/worktrees/<name>/modules/<submodule>/worktrees/<sub-name>). Removing the outer task
# worktree -- exactly the `tidy` Phase 3 flow, `git worktree remove` + `--force` after a submodule
# deinit -- deletes that admin dir out from under the instance's src/.git gitdir pointer, and the
# instance silently stops resolving. Measured 2026-09-08: two live instances anchor into wt-repin
# and wt-s2-residency this way.
#
# Usage:
#   check_worktree_toolchain_anchor.sh <repo-path> <worktree-path> [instances-root]
#     instances-root defaults to XDNA_CACHE/instances (see cache_env.sh), or <workspace-root>/.cache/instances.
#
# Exit 0  = safe to remove (no instance anchored here).
# Exit 1  = REFUSED -- an instance's src/.git resolves inside this worktree's admin dir; naming the
#           instance(s), so `git worktree remove <path>` is not the next command to run blind.
set -euo pipefail

usage() { echo "usage: check_worktree_toolchain_anchor.sh <repo-path> <worktree-path> [instances-root]" >&2; exit 2; }
[ $# -ge 2 ] || usage
REPO_PATH="$1"; WT_PATH="$2"
[ -d "$WT_PATH" ] || { echo "not a directory: $WT_PATH" >&2; exit 2; }

if [ $# -ge 3 ]; then
  INSTROOT="$3"
else
  # Same resolution toolchain_up.sh/toolchain_gc.sh use, so this checks the root gc_instances GCs.
  WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  INSTROOT="${XDNA_CACHE:-$WS_ROOT/.cache}/instances"
fi

WT_GITDIR="$(git -C "$WT_PATH" rev-parse --git-dir 2>/dev/null)" || { echo "not a git worktree: $WT_PATH" >&2; exit 2; }
case "$WT_GITDIR" in /*) : ;; *) WT_GITDIR="$(cd "$WT_PATH" && cd "$(dirname "$WT_GITDIR")" && pwd)/$(basename "$WT_GITDIR")" ;; esac

[ -d "$INSTROOT" ] || { echo "[check_worktree_toolchain_anchor] no instances root at $INSTROOT -- nothing to check, safe"; exit 0; }

hits=()
for gitfile in "$INSTROOT"/*/src/.git; do
  [ -f "$gitfile" ] || continue   # only the gitdir-pointer form; a real .git dir here is not a worktree
  target="$(sed -n 's/^gitdir: //p' "$gitfile")"
  case "$target" in
    "$WT_GITDIR"/*|"$WT_GITDIR")
      hits+=("$(basename "$(dirname "$(dirname "$gitfile")")")  ->  $target") ;;
  esac
done

if [ "${#hits[@]}" -gt 0 ]; then
  echo "[check_worktree_toolchain_anchor] REFUSED: $WT_PATH anchors ${#hits[@]} toolchain instance(s):" >&2
  for h in "${hits[@]}"; do echo "  $h" >&2; done
  echo "[check_worktree_toolchain_anchor] removing this worktree orphans them silently (self-healing on next toolchain_up.sh, but the running instance breaks with no attribution)." >&2
  echo "[check_worktree_toolchain_anchor] rebuild those instances from the submodule's primary checkout first, or accept the rebuild cost, before removing." >&2
  exit 1
fi

echo "[check_worktree_toolchain_anchor] safe: no toolchain instance anchored to $WT_PATH"
