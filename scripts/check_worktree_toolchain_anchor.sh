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
# Exit 1  = REFUSED -- instance source links or Git administration depend on this worktree.
# Exit 2  = invalid arguments or an incomplete source/administration scan.
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

WT_ABS="$(cd "$WT_PATH" && pwd -P)"

[ -d "$INSTROOT" ] || { echo "[check_worktree_toolchain_anchor] no instances root at $INSTROOT -- nothing to check, safe"; exit 0; }

python3 - "$WT_GITDIR" "$WT_ABS" "$INSTROOT" <<'PY'
import os
from pathlib import Path
import sys

admin, worktree, instances = (Path(arg).resolve() for arg in sys.argv[1:])
hits = []
errors = []


def anchored(path):
    target = path.resolve()
    return target.is_relative_to(admin) or target.is_relative_to(worktree)


def inspect(path):
    try:
        # Do not traverse linked source trees; resolve their dependency directly.
        if path.is_symlink():
            try:
                path.resolve(strict=True)
            except FileNotFoundError:
                pass  # A missing target can still point inside the carrier.
            if anchored(path):
                hits.append(f"{path.relative_to(instances)} -> {path.resolve()}")
        if path.name == ".git" and path.is_file():
            pointer = path.read_text().strip()
            if not pointer.startswith("gitdir: ") or not pointer[8:]:
                raise ValueError(f"malformed gitdir pointer: {path}")
            target = (path.parent / pointer[8:]).resolve()
            if anchored(target):
                hits.append(f"{path.relative_to(instances)} -> {target}")
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(str(error))


try:
    for instance in sorted(instances.iterdir()):
        source = instance / "src"
        inspect(source)
        if not source.is_dir():
            continue
        # Nested submodules can own administration under the carrier even after
        # src/.git has been rehomed. Never descend into Git object databases.
        for root, directories, files in os.walk(source, onerror=lambda e: errors.append(str(e))):
            for name in sorted(directories + files):
                inspect(Path(root) / name)
            directories[:] = [name for name in directories if name != ".git"]
except (OSError, RuntimeError) as error:
    errors.append(str(error))

if hits or errors:
    print(f"[check_worktree_toolchain_anchor] REFUSED: {worktree}", file=sys.stderr)
    for hit in hits:
        print(f"  {hit}", file=sys.stderr)
    for error in errors:
        print(f"  incomplete scan: {error}", file=sys.stderr)
    print("[check_worktree_toolchain_anchor] preserve/rehome these dependencies and verify consumers "
          "before requesting removal; this check does not authorize a rebuild.", file=sys.stderr)
    sys.exit(2 if errors else 1)

print(f"[check_worktree_toolchain_anchor] safe: no toolchain instance anchored to {worktree}")
PY
