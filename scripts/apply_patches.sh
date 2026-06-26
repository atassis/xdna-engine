#!/usr/bin/env bash
# =============================================================================
# Generic patch-series applier.
#
# Applies an ORDERED series of patches onto a target git repo. Reusable across
# every external AMD repo we patch (mlir-aie submodule, amd/IRON, llvm-aie/Peano,
# mlir-air, ...). The single entry point `setup_amd_toolchains.sh` drives this.
#
#   scripts/apply_patches.sh <series-file> <target-repo-dir> [--reset]
#
# series-file: one patch path per line, in STACKING ORDER. Paths are resolved
#   relative to the series file's own directory (absolute paths also allowed).
#   Lines starting with '#' and blank lines are ignored.
#
# MODEL (deliberate): clean-base + ordered series + apply-all, FAIL-HARD. This is
#   NOT per-patch idempotent: patches stack on the SAME files, so a per-patch
#   `git apply --reverse --check` can't distinguish "already applied" from
#   "out of order". Idempotency = reset-then-apply. Pass --reset to restore the
#   target's TRACKED files to HEAD (un-applies prior patches) before applying.
#   (Untracked files are left alone — we never `clean -fd` a toolchain tree.)
# =============================================================================
set -uo pipefail

SERIES="${1:-}"
TARGET="${2:-}"
RESET="${3:-}"

if [ -z "$SERIES" ] || [ -z "$TARGET" ]; then
  echo "usage: apply_patches.sh <series-file> <target-repo-dir> [--reset]" >&2
  exit 2
fi
[ -f "$SERIES" ] || { echo "FATAL: series file not found: $SERIES" >&2; exit 2; }
[ -e "$TARGET/.git" ] || { echo "FATAL: target is not a git repo: $TARGET" >&2; exit 2; }

SERIES_DIR="$(cd "$(dirname "$SERIES")" && pwd)"
resolve() { case "$1" in /*) printf '%s' "$1";; *) printf '%s/%s' "$SERIES_DIR" "$1";; esac; }

# Read the series (skip comments + blank lines), preserving order.
PATCHES=()
while IFS= read -r line; do
  case "$line" in ''|\#*) continue;; esac
  PATCHES+=("$line")
done < "$SERIES"

if [ "${#PATCHES[@]}" -eq 0 ]; then
  echo "[apply_patches] $(basename "$SERIES"): no active patches; nothing to do."
  exit 0
fi

if [ "$RESET" = "--reset" ]; then
  echo "[apply_patches] --reset: restoring tracked files in $TARGET to HEAD"
  git -C "$TARGET" checkout -- . || { echo "FATAL: reset failed" >&2; exit 1; }
fi

# Clean-base guard: the first patch must apply, else the target is dirty/drifted.
first="$(resolve "${PATCHES[0]}")"
[ -f "$first" ] || { echo "FATAL: missing patch: $first" >&2; exit 2; }
if ! git -C "$TARGET" apply --check "$first" >/dev/null 2>&1; then
  echo "FATAL: first patch (${PATCHES[0]}) does NOT apply onto $TARGET." >&2
  echo "  -> target is not at a clean base (already patched, or drifted from the pin)." >&2
  echo "  -> re-run with --reset to restore tracked files, or verify the pinned commit." >&2
  exit 1
fi

n=0
for p in "${PATCHES[@]}"; do
  f="$(resolve "$p")"
  [ -f "$f" ] || { echo "FATAL: missing patch: $f" >&2; exit 2; }
  if git -C "$TARGET" apply "$f"; then
    echo "  [apply] $p"
    n=$((n + 1))
  else
    echo "FATAL: '$p' failed to apply (out of order, or target drifted from the pin)." >&2
    exit 3
  fi
done

echo "[apply_patches] OK: applied $n patch(es) from $(basename "$SERIES") onto $TARGET"
