#!/usr/bin/env bash
# .cache/build-<name> CMake/ninja trees, named for the branch that produced them. The name is a CLAIM
# about a branch; this asks git whether the claim still holds. They are not renameable to a content key
# (CMakeCache.txt bakes the absolute path), so the fix is to make the claim checkable and collect the
# ones whose branch is gone.
#
# Usage:
#   scripts/build_cache.sh --list                 # name, size, live|orphan
#   scripts/build_cache.sh --gc [--dry-run]       # remove trees whose branch no longer exists
#
# Env overrides:
#   BUILD_CACHE_HOME   where the build-* trees live (default: <workspace>/.cache)
#   ALIVE_BRANCHES     space-separated branch names to treat as live, bypassing git (testing)
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
HOME_DIR="${BUILD_CACHE_HOME:-$WS/.cache}"
MODE=list; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --list) MODE=list; shift ;;
    --gc) MODE=gc; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

_alive() {   # branch name -> "live" | "orphan"
  local b="$1" r
  if [ -n "${ALIVE_BRANCHES:-}" ]; then
    case " $ALIVE_BRANCHES " in *" $b "*) echo live; return 0 ;; esac
    echo orphan; return 0
  fi
  for r in "$WS/mlir-aie" "$REPO/mlir-aie" "$WS/llvm-aie" "$REPO"; do
    [ -d "$r" ] || continue
    git -C "$r" rev-parse --verify -q "$b" >/dev/null 2>&1 && { echo live; return 0; }
  done
  echo orphan; return 0
}

for d in "$HOME_DIR"/build-*/; do
  [ -d "$d" ] || continue
  d="${d%/}"; b="$(basename "$d")"; br="${b#build-}"
  st="$(_alive "$br")"
  sz="$(du -sh "$d" 2>/dev/null | cut -f1)"
  if [ "$MODE" = list ]; then
    printf '%-34s %-7s %s\n' "$b" "$sz" "$st"
  elif [ "$st" = orphan ]; then
    if [ "$DRY" = 1 ]; then echo "WOULD REMOVE $b ($sz, no branch '$br')"
    else echo "REMOVE $b ($sz, no branch '$br')"; rm -rf "$d"; fi
  else
    echo "KEEP   $b ($sz, branch '$br' exists)"
  fi
done
exit 0
