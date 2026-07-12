#!/usr/bin/env bash
# Instance-dir GC for the content-addressed toolchain instances. Sourced by toolchain_up.sh; called ONLY
# on the cold-build path (= a toolchain change), never on the warm cache-hit path. Keeps the newest
# KEEP_N instance dirs by mtime (last-used, kept current by toolchain_up's touch-on-use), rm -rf's the
# rest. Missing instances self-heal (toolchain_up rebuilds on next use), so pruning is recoverable online.
#   gc_instances <instances_root> <keep_n> <protect_dir>
# Env: TOOLCHAIN_GC=0 disables.

gc_instances() {
  local root="$1" keep="${2:-4}" protect="${3:-}"
  [ "${TOOLCHAIN_GC:-1}" = "0" ] && return 0
  [ -d "$root" ] || return 0
  local i=0 d b
  while IFS= read -r d; do
    b="$(basename "$d")"
    [[ "$b" =~ ^[0-9a-f]{12}$ ]] || continue          # only lock-hash dirs are eligible
    i=$((i+1))
    [ "$i" -le "$keep" ] && continue                   # keep the newest KEEP_N
    [ "$d" = "$protect" ] && continue                  # never delete the just-built instance
    case "$d" in "$root"/*) : ;; *) continue ;; esac   # path guard: must be directly under root
    echo "[toolchain_up] GC: remove stale instance $b (last-used $(date -d @"$(stat -c %Y "$d" 2>/dev/null)" '+%Y-%m-%d %H:%M' 2>/dev/null))" >&2
    rm -rf "$d"
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk '{print $2}')
}
