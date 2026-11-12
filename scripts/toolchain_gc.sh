#!/usr/bin/env bash
# Instance-dir GC for the content-addressed toolchain instances. Sourced by toolchain_up.sh; called ONLY
# on the cold-build path (= a toolchain change), never on the warm cache-hit path. Keeps the newest
# KEEP_N instance dirs by mtime (last-used, kept current by toolchain_up's touch-on-use), rm -rf's the
# rest. Missing instances self-heal (toolchain_up rebuilds on next use), so pruning is recoverable online.
#   gc_instances <instances_root> <keep_n> <protect_dir>
# Env: TOOLCHAIN_GC=0 disables.

# External-anchor precheck. mtime rank only knows who used an instance last THROUGH THIS ROOT; the
# readers that actually break are the ones that resolved into it from somewhere else. On 2026-11-12
# the live FSR1 lane's headers (scratch/fsr1-main/hdr-{wheel,fork}/aie -> instances/3ce32a1d7465/
# build/include/aie) made the *oldest* instance a production input, and KEEP_N<=4 evicted it first.
# An instance with an outside reader is live whatever its rank says, so it is kept and reported.
#   TOOLCHAIN_GC_ANCHOR_ROOTS   ':'-separated roots to scan (default: scratch, build, artifacts and
#                               the install root -- where lanes put a header or python/aie link)
#   TOOLCHAIN_GC_ANCHOR_RESOLVE =1 resolves every symlink instead of matching the instance key in the
#                               link target text: exact but a full metadata walk, for a bounded run.
# A candidate whose scan is not provable is kept, never evicted: a skipped eviction self-heals, a
# dangling header link in a live lane does not. Reaching an instance through a compatibility symlink
# that does not name the key is covered by the compat-pin rule below, not by this scan.

gc_anchor_roots() {
  local roots="${TOOLCHAIN_GC_ANCHOR_ROOTS:-}"
  if [ -z "$roots" ]; then
    roots="${XDNA_SCRATCH:-}:${XDNA_BUILD:-}:${XDNA_ARTIFACTS:-}:${HOME}/.local/share/npu"
  fi
  local r IFS=':'
  for r in $roots; do
    [ -n "$r" ] && [ -d "$r" ] && printf '%s\n' "$r"
  done
}

# Does anything OUTSIDE the instances root read into $1? Echoes the reader on stdout, exit 0 if so.
instance_has_external_reader() {
  local inst="$(readlink -f "$1")" key="$(basename "$1")" r l hit=""
  [ -n "$key" ] || return 1
  while IFS= read -r r; do
    [ -n "$r" ] || continue
    case "$inst" in "$(readlink -f "$r")"/*) continue ;; esac     # scanning the root itself proves nothing
    if [ "${TOOLCHAIN_GC_ANCHOR_RESOLVE:-0}" = 1 ]; then
      # Exact: resolve every link and test containment. Slow, so it is opt-in.
      while IFS= read -r l; do
        case "$(readlink -f "$l" 2>/dev/null)" in "$inst"|"$inst"/*) hit="$l"; break 2 ;; esac
      done < <(find "$r" -xdev -type l 2>/dev/null)
    else
      hit="$(find "$r" -xdev -type l -lname "*/$key/*" -print -quit 2>/dev/null)"
      [ -n "$hit" ] && break
    fi
  done < <(gc_anchor_roots)
  [ -n "$hit" ] || return 1
  echo "$hit"
}

gc_instances() {
  local root="$1" keep="${2:-4}" protect="${3:-}"
  [ "${TOOLCHAIN_GC:-1}" = "0" ] && return 0
  [ -d "$root" ] || return 0
  local i=0 d b protect_real pinned
  # Resolve the protected path: an instance may be reached through a compatibility SYMLINK (a lock
  # whose semantic key differs from the legacy whole-file key adopts the existing build that way).
  # Comparing unresolved paths would then fail to protect the very directory in use.
  protect_real="$(readlink -f "$protect" 2>/dev/null || echo "$protect")"
  # Any directory that a symlink in this root points at is live under another key -- never GC it.
  # `[ -L ] && readlink` as the loop's last command makes the substitution exit 1 whenever the
  # final entry is a real directory, which set -e then turns into a failed toolchain_up AFTER a
  # fully successful build -- indistinguishable from a compile error. Use if/fi, which always
  # exits 0.
  pinned=" $(for l in "$root"/*; do if [ -L "$l" ]; then readlink -f "$l"; fi; done | tr '\n' ' ') "
  while IFS= read -r d; do
    b="$(basename "$d")"
    [[ "$b" =~ ^[0-9a-f]{12}$ ]] || continue          # only lock-hash dirs are eligible
    i=$((i+1))
    [ "$i" -le "$keep" ] && continue                   # keep the newest KEEP_N
    [ "$(readlink -f "$d")" = "$protect_real" ] && continue   # never delete the just-built instance
    case "$pinned" in *" $(readlink -f "$d") "*) continue ;; esac  # referenced by a compat symlink
    case "$d" in "$root"/*) : ;; *) continue ;; esac   # path guard: must be directly under root
    # Rank says this instance is stale; an outside reader says it is load-bearing. Trust the reader.
    local reader
    if reader="$(instance_has_external_reader "$d")"; then
      echo "[toolchain_up] GC: KEEP $b -- ranked stale but read from outside the instances root: $reader" >&2
      continue
    fi
    echo "[toolchain_up] GC: remove stale instance $b (last-used $(date -d @"$(stat -c %Y "$d" 2>/dev/null)" '+%Y-%m-%d %H:%M' 2>/dev/null))" >&2
    rm -rf "$d"
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -rn | awk '{print $2}')
}
