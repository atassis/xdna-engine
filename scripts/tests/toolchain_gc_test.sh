#!/usr/bin/env bash
# Unit test for gc_instances (scripts/toolchain_gc.sh). Self-contained; fake instance dirs, no real build.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # scripts/tests -> repo root
source "$HERE/scripts/toolchain_gc.sh"

pass=0 fail=0
ok()  { echo "  ok: $1"; pass=$((pass+1)); }
bad() { echo "  FAIL: $1"; fail=$((fail+1)); }

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
root="$tmp/instances"; mkdir -p "$root"
# 5 fake instances (12-hex names) with staggered mtimes: h1 oldest ... h5 newest.
for i in 1 2 3 4 5; do
  d="$root/$(printf '%012x' "$i")"; mkdir -p "$d"
  touch -d "2026-01-0$i 00:00" "$d"
done
newest="$root/$(printf '%012x' 5)"
oldest="$root/$(printf '%012x' 1)"

# 1) keep newest 2, protect the newest: expect h4,h5 kept, h1,h2,h3 gone
gc_instances "$root" 2 "$newest"
[ -d "$newest" ] && [ -d "$root/$(printf '%012x' 4)" ] && ok "keep: newest 2 survive" || bad "keep: newest 2 not preserved"
[ ! -d "$oldest" ] && [ ! -d "$root/$(printf '%012x' 3)" ] && ok "prune: older removed" || bad "prune: older survived"

# 2) protect is never deleted even if it falls outside keep-N
for i in 1 2 3 4 5; do d="$root/$(printf '%012x' "$i")"; mkdir -p "$d"; touch -d "2026-01-0$i 00:00" "$d"; done
gc_instances "$root" 1 "$oldest"       # keep newest 1 (h5) + protect h1
[ -d "$oldest" ] && ok "protect: protected dir survives keep-N" || bad "protect: protected dir deleted"

# 3) opt-out: TOOLCHAIN_GC=0 -> nothing deleted
for i in 1 2 3 4 5; do d="$root/$(printf '%012x' "$i")"; mkdir -p "$d"; touch -d "2026-01-0$i 00:00" "$d"; done
TOOLCHAIN_GC=0 gc_instances "$root" 1 "$newest"
[ "$(find "$root" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 5 ] && ok "opt-out: nothing pruned" || bad "opt-out: pruned anyway"

# 4) non-lockhash dirs are never touched (guard)
mkdir -p "$root/not-a-hash"; touch -d "2020-01-01" "$root/not-a-hash"
gc_instances "$root" 1 "$newest"
[ -d "$root/not-a-hash" ] && ok "guard: non-hash dir untouched" || bad "guard: deleted non-hash dir"

echo "== toolchain_gc: $pass passed, $fail failed =="; [ "$fail" -eq 0 ]
