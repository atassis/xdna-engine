#!/usr/bin/env bash
# .cache/build-<branch> trees are named for the branch that made them; 8 of 9 had no worktree and 5 of 9
# no branch. They cannot be renamed (CMakeCache.txt bakes the absolute path), so the tool identifies
# them by the branch they claim and reports which claims are dead.
set -uo pipefail
cd "$(dirname "$0")/.."
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok(){ echo "  PASS $1"; }
no(){ echo "  FAIL $1"; fail=1; }

mkdir -p "$T/build-alive" "$T/build-dead"
echo x > "$T/build-alive/CMakeCache.txt"; echo x > "$T/build-dead/CMakeCache.txt"

out="$(BUILD_CACHE_HOME="$T" ALIVE_BRANCHES="alive" bash scripts/build_cache.sh --list 2>&1)"
grep -q "build-alive.*live"   <<<"$out" && ok "live branch reported live"   || no "live not reported: $out"
grep -q "build-dead.*orphan"  <<<"$out" && ok "dead branch reported orphan" || no "orphan not reported: $out"

BUILD_CACHE_HOME="$T" ALIVE_BRANCHES="alive" bash scripts/build_cache.sh --gc --dry-run >/dev/null 2>&1
[ -d "$T/build-dead" ] && ok "dry-run deletes nothing" || no "dry-run deleted a tree"

BUILD_CACHE_HOME="$T" ALIVE_BRANCHES="alive" bash scripts/build_cache.sh --gc >/dev/null 2>&1
[ ! -d "$T/build-dead" ] && ok "orphan collected" || no "orphan survived --gc"
[ -d "$T/build-alive" ] && ok "live tree kept"    || no "collected a live tree"
exit $fail
