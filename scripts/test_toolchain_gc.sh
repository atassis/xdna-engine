#!/usr/bin/env bash
# Test gc_instances retention, including the compatibility-symlink case.
#
# An instance can be LIVE under a key that is not its directory name: when the lock's semantic key
# differs from the legacy whole-file key, toolchain_up.sh adopts the existing build by symlinking
# the new key at the old directory (it cannot rename it -- build/CMakeCache.txt bakes the absolute
# path and src/ is a git worktree). `find -type d` does not see that symlink, so without resolving
# it the GC would happily delete the very instance in use.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/toolchain_gc.sh
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0
ok(){ echo "  PASS $1"; }
no(){ echo "  FAIL $1"; fail=1; }

for h in aaaaaaaaaaaa bbbbbbbbbbbb cccccccccccc dddddddddddd; do mkdir -p "$T/$h"; done
touch -d "2026-01-01" "$T/aaaaaaaaaaaa"
touch -d "2026-02-01" "$T/bbbbbbbbbbbb"
touch -d "2026-03-01" "$T/cccccccccccc"
touch -d "2026-04-01" "$T/dddddddddddd"
ln -sfn aaaaaaaaaaaa "$T/eeeeeeeeeeee"     # oldest dir is live under a new semantic key

gc_instances "$T" 2 "$T/eeeeeeeeeeee" >/dev/null 2>&1
[ -d "$T/aaaaaaaaaaaa" ] && ok "symlinked target survives though oldest and over quota" \
                        || no "deleted the live instance reached through a compat symlink"
[ -d "$T/dddddddddddd" ] && ok "newest instance kept" || no "newest instance deleted"
[ ! -d "$T/bbbbbbbbbbbb" ] && ok "genuinely stale instance still collected" \
                          || no "collected nothing -- retention is not working at all"

echo "== TOOLCHAIN_GC=0 disables collection entirely =="
mkdir -p "$T/ffffffffffff"; touch -d "2025-01-01" "$T/ffffffffffff"
TOOLCHAIN_GC=0 gc_instances "$T" 1 "$T/dddddddddddd" >/dev/null 2>&1
[ -d "$T/ffffffffffff" ] && ok "TOOLCHAIN_GC=0 respected" || no "collected despite TOOLCHAIN_GC=0"
exit $fail
