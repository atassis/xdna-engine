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

# Keep the external-anchor scan (below) hermetic: left to its defaults it would walk the real
# scratch/build/artifacts roots from a unit test, which is slow and reads another lane's tree.
export TOOLCHAIN_GC_ANCHOR_ROOTS="$T/anchors"
mkdir -p "$T/anchors"

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

# mtime rank only sees readers that came through the instances root. On 2026-11-12 the live FSR1
# lane's headers (scratch/fsr1-main/hdr-{wheel,fork}/aie -> instances/3ce32a1d7465/build/include/aie)
# made the OLDEST instance a production input, and every KEEP_N <= 4 evicted it first. These cases
# seed the outside reader and assert both directions -- eviction must still happen when nothing
# reads the instance, or the guard would just be a blanket "keep".
echo "== an instance read from outside the instances root survives its rank =="
I="$T/g"; mkdir -p "$I/instances" "$I/scratch/fsr1-main/hdr-wheel"
for h in 1a1a1a1a1a1a 2b2b2b2b2b2b 3c3c3c3c3c3c 4d4d4d4d4d4d; do
  mkdir -p "$I/instances/$h/build/include/aie"
done
touch -d "2026-11-07" "$I/instances/1a1a1a1a1a1a"
touch -d "2026-10-28" "$I/instances/2b2b2b2b2b2b"
touch -d "2026-10-03" "$I/instances/3c3c3c3c3c3c"
touch -d "2026-10-01" "$I/instances/4d4d4d4d4d4d"
ln -sfn "$I/instances/4d4d4d4d4d4d/build/include/aie" "$I/scratch/fsr1-main/hdr-wheel/aie"
TOOLCHAIN_GC_ANCHOR_ROOTS="$I/scratch" gc_instances "$I/instances" 2 "" 2>/dev/null
[ -d "$I/instances/4d4d4d4d4d4d" ] && ok "outside reader keeps the oldest instance despite its rank" \
                                   || no "deleted an instance a lane outside the root is reading"
[ ! -d "$I/instances/3c3c3c3c3c3c" ] && ok "unanchored stale instance still collected" \
                                     || no "the guard suppressed eviction everywhere"

echo "== the same layout with no outside reader evicts both stale instances =="
J="$T/h"; mkdir -p "$J/instances" "$J/scratch"
for h in 5e5e5e5e5e5e 6f6f6f6f6f6f 7a7a7a7a7a7a 8b8b8b8b8b8b; do
  mkdir -p "$J/instances/$h/build/include/aie"
done
touch -d "2026-11-07" "$J/instances/5e5e5e5e5e5e"
touch -d "2026-10-28" "$J/instances/6f6f6f6f6f6f"
touch -d "2026-10-03" "$J/instances/7a7a7a7a7a7a"
touch -d "2026-10-01" "$J/instances/8b8b8b8b8b8b"
TOOLCHAIN_GC_ANCHOR_ROOTS="$J/scratch" gc_instances "$J/instances" 2 "" 2>/dev/null
[ ! -d "$J/instances/7a7a7a7a7a7a" ] && [ ! -d "$J/instances/8b8b8b8b8b8b" ] \
  && ok "nothing read -> rank decides, both collected" \
  || no "retention degraded into 'keep everything'"

echo "== TOOLCHAIN_GC_ANCHOR_RESOLVE=1 finds the same reader by resolving links =="
K="$T/i"; mkdir -p "$K/instances" "$K/scratch/lane"
for h in 9c9c9c9c9c9c 0d0d0d0d0d0d; do mkdir -p "$K/instances/$h/build/include/aie"; done
touch -d "2026-11-07" "$K/instances/9c9c9c9c9c9c"
touch -d "2026-10-01" "$K/instances/0d0d0d0d0d0d"
ln -sfn "$K/instances/0d0d0d0d0d0d/build/include/aie" "$K/scratch/lane/aie"
TOOLCHAIN_GC_ANCHOR_ROOTS="$K/scratch" TOOLCHAIN_GC_ANCHOR_RESOLVE=1 gc_instances "$K/instances" 1 "" 2>/dev/null
[ -d "$K/instances/0d0d0d0d0d0d" ] && ok "resolving scan protected the anchored instance" \
                                   || no "resolving scan deleted an anchored instance"

exit $fail
