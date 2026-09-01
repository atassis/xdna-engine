#!/usr/bin/env bash
# Test the peano-local identity store: content-addressed naming, duplicate collapse, alias
# preservation, and per-LLVM-major GC retention.
#
# Runs entirely against STUB installs in a throwaway PEANO_LOCAL_HOME -- each stub is a shell
# script that prints a clang version banner -- so no toolchain is needed and the archiver asserts
# are never reached. That is the point: the naming/retention logic is what deletes directories,
# so it is what needs a test that can run anywhere.
#
# The load-bearing case is the last block: retention is per-major because crossing an LLVM major
# is what makes a rollback expensive (it needs a matching PEANO_DIST seed). A flat "keep newest N"
# drops the older major entirely -- which on 2026-09-01 would have deleted the only LLVM-21 install
# we had, the one toolchain.lock names as its rollback target.
set -uo pipefail
SCRIPT="${1:-$(cd "$(dirname "$0")" && pwd)/install_peano_local.sh}"
[ -x "$SCRIPT" ] || [ -f "$SCRIPT" ] || { echo "FAIL: no install_peano_local.sh at $SCRIPT" >&2; exit 1; }
T="$(mktemp -d)"; export PEANO_LOCAL_HOME="$T"
trap 'rm -rf "$T"' EXIT
fail=0
ok(){ echo "  PASS $1"; }
no(){ echo "  FAIL $1"; fail=1; }

mkstub() { # <dirname> <major> <sha40> <age-days>
  mkdir -p "$T/$1/bin"
  cat > "$T/$1/bin/clang++" <<EOS
#!/bin/sh
echo "clang version $2.0.0git (git@github.com:atassis/llvm-aie.git $3)"
EOS
  chmod +x "$T/$1/bin/clang++"
  touch -d "$4 days ago" "$T/$1"
}

echo "== fixture: 3x llvm21 (one a duplicate under 2 names) + 2x llvm22 =="
mkstub old-a      21 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 40
mkstub old-b      21 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb 30
mkstub old-c      21 cccccccccccccccccccccccccccccccccccccccc 20
mkstub dup-of-a   21 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 10
mkstub new-x      22 dddddddddddddddddddddddddddddddddddddddd 5
mkstub new-y      22 eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee 1

echo "== migrate =="
bash "$SCRIPT" --migrate >/dev/null 2>&1
n_real=$(find "$T" -maxdepth 1 -type d ! -path "$T" | wc -l)
n_link=$(find "$T" -maxdepth 1 -type l | wc -l)
[ "$n_real" = 5 ] && ok "6 dirs -> 5 real builds (duplicate collapsed)" || no "expected 5 real, got $n_real"
[ "$n_link" = 6 ] && ok "6 legacy names kept as aliases" || no "expected 6 aliases, got $n_link"
[ -d "$T/llvm21-aaaaaaaaaaaa" ] && ok "identity-named dir exists" || no "identity dir missing"
[ "$(readlink "$T/dup-of-a")" = "llvm21-aaaaaaaaaaaa" ] && ok "duplicate name aliased to the one build" || no "dup not aliased"
[ "$(readlink "$T/old-a")"    = "llvm21-aaaaaaaaaaaa" ] && ok "original name aliased too" || no "orig not aliased"

echo "== idempotence: migrate again changes nothing =="
bash "$SCRIPT" --migrate >/dev/null 2>&1
n2=$(find "$T" -maxdepth 1 -type d ! -path "$T" | wc -l)
[ "$n2" = 5 ] && ok "re-migrate is a no-op" || no "re-migrate changed the set ($n2)"

echo "== gc --dry-run removes nothing =="
bash "$SCRIPT" --gc --keep 1 --dry-run >/dev/null 2>&1
n3=$(find "$T" -maxdepth 1 -type d ! -path "$T" | wc -l)
[ "$n3" = 5 ] && ok "dry-run is non-destructive" || no "dry-run deleted something ($n3)"

echo "== gc --keep 1: keeps newest of EACH major, not newest overall =="
bash "$SCRIPT" --gc --keep 1 >/dev/null 2>&1
[ -d "$T/llvm22-eeeeeeeeeeee" ] && ok "kept newest llvm22" || no "dropped newest llvm22"
[ -d "$T/llvm21-aaaaaaaaaaaa" ] && ok "kept newest llvm21 (a flat keep-1 would have dropped it)" || no "dropped newest llvm21"
[ ! -d "$T/llvm21-bbbbbbbbbbbb" ] && ok "removed older llvm21" || no "kept an over-quota llvm21"
[ ! -d "$T/llvm22-dddddddddddd" ] && ok "removed older llvm22" || no "kept an over-quota llvm22"
[ ! -e "$T/old-b" ] && ok "alias of a removed build cleaned up" || no "dangling alias left behind"
[ -L "$T/dup-of-a" ] && ok "alias of a surviving build kept" || no "alias of survivor lost"
exit $fail
