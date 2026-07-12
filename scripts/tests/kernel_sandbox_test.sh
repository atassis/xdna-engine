#!/usr/bin/env bash
# Unit test for ensure_fresh_sandbox (scripts/kernel_sandbox.sh). Self-contained; no real build.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # scripts/tests -> repo root
source "$HERE/scripts/kernel_sandbox.sh"

pass=0 fail=0
ok()  { echo "  ok: $1"; pass=$((pass+1)); }
bad() { echo "  FAIL: $1"; fail=$((fail+1)); }

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
# A fake REPO with a toolchain.lock, and a build dir under the guarded mlir-aie tree.
export REPO="$tmp/repo"
mkdir -p "$REPO"; printf 'MLIR_AIE_FORK_COMMIT=aaa\n' > "$REPO/toolchain.lock"
bd="$REPO/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build"

# 1) fresh dir: creates dir + stamp
ensure_fresh_sandbox "$bd"
[ -f "$bd/.toolchain-stamp" ] && ok "fresh: stamp written" || bad "fresh: no stamp"

# 2) same lock-hash: no purge (marker file survives)
touch "$bd/marker.o"
ensure_fresh_sandbox "$bd"
[ -f "$bd/marker.o" ] && ok "unchanged: sandbox preserved" || bad "unchanged: sandbox wrongly purged"

# 3) lock-hash changes: purge (marker gone), new stamp
printf 'MLIR_AIE_FORK_COMMIT=bbb\n' > "$REPO/toolchain.lock"
ensure_fresh_sandbox "$bd"
[ ! -e "$bd/marker.o" ] && ok "changed: sandbox purged" || bad "changed: stale file survived"
[ -f "$bd/.toolchain-stamp" ] && ok "changed: new stamp written" || bad "changed: no new stamp"

# 4) opt-out: KERNEL_SANDBOX_GC=0 -> no purge even on change
touch "$bd/marker2.o"; printf 'MLIR_AIE_FORK_COMMIT=ccc\n' > "$REPO/toolchain.lock"
KERNEL_SANDBOX_GC=0 ensure_fresh_sandbox "$bd"
[ -f "$bd/marker2.o" ] && ok "opt-out: no purge" || bad "opt-out: purged anyway"

# 5) path guard: refuse a dir outside the mlir-aie build tree
if ensure_fresh_sandbox "$tmp/not/guarded/build" 2>/dev/null; then bad "guard: accepted bad path"; else ok "guard: refused bad path"; fi

echo "== kernel_sandbox: $pass passed, $fail failed =="; [ "$fail" -eq 0 ]
