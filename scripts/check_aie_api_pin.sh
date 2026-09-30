#!/usr/bin/env bash
# Are the aie_api headers kernels COMPILE against the ones toolchain.lock PINS?
#
# They should be: toolchain_up.sh links `$INST/build/include/aie_api` at the instance's own pinned
# tree. It RATCHETS rather than asserting equality:
# the state is recorded in a baseline and any CHANGE to it fails, whichever direction it moves.
# A pin bump that shifts the headers under the kernels then announces itself here instead of being
# discovered later by grepping a stale copy and drawing a conclusion about the wrong tree.
#
#   check_aie_api_pin.sh [--write-baseline]      exit 0 = matches baseline, 1 = drifted
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="$REPO/scripts/aie_api_pin_baseline.txt"
# SOURCE the lock, never parse it: MLIR_AIE_FORK_COMMIT's line carries a multi-hundred-word inline
# comment, so `cut -d= -f2` returns the essay along with the sha.
set -a; . "$REPO/toolchain.lock"; set +a

# Hash a header tree by RELATIVE path + content, so two trees at different prefixes are comparable.
_tree_hash() {
  local dir="$1"
  [ -d "$dir" ] || { echo "MISSING"; return 0; }
  ( cd "$dir" && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16 )
}

# A MISSING side means this checkout cannot answer the question (a worktree without the mlir-aie
# submodule, an unbuilt instance) -- refuse rather than record "MISSING" as if it were a state.
_require_present() {
  case "$1" in MISSING)
    echo "[aie_api_pin] $2 not present at $3 -- run from the main checkout with a built instance" >&2
    exit 2 ;;
  esac
}

INST="$("$REPO/scripts/toolchain_up.sh")"
linked="$INST/build/include/aie_api"
pinned="$REPO/mlir-aie/third_party/aie_api/include/aie_api"

# Two submodule identities, and they answer different questions:
#   lock -- the aie_api commit MLIR_AIE_FORK_COMMIT records, i.e. what the repo CLAIMS to pin;
#   disk -- what is actually checked out under third_party/aie_api, i.e. what `$pinned` above hashed.
# Deliberately NOT the gitlink at the mlir-aie checkout's own HEAD: that tracks whatever branch the
# shared checkout sits on, which does not affect a build (toolchain_up builds from a clean worktree
# at MLIR_AIE_FORK_COMMIT), so folding it in would churn this baseline on unrelated branch switches.
_short() { case "$1" in [0-9a-f][0-9a-f]*) printf '%s' "${1:0:12}" ;; *) printf 'unknown' ;; esac; }
lock_sha="$(_short "$(git -C "$REPO/mlir-aie" rev-parse --verify -q "${MLIR_AIE_FORK_COMMIT:-HEAD}:third_party/aie_api" 2>/dev/null || true)")"
disk_sha="$(_short "$(git -C "$REPO/mlir-aie/third_party/aie_api" rev-parse --verify -q HEAD 2>/dev/null || true)")"

linked_h="$(_tree_hash "$linked")"; _require_present "$linked_h" "instance aie_api" "$linked"
pinned_h="$(_tree_hash "$pinned")"; _require_present "$pinned_h" "pinned aie_api" "$pinned"
now="linked=$linked_h pinned=$pinned_h lock_sha=$lock_sha disk_sha=$disk_sha"

# Direct proof the compiler resolved OUR tree, not just a tree-hash comparison to a baseline it could
# ratchet forward on a bad bump. __XDNA_AIE_API_FORK__ is a sentinel only atassis/aie_api defines
# (proposed as commit 1d91fa0 on its local/pin-sentinel-20260930 branch, not yet in this pin or
# pushed) -- upstream Xilinx/aie_api and the vendored mlir_aie wheel copy never define it. Once a
# pin carries it this block should be made fatal; until then it is informational, since MLIR_AIE_FORK_COMMIT
# 8e3958b596a's third_party/aie_api (1db7142) predates the sentinel commit.
_probe_sentinel() {
  local peano="$REPO/.venv-iron/lib/python3.14/site-packages/llvm-aie/bin/clang++"
  [ -x "$peano" ] || { echo "[aie_api_pin] sentinel probe skipped: no Peano clang++ at $peano"; return 0; }
  local tu; tu="$(mktemp -d)/p.cc"
  printf '%s\n' '#include <aie_api/aie.hpp>' \
    '#ifdef __XDNA_AIE_API_FORK__' 'static_assert(true, "fork sentinel present");' '#else' \
    '#error "no __XDNA_AIE_API_FORK__ -- aie_api did not resolve the fork pin"' '#endif' > "$tu"
  if "$peano" -I"$linked" --system-header-prefix=aie_api/ -std=c++20 -D__AIE_API_AIE_ADF_HPP__ \
       --target=aie2p-none-unknown-elf -w -fsyntax-only "$tu" 2>/dev/null; then
    echo "[aie_api_pin] sentinel probe: __XDNA_AIE_API_FORK__ present at $linked"
  else
    echo "[aie_api_pin] sentinel probe: __XDNA_AIE_API_FORK__ NOT in the current pin (expected until" \
         "the fork commit above is picked up) -- tree-hash ratchet below is the live gate"
  fi
  rm -rf "$(dirname "$tu")"
}
_probe_sentinel

if [ "${1:-}" = "--write-baseline" ]; then
  printf '%s\n' "$now" > "$BASELINE"
  echo "[aie_api_pin] baseline written: $now"
  exit 0
fi

[ -f "$BASELINE" ] || { echo "[aie_api_pin] no baseline; record one with --write-baseline"; exit 1; }
was="$(cat "$BASELINE")"
if [ "$now" = "$was" ]; then
  echo "[aie_api_pin] OK (unchanged): $now"
  exit 0
fi
cat >&2 <<EOF
[aie_api_pin] DRIFT -- the aie_api header situation changed since the baseline.
  was: $was
  now: $now
  linked (what kernels compile against): $(readlink -f "$linked" 2>/dev/null || echo "$linked")
  pinned (what toolchain.lock describes): $pinned
If you moved MLIR_AIE_FORK_COMMIT, the kernels are now compiling against headers from a different
version than the lock claims -- re-gate the kernels before trusting a build. If you deliberately
repointed the symlink, that is a behaviour change (it has measurably cost rope_lut before): re-run
the kernel gates, then record the new state with: scripts/check_aie_api_pin.sh --write-baseline
EOF
exit 1
