#!/usr/bin/env bash
# build_peano_fast.sh -- build/rebuild a patched Peano (llvm-aie) fast enough to iterate.
#
# The prod Peano is a wheel (PEANO_DIST in toolchain.lock); we build locally only to CARRY
# PATCHES on the AIE backend. The workflow that matters is the patch->rebuild->test LOOP, so
# this script optimizes the INCREMENTAL rebuild, not the cold build:
#
#   1. Persistent build tree (build-fast/). We NEVER rm -rf it -- ninja rebuilds only what
#      changed. This is the whole game.
#   2. DYLIB (shared libLLVM.so / libclang-cpp.so). An AIE-backend one-file edit relinks ONE
#      .so instead of statically relinking the ~1 GB clang/llc/opt/lld -- the difference
#      between a >5 min and a <1 min loop. (Flags live in fb_llvm_fast_flags.)
#   3. mold + ccache + optimized tablegen + -j nproc (all from fast_build_env.sh).
#
# Cold-build reality + how we beat it (MEASURED 2026-07-15, 20 cores / 32 GB):
#   - True cold, EMPTY ccache: full clang;lld+AIE = 922 s (15m). This cold time is FRONT-END-parse-
#     bound, not optimizer-bound: lowering the build's own opt level (-O0) saves only ~12% while
#     shipping a >=1.7x SLOWER llc/clang + 5x tree bloat -- a bad trade, since kernel compile is a
#     per-iteration hot path. Keep -O3; beat cold via cache, not by cutting scope or opt. One-time
#     per machine; genuine compile physics. (On a KNOWN-cold first build, CCACHE_DISABLE=1 is ~15%
#     faster -- the ccache launcher's fork+hash costs with no hits to amortize.)
#   - WARM ccache (base_dir-normalized, set in fast_build_env.sh): a full nuke + reconfigure +
#     rebuild REPLAYS from cache -- measured 11 s (100% cache hit) for the full clang;lld toolchain.
#     So every "cold" AFTER the first (nuked build dir, reconfigure, or fresh clone SHARING the
#     in-tree .cache/ccache) is well under 5 min. This only works because CCACHE_BASEDIR makes
#     entries reusable across build trees -- without it every new tree is a cold miss.
#   - INCREMENTAL one-file AIE edit on a persistent tree = 1-4 s (dylib -> one .so relink).
#
# We build the FULL toolchain (clang;lld): clang is the FRONTEND that compiles kernel .cc files
# (Peano's clang++ is what aiecc/aircc invoke), so you need it for the real engine workflow. Because
# dylib puts the AIE backend in a shared libLLVM.so that clang, llc, opt and lld all link, you build
# the full toolchain ONCE and a backend patch then relinks only that .so (~1-4 s) -- the full clang
# picks up the patched backend for free. So there is no reason to skip clang. (To bootstrap ONLY
# llc/opt for a quick .ll/IR-path check without clang, pass explicit targets: `... llc opt` -- ninja
# builds just those; NOTE such a tree has no clang and cannot compile kernels.)
#
# Usage:
#   scripts/build_peano_fast.sh                 # full clang;lld toolchain (configure if needed)
#   scripts/build_peano_fast.sh llc opt         # build only these ninja targets
#   scripts/build_peano_fast.sh --bench-incremental
#                                               # touch one AIE-backend TU, rebuild, TIME it
#                                               # (validates the kill-if: warm one-file edit <5 min)
#
# Overrides (env):
#   LLVM_AIE_SRC   llvm-aie checkout (default: <workspace>/llvm-aie)
#   BUILD_DIR      build tree        (default: $LLVM_AIE_SRC/build-fast)
#   FB_LINK_JOBS   parallel link jobs (default 6; lower on tight RAM to avoid OOM)
#   FB_LLVM_PROJECTS  project list   (default "clang;lld")
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# fast-build levers: ccache, mold, -j nproc, and the fb_* cmake flag fragments.
# shellcheck source=/dev/null
. "$REPO/scripts/fast_build_env.sh"

LLVM_AIE_SRC="${LLVM_AIE_SRC:-$XDNA_WS/llvm-aie}"
BUILD_DIR="${BUILD_DIR:-$LLVM_AIE_SRC/build-fast}"
# Default target set: the tools a Peano dist actually invokes. With dylib they all share
# libLLVM.so, so this is cheap on the incremental path.
DEFAULT_TARGETS="llc opt clang lld"
# Representative AIE-backend TU touched by --bench-incremental (one .cpp -> libLLVM.so relink).
BENCH_TU="llvm/lib/Target/AIE/AIE2Subtarget.cpp"

if [ ! -f "$LLVM_AIE_SRC/llvm/CMakeLists.txt" ]; then
  echo "ERROR: no llvm-aie checkout at $LLVM_AIE_SRC (set LLVM_AIE_SRC)." >&2
  exit 1
fi

fb_print_status
echo "[build_peano_fast] src=$LLVM_AIE_SRC"
echo "[build_peano_fast] build=$BUILD_DIR"

# --- configure once (persistent tree; never rm -rf) --------------------------------------
configure() {
  if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "[build_peano_fast] build tree already configured -- reusing (persistent)."
    return 0
  fi
  echo "[build_peano_fast] configuring build-fast/ (one-time) ..."
  # shellcheck disable=SC2046  # fb_*_flags intentionally word-split into cmake args
  cmake -S "$LLVM_AIE_SRC/llvm" -B "$BUILD_DIR" $(fb_llvm_fast_flags)
}

# fractional wall-clock (seconds, 1 decimal). Integer $SECONDS rounds a fast incremental
# relink to a misleading "0s"; the dylib loop is single-digit seconds and we want to see it.
_fb_now() { date +%s.%N; }
_fb_elapsed() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.1f", b-a}'; }

# --- timed ninja build -------------------------------------------------------------------
timed_build() {
  local -a targets=("$@")
  [ "${#targets[@]}" -gt 0 ] || targets=($DEFAULT_TARGETS)
  echo "[build_peano_fast] building: ${targets[*]}"
  local t0; t0=$(_fb_now)
  ninja -C "$BUILD_DIR" -j "$FB_NPROC" "${targets[@]}"
  local dt; dt=$(_fb_elapsed "$t0" "$(_fb_now)")
  echo "[build_peano_fast] DONE in ${dt}s. Binaries: $BUILD_DIR/bin"
  return 0
}

# --- --bench-incremental: measure the loop that actually matters -------------------------
bench_incremental() {
  local tu="$LLVM_AIE_SRC/$BENCH_TU"
  [ -f "$tu" ] || { echo "ERROR: bench TU missing: $tu" >&2; exit 1; }
  # Realistic patch: change file CONTENT (a unique comment) so it is a genuine ccache MISS +
  # real recompile, not a `touch` that ccache would replay as a hit. Reverted on exit.
  echo "[build_peano_fast] warm-tree incremental probe: edit $BENCH_TU -> rebuild"
  local marker="// peano-build-speed incremental probe $$"
  cp "$tu" "$tu.fastprobe.bak"
  # shellcheck disable=SC2064
  trap "mv -f '$tu.fastprobe.bak' '$tu' 2>/dev/null || true" RETURN
  printf '\n%s\n' "$marker" >> "$tu"
  local t0; t0=$(_fb_now)
  ninja -C "$BUILD_DIR" -j "$FB_NPROC" $DEFAULT_TARGETS
  local dt; dt=$(_fb_elapsed "$t0" "$(_fb_now)")
  echo "[build_peano_fast] INCREMENTAL one-file edit: ${dt}s"
  if awk -v d="$dt" 'BEGIN{exit !(d < 300)}'; then
    echo "[build_peano_fast] PASS kill-if: warm one-file AIE edit < 5 min."
  else
    echo "[build_peano_fast] FAIL kill-if: ${dt}s >= 300s -- investigate (dylib on? static relink?)."
  fi
}

case "${1:-}" in
  --bench-incremental)
    [ -f "$BUILD_DIR/CMakeCache.txt" ] || { echo "ERROR: build first (no $BUILD_DIR)." >&2; exit 1; }
    bench_incremental
    ;;
  *)
    configure
    timed_build "$@"
    ;;
esac
