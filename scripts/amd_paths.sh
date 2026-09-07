#!/usr/bin/env bash
# amd_paths.sh -- single relocatable anchor for the AMD/Xilinx upstream checkouts.
#
# SOURCE this from any script that needs the IRON / XRT / mlir-air checkouts. It
# derives the umbrella-workspace root from THIS file's own location (the upstream
# checkouts live as siblings of the engine repo under the workspace), so the whole
# tree is RELOCATABLE -- no hardcoded $HOME/absolute paths. Every var is overridable
# from the environment (export IRON_DIR=... before sourcing to point elsewhere).
#
#   source "$(dirname "${BASH_SOURCE[0]}")/amd_paths.sh"
#   ... use "$IRON_DIR" / "$XRT_SRC_DIR" / "$MLIR_AIR_DIR" / "$AIEBU_ASM_DIR"
#
# Layout it assumes:  <workspace>/{xdna-engine/scripts/amd_paths.sh, IRON, XRT-src, mlir-air, ...}
#                     (upstream checkouts are flat siblings of the engine repo at the workspace root)

# workspace root = parent of the engine repo (this file lives in <engine>/scripts/)
XDNA_WS="${XDNA_WS:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
export XDNA_WS

# IRON_DIR points at the INTEGRATION STACK, not the bare fork checkout.
#
# The shared $XDNA_WS/IRON checkout sits on whatever branch it was last left on and carries neither
# iron/operators/tmatvec/ nor iron/operators/gemv/quant.py -- both imported at module scope by
# designs/decode_fused/gen_llm_decode.py. So the documented build command for the LLM decode failed
# at import with the default resolution, and every caller had to know to pass IRON=<worktree>.
#
# wt-iron-integ is the integration-stack model every other fork here already uses: latest upstream
# as the base, our carries cherry-picked on top, dropped as they land upstream. Rebased 2026-09-07
# onto upstream/devel deb6e1e with all carries applied and gated -- device-free tests, bf16-oracle
# parity, DDR bytes, interleaved timing, and a byte-identical decode ELF against the pre-rebase
# build. See the journal task iron-back-onto-the-integration-stack-model.
#
# Still overridable: `IRON=<dir>` on any caller, or IRON_DIR in the environment.
export IRON_DIR="${IRON_DIR:-$XDNA_WS/wt-iron-integ}"

# amd/IRON's two aiecc rules default AIECC_JOBS to '1', so every design's per-core
# compiles run one at a time. On the 24-core encoder-MHA design that is 7.7 s against
# 6.0 s at aiecc's own auto-detect (0); nothing above 8 helps. Set here rather than in
# the IRON checkout: IRON is shared by every worktree on this box and is not ours to
# edit in place, and it already reads this from the environment.
#
# Safe because -j does not change what aiecc produces -- MEASURED on that design,
# insts.bin and all 24 per-core ELFs are byte-identical between -j1 and -j16, and
# input_with_addresses.mlir differs only in the work-dir path it embeds, which two runs
# at the SAME -j differ in too.
export AIECC_JOBS="${AIECC_JOBS:-0}"

# Route kernel .cc compiles through ccache. Measured on 8 kernels: 6.72 s of misses
# against 0.09 s of hits, objects identical. It composes with the intrinsics PCH,
# and it is the only thing that removes DUPLICATED compilation -- batching every
# kernel into one clang invocation was measured 32% SLOWER than separate calls,
# and the per-invocation floor with the PCH is only ~0.025 s, so there is nothing
# for a shared process to recover.
#
# time_macros is not optional and not cosmetic: ccache silently refuses to cache
# ANY compile that uses -include-pch without it -- the call is counted
# "uncacheable", not a miss, and the only explanation appears under CCACHE_DEBUG.
# Set both levers and get neither. What it makes sloppy is __DATE__/__TIME__/
# __TIMESTAMP__, and nothing under aie_kernels/ or in aie_api uses them (checked).
# Appended rather than assigned, so an existing sloppiness is kept.
if command -v ccache >/dev/null 2>&1; then
  # ${VAR-default}, NOT ${VAR:-default}: the colon form substitutes an explicitly
  # EMPTY value too, so `AIE_KERNEL_COMPILER_LAUNCHER= ` would silently be turned
  # back into ccache and the documented off switch would not exist.
  export AIE_KERNEL_COMPILER_LAUNCHER="${AIE_KERNEL_COMPILER_LAUNCHER-ccache}"
  case ",${CCACHE_SLOPPINESS:-}," in
    *,time_macros,*) : ;;
    ,,) export CCACHE_SLOPPINESS="time_macros" ;;
    *) export CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS},time_macros" ;;
  esac
fi
export XRT_SRC_DIR="${XRT_SRC_DIR:-$XDNA_WS/XRT-src}"
export AIEBU_ASM_DIR="${AIEBU_ASM_DIR:-$XRT_SRC_DIR/src/runtime_src/core/common/aiebu/build/Release/src/cpp/utils/asm}"

# mlir-air / llvm-aie are NOT defaulted here: in setup_amd_toolchains.sh an EMPTY
# MLIR_AIR_DIR/LLVM_AIE_DIR is the "do not patch this repo" gate. Their canonical
# location (when you do opt in) is $XDNA_WS/{mlir-air,llvm-aie}.

# iron_require_api -- gate the shared IRON checkout on the API SURFACE the caller needs,
# NOT on a branch name. Branch names have drifted twice (xdna2-asr -> integration-stack) and
# each drift broke every script that hard-required the old name, while the checkout was fine.
# What a build actually depends on is whether the symbols its generators import are present.
#
#   iron_require_api <label> <path-under-IRON>:<literal-symbol> ...
#
# Names every missing symbol (not just the first) and prints the branch for the report line.
# Checks ${IRON:-$IRON_DIR}, i.e. the tree the CALLER will actually build with. Every caller uses
# the same `IRON="${IRON:-$IRON_DIR}"` idiom and then puts $IRON on PYTHONPATH, so gating $IRON_DIR
# verified one tree and built with another whenever IRON was overridden -- silently, since the
# report line said "API surface verified" either way. Found 2026-09-06 by an override that pointed
# at a worktree while the shared checkout satisfied the gate.
iron_require_api() {
  local label="$1"; shift
  local dir="${IRON:-$IRON_DIR}"
  local on spec f sym missing=0
  on="$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  for spec in "$@"; do
    f="${spec%%:*}"; sym="${spec#*:}"
    grep -qF -- "$sym" "$dir/$f" 2>/dev/null && continue
    echo "ERROR: $dir ('$on') lacks '$sym' in $f -- required by $label" >&2
    missing=1
  done
  [ "$missing" = 0 ] || {
    echo "  the fork line carrying it is 'integration-stack'; 'xdna2-asr' is its stale predecessor." >&2
    return 1
  }
  echo "$on @ $(git -C "$dir" rev-parse --short HEAD 2>/dev/null)"
}

# iron_require_pin -- the IRON tree must CONTAIN toolchain.lock's IRON_FORK_COMMIT.
#
# Ancestry, not equality: every IRON worktree here carries local commits on top of the pinned
# floor, so an exact-sha gate would fail all of them. The floor is the merge-base of every IRON
# line in the workspace, so "contains it" means "descends from the state we all agreed on".
#
# An ABSENT pin is not a pass -- a lock that forgot the key must not read as unlocked.
iron_require_pin() {
  local dir="${IRON:-$IRON_DIR}"
  local lock="${IRON_LOCK:-$(dirname "${BASH_SOURCE[0]:-$0}")/../toolchain.lock}"
  local want
  want="$(sed -n 's/^IRON_FORK_COMMIT=\([0-9a-f]\{7,\}\).*/\1/p' "$lock" 2>/dev/null | head -1)"
  [ -n "$want" ] || { echo "ERROR: no IRON_FORK_COMMIT in $lock -- refusing to build unpinned" >&2; return 1; }
  git -C "$dir" cat-file -e "$want^{commit}" 2>/dev/null || {
    echo "ERROR: $dir does not have pinned IRON_FORK_COMMIT $want (fetch the fork?)" >&2; return 1; }
  git -C "$dir" merge-base --is-ancestor "$want" HEAD 2>/dev/null || {
    echo "ERROR: $dir HEAD ($(git -C "$dir" rev-parse --short HEAD)) does not contain pinned $want." >&2
    echo "  Rebase onto the pin, or re-pin toolchain.lock to a new merge-base if the floor moved." >&2
    return 1; }
}
