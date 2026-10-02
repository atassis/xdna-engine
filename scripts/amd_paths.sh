#!/usr/bin/env bash
# amd_paths.sh -- single anchor for the AMD/Xilinx upstream checkouts (IRON / XRT /
# mlir-air). SOURCE this from any script that needs them. Every var is overridable
# from the environment (export IRON_DIR=... before sourcing to point elsewhere).
#
#   source "$(dirname "${BASH_SOURCE[0]}")/amd_paths.sh"
#   ... use "$IRON_DIR" / "$XRT_SRC_DIR" / "$MLIR_AIR_DIR" / "$AIEBU_ASM_DIR"
#
# IRON defaults to this repository's pinned third_party/iron submodule. IRON_DIR and IRON
# are explicit development overrides. A caller that never touches IRON does not need it initialized.
# Per-machine locations (XRT_SRC_DIR, AIEBU_ASM_DIR, ...) come from config/local.env.
_AMD_PATHS_DIR="${_AMD_PATHS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
. "$_AMD_PATHS_DIR/lib/data_root.sh"
export IRON_DIR="${IRON_DIR:-${IRON:-$REPO/third_party/iron}}"


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
# Same property for the runtime-sequence partitions (36/36 instruction streams identical on the
# 48-layer gemma4 prefill, toolchain.lock's mlir-aie note); IRON passes the flag when this is 1.
export AIECC_PARTITION_RUNTIME_SEQUENCES="${AIECC_PARTITION_RUNTIME_SEQUENCES:-1}"

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
# XRT_SRC_DIR/MLIR_AIR_DIR/LLVM_AIE_DIR are not defaulted here. An empty MLIR_AIR_DIR/LLVM_AIE_DIR
# also doubles as setup_amd_toolchains.sh's "do not patch this repo" gate.
export XRT_SRC_DIR="${XRT_SRC_DIR:-}"
export AIEBU_ASM_DIR="${AIEBU_ASM_DIR:-${XRT_SRC_DIR:+$XRT_SRC_DIR/src/runtime_src/core/common/aiebu/build/Release/src/cpp/utils/asm}}"

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
  [ -n "$dir" ] || { echo "ERROR: IRON_DIR is not set. Point it at your IRON checkout: export IRON_DIR=/path/to/IRON" >&2; return 1; }
  local on spec f sym missing=0
  on="$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  for spec in "$@"; do
    f="${spec%%:*}"; sym="${spec#*:}"
    # A spec of the form `file:def fn(kwarg[,kwarg...])` is checked by PARSING the file and
    # asking whether `fn` accepts those arguments. `grep -qF` alone answers "does this string
    # appear", which is a different question and passed a tree whose `quantize_weight` existed
    # with the wrong signature -- K019, a gate that cannot fail the case it was written for.
    case "$sym" in
      "def "*"("*")")
        if python3 - "$dir/$f" "$sym" <<'PYEOF'
import ast, sys
path, spec = sys.argv[1], sys.argv[2]
fn = spec[4:spec.index("(")].strip()
want = [a.strip() for a in spec[spec.index("(") + 1:spec.rindex(")")].split(",") if a.strip()]
try:
    tree = ast.parse(open(path).read())
except (OSError, SyntaxError):
    sys.exit(1)
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn:
        a = node.args
        have = {x.arg for x in a.args + a.posonlyargs + a.kwonlyargs}
        if a.kwarg or all(w in have for w in want):
            sys.exit(0)
sys.exit(1)
PYEOF
        then continue; fi ;;
      *)
        grep -qF -- "$sym" "$dir/$f" 2>/dev/null && continue ;;
    esac
    echo "ERROR: $dir ('$on') lacks '$sym' in $f -- required by $label" >&2
    missing=1
  done
  [ "$missing" = 0 ] || {
    echo "  the fork line carrying it is 'integration-stack'; 'xdna2-asr' is its stale predecessor." >&2
    return 1
  }
  echo "$on @ $(git -C "$dir" rev-parse --short HEAD 2>/dev/null)"
}

iron_identity_is_allowed() {
  local want="$1" identity="$2" got
  case "$identity" in
    "pinned:$want") return 0 ;;
    override:*)
      [[ "$identity" =~ ^override:[0-9a-f]{40}$ ]] && [ "${IRON_ALLOW_UNPINNED:-}" = 1 ] && return 0 ;;
    dirty:*)
      [[ "$identity" =~ ^dirty:([0-9a-f]{40}):([0-9a-f]{64})$ ]] || return 1
      got="${BASH_REMATCH[1]}"
      [ "${IRON_ALLOW_DIRTY:-}" = 1 ] && \
        { [ "$got" = "$want" ] || [ "${IRON_ALLOW_UNPINNED:-}" = 1 ]; } && return 0 ;;
  esac
  return 1
}

iron_require_snapshot() {
  local want="$1" dir="$2" snapshot="${IRON_SOURCE_SNAPSHOT:-}"
  local manifest="${IRON_SOURCE_MANIFEST:-}" verifier="$_AMD_PATHS_DIR/buildstore/verify_iron_snapshot.py"
  local identity
  [ -r "$snapshot" ] && [ -r "$manifest" ] && [ -f "$verifier" ] || return 1
  identity="$(python3 "$verifier" --snapshot "$snapshot" --manifest "$manifest" --dir "$dir")" || return 1
  iron_identity_is_allowed "$want" "$identity" || return 1
  export IRON_DIR="$dir" IRON_SOURCE_IDENTITY="$identity"
}

iron_dirty_hash() {
  local dir="$1" untracked_hash
  untracked_hash="$(
    set -o pipefail
    git -C "$dir" ls-files --others --exclude-standard -z |
      while IFS= read -r -d '' path; do
        blob="$(git -C "$dir" hash-object --no-filters -- "$path")" || exit 1
        printf '%s\t%s\n' "$path" "$blob"
      done | sha256sum | cut -d' ' -f1
  )" || return 1
  { git -C "$dir" diff --binary HEAD --; printf '%s\n%s\n' "--untracked--" "$untracked_hash"; } |
    sha256sum | cut -d' ' -f1
}

iron_require_source() {
  local dir="${IRON:-$IRON_DIR}"
  local lock="${IRON_LOCK:-$(dirname "${BASH_SOURCE[0]:-$0}")/../toolchain.lock}"
  local want got dirty dirty_hash identity
  want="$(sed -n 's/^IRON_SOURCE_COMMIT=\([0-9a-f]\{40\}\).*/\1/p' "$lock" 2>/dev/null | head -1)"
  [ -n "$want" ] || { echo "ERROR: no IRON_SOURCE_COMMIT in $lock -- refusing to build." >&2; return 1; }
  if ! { [ -d "$dir" ] && git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; }; then
    iron_require_snapshot "$want" "$dir" && return 0
    echo "ERROR: IRON dependency is missing or uninitialized at $dir. Run: git submodule update --init --recursive" >&2
    return 1
  fi
  got="$(git -C "$dir" rev-parse HEAD 2>/dev/null)" || return 1
  if [ "$got" != "$want" ] && [ "${IRON_ALLOW_UNPINNED:-}" != 1 ]; then
    echo "ERROR: IRON source is $got, expected $want. Set IRON_ALLOW_UNPINNED=1 only for a recorded development override." >&2
    return 1
  fi
  dirty="$(git -C "$dir" status --porcelain --untracked-files=all 2>/dev/null)"
  if [ -n "$dirty" ]; then
    [ "${IRON_ALLOW_DIRTY:-}" = 1 ] || {
      echo "ERROR: IRON source at $dir is dirty. Set IRON_ALLOW_DIRTY=1 only for a recorded development override." >&2
      return 1
    }
    dirty_hash="$(iron_dirty_hash "$dir")" || {
      echo "ERROR: failed to fingerprint dirty IRON source at $dir." >&2
      return 1
    }
    identity="dirty:$got:$dirty_hash"
  elif [ "$got" = "$want" ]; then
    identity="pinned:$got"
  else
    identity="override:$got"
  fi
  export IRON_DIR="$dir" IRON_SOURCE_IDENTITY="$identity"
}

iron_require_pin() {
  iron_require_source
}

iron_require_fused_attn() {
  iron_require_source || return 1
  [ -f "$IRON_DIR/aie_kernels/aie2p/fused_attn.cc" ] || {
    echo "ERROR: pinned IRON source lacks aie_kernels/aie2p/fused_attn.cc: $IRON_DIR" >&2
    return 1
  }
}

# aiecc_require_pin [path] -- the aiecc that will RUN must be the one built from toolchain.lock's
# MLIR_AIE_FORK_COMMIT. Equality, not ancestry (the opposite of iron_require_pin above): aiecc is a
# built binary, not a source tree, so "descends from the pin" says nothing about what is inside it.
#
# The hole this closes: every build script resolves the compiler as ${AIECC_PATH:-$INST/bin/aiecc},
# so an env var silently replaces the pinned toolchain, and provenance records the LOCK rather than
# the binary -- an artifact built by another compiler is indistinguishable from a pinned one.
# aiecc has always self-reported its git SHA and mlir-aie's _tool_identity has always read it, but
# only as a cache key, so a wrong compiler minted a fresh cache namespace instead of failing.
#
# AIECC_PIN_OVERRIDE must NAME the sha it accepts, so it cannot be exported once and forgotten:
# it goes stale the moment the binary changes. An absent sha is not a pass.
# Captured when this file is SOURCED: inside a function BASH_SOURCE resolves
# against however the caller spelled the source path, so a relative `. scripts/amd_paths.sh` lost
# the repo root and the check failed closed on a lock it simply could not find.
aiecc_require_pin() {
  local bin="${1:-${AIECC_PATH:-}}"
  local lock="${MLIR_AIE_LOCK:-$_AMD_PATHS_DIR/../toolchain.lock}"
  local want got
  [ -x "$bin" ] || { echo "ERROR: aiecc_require_pin: no aiecc at '${bin:-<unset>}'" >&2; return 1; }
  want="$(sed -n 's/^MLIR_AIE_FORK_COMMIT=\([0-9a-f]\{7,\}\).*/\1/p' "$lock" 2>/dev/null | head -1)"
  [ -n "$want" ] || { echo "ERROR: no MLIR_AIE_FORK_COMMIT in $lock -- refusing to build unpinned" >&2; return 1; }
  got="$("$bin" --version 2>/dev/null | sed -n 's/^[[:space:]]*git SHA:[[:space:]]*\([0-9a-f]\{7,\}\).*/\1/p' | head -1)"
  [ -n "$got" ] || { echo "ERROR: $bin printed no git SHA -- cannot identify it, refusing to build" >&2; return 1; }
  case "$want" in "$got"*) return 0 ;; esac
  [ "${AIECC_PIN_OVERRIDE:-}" = "$got" ] && {
    echo "[aiecc] WARNING: running UNPINNED aiecc $got (pin $want) by AIECC_PIN_OVERRIDE" >&2; return 0; }
  echo "ERROR: aiecc is not the pinned compiler." >&2
  echo "  binary: $bin" >&2
  echo "  is:     $got" >&2
  echo "  pin:    $want  (MLIR_AIE_FORK_COMMIT in $lock)" >&2
  echo "  Unset AIECC_PATH to use the pinned instance, or re-pin toolchain.lock to land the fix" >&2
  echo "  you want in the pin. To accept this binary deliberately: AIECC_PIN_OVERRIDE=$got" >&2
  return 1
}

# aiecc_resolve [instance_dir] -- the ONLY way a build should obtain an aiecc.
#
# Thirteen scripts each spelled `export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"`, so the choice
# of compiler had thirteen owners and no checker. It keeps the env override (AIECC_PIN_OVERRIDE is
# the deliberate way past) but no longer lets it pass unexamined.
aiecc_resolve() {
  local inst="${1:-${MLIR_AIE_INSTANCE:-}}"
  [ -n "$inst" ] || { echo "ERROR: aiecc_resolve: no instance dir (pass one, or set MLIR_AIE_INSTANCE)" >&2; return 1; }
  export AIECC_PATH="${AIECC_PATH:-$inst/bin/aiecc}"
  aiecc_require_pin
}
