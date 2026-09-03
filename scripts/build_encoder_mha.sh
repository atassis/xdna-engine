#!/usr/bin/env bash
# Build the encoder-MHA xclbin AND record what built it.
#
# WHY THIS EXISTS. Measured 2026-09-03: the shipped StaticMHA_h20 artifact had NO tracked build
# driver. Nothing under scripts/ invoked gen_encoder_mha.py, the xclbin is gitignored, its kernel
# source mha.cc exists in TWO divergent versions across four IRON worktrees (md5 8ee49f1f51bb in
# wt-iron-pr-comment and wt-iron-kernel-obj-arch, c55d556ae523 in IRON and wt-iron-qwen3) with no
# record of which produced it, and a same-named 165951 B copy sat in build/ beside the 159167 B one
# the measurements actually used. A shipped-default kernel that cannot be rebuilt deterministically
# fails the recreatable-if-gitignored rule; worse, it makes any A/B on this kernel unsound, because
# the arms can differ in SOURCE as well as in the flag under test -- which is the confound that cost
# this project a day (a 06-29 kernel in a 09-03 A/B).
#
# The canonical IRON tree is whatever amd_paths.sh resolves IRON_DIR to (default $XDNA_WS/IRON),
# NOT the PR worktrees. Override with IRON_DIR=... to build an arm against a different tree -- and
# the stamp will say so, which is the point.
#
# Usage:  scripts/build_encoder_mha.sh [--heads 20] [--pipelines 8] [--out DIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

HEADS=20; PIPELINES=8; OUT="$REPO/artifacts/encoder_mha"
while [ $# -gt 0 ]; do
  case "$1" in
    --heads)     HEADS="$2"; shift 2 ;;
    --pipelines) PIPELINES="$2"; shift 2 ;;
    --out)       OUT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

. "$REPO/scripts/amd_paths.sh"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: $VENV_IRON/bin/python missing" >&2; exit 1; }
[ -d "$IRON_DIR/iron" ]        || { echo "ERROR: amd/IRON not at $IRON_DIR" >&2; exit 1; }

GEN="$REPO/route_b_kernels/decode_fused/gen_encoder_mha.py"
[ -f "$GEN" ] || { echo "ERROR: generator missing: $GEN" >&2; exit 1; }

# The two identities that decide what the object IS, captured BEFORE the build so a failed build
# still leaves the question answered.
IRON_REV="$(git -C "$IRON_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') @ $(git -C "$IRON_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
MHA_CC="$IRON_DIR/aie_kernels/aie2p/mha.cc"
MHA_MD5="$(md5sum "$MHA_CC" 2>/dev/null | cut -d' ' -f1 || echo 'MISSING')"
LOCK_ID="$(sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$REPO/toolchain.lock" | sha256sum | cut -c1-12)"

export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:${AIEBU_ASM_DIR:-}:$PATH"
export PEANO_INSTALL_DIR="$VENV_IRON/lib/python3.14/site-packages/llvm-aie"
# The generator's own directory must be importable (newstack_compat) even though we do NOT run from
# there -- see WORK below. STATIC_DESIGN resolves via __file__, so it needs no cwd.
export PYTHONPATH="$IRON_DIR:$REPO/route_b_kernels/decode_fused${PYTHONPATH:+:$PYTHONPATH}"

# PER-BUILD build_dir, and this is load-bearing. iron's AIEContext() takes build_dir from $(cwd)/build.
# Running every arm from route_b_kernels/decode_fused makes them all share ONE long-lived cache, and
# the cache returns the PREVIOUS artifact byte-identically -- measured 2026-09-03 while writing this
# script: four --pipelines values (4/6/8/10) produced the same md5 and emitted no MLIR. That is
# a cache hit masquerading as a build, and it would make this script certify provenance it had not
# actually rebuilt. Own cwd per build; the tells are an instant build and a missing .mlir.
WORK="${WORK:-$OUT/.build}"
mkdir -p "$WORK"

echo "[enc-mha] IRON      $IRON_DIR ($IRON_REV)"
echo "[enc-mha] mha.cc    ${MHA_MD5:0:12}  $MHA_CC"
echo "[enc-mha] toolchain $LOCK_ID"
echo "[enc-mha] building  heads=$HEADS pipelines=$PIPELINES -> $OUT"

# REFUSE to overwrite an existing artifact without --force. The shipped 159167 B StaticMHA_h20 is
# NOT reproducible from this tree (measured 2026-09-03: a clean build of the same params from the
# canonical IRON gives 159023 B, and pipelines / out-path / build_dir-path / nondeterminism are all
# ruled out). Overwriting it would destroy the object every measured encoder-MHA result was taken
# against, and it could not be rebuilt. Build somewhere else and compare first.
EXIST="$(ls -1 "$OUT"/StaticMHA_h${HEADS}_*.xclbin 2>/dev/null | head -1 || true)"
if [ -n "$EXIST" ] && [ -z "${FORCE:-}" ]; then
  echo "ERROR: $EXIST already exists ($(stat -c%s "$EXIST") B)." >&2
  echo "       This artifact is not reproducible; overwriting loses it. Use --out <scratch> to" >&2
  echo "       build a comparison arm, or FORCE=1 if you really mean to replace it." >&2
  exit 3
fi

mkdir -p "$OUT"
( cd "$WORK" && "$VENV_IRON/bin/python" "$GEN" \
    --out "$OUT" --heads "$HEADS" --pipelines "$PIPELINES" )

# Self-describing artifact: the stamp answers "what built this" without a git archaeology session.
XB="$(ls -1 "$OUT"/StaticMHA_h${HEADS}_*.xclbin 2>/dev/null | head -1)"
{
  echo "iron_dir=$IRON_DIR"
  echo "iron_rev=$IRON_REV"
  echo "mha_cc_md5=$MHA_MD5"
  echo "toolchain_lock_id=$LOCK_ID"
  echo "heads=$HEADS pipelines=$PIPELINES"
  echo "build_dir=$WORK"
  [ -n "$XB" ] && echo "xclbin=$(basename "$XB") bytes=$(stat -c%s "$XB") md5=$(md5sum "$XB" | cut -d' ' -f1)"
} > "$OUT/.provenance"
echo "[enc-mha] stamped   $OUT/.provenance"
[ -n "$XB" ] && echo "[enc-mha] artifact  $(basename "$XB") $(stat -c%s "$XB") B"
