#!/usr/bin/env bash
# Build the lever-3 vector-(b) Milestone-0 batching probe ELFs (single fc1 GEMM at a sweep of N).
# Compile-only (no NPU). Mirrors build_deepc_decode.sh env. See gen_gemm_probe.py for the rationale.
#
#   bash scripts/build_gemm_probe.sh ["16 32 64 128"] [OUT_ROOT]
#   $1  space-separated N values   default "16 32 64 128"
#   $2  output root                default <repo>/artifacts/gemm_probe   (per-N: <root>[_c<cols>]_N<n>)
#
# Env: NUM_COLS (default 1) = num_aie_columns; TILE_N (default 16). Full-array sweep:
#   NUM_COLS=8 bash scripts/build_gemm_probe.sh "128 256 512"   (min N = TILE_N*NUM_COLS = 128)
# Other env (same defaults as build_deepc_decode.sh): VENV_IRON, IRON, AIEBU_DIR, WEIGHTS
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS="${1:-16 32 64 128}"
OUT_ROOT="${2:-$REPO/artifacts/gemm_probe}"
NUM_COLS="${NUM_COLS:-1}"
TILE_N="${TILE_N:-16}"
SUF=""; [ "$NUM_COLS" != "1" ] && SUF="_c${NUM_COLS}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
. "$REPO/scripts/amd_paths.sh"       # -> IRON_DIR, AIEBU_ASM_DIR (relocatable; env-overridable)
IRON="${IRON:-$IRON_DIR}"
AIEBU_DIR="${AIEBU_DIR:-$AIEBU_ASM_DIR}"
WEIGHTS="${WEIGHTS:-$REPO/artifacts/whisper-small/whisper_decoder}"
GEN="$REPO/route_b_kernels/decode_fused/gen_gemm_probe.py"

[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: $VENV_IRON/bin/python missing"; exit 1; }
[ -d "$IRON/iron" ] || { echo "ERROR: amd/IRON not at $IRON"; exit 1; }
[ -x "$AIEBU_DIR/aiebu-asm" ] || { echo "ERROR: aiebu-asm not at $AIEBU_DIR"; exit 1; }
[ -f "$WEIGHTS/L0/fc1.weight.npy" ] || { echo "ERROR: weights not at $WEIGHTS"; exit 1; }

# GEMM fusion-prefix + M-stationary need the fork API (gemm-fusion-prefix for any GEMM under
# FusedMLIROperator; m-stationary is the opt-in --m-stationary mode, default N-stationary unchanged).
# This script pioneered checking the API surface instead of a branch name, after names drifted once
# (2026-07-30, probe worktree); iron_require_api in amd_paths.sh is that check, now shared.
iron_at="$(iron_require_api "gen_gemm_probe.py" \
  "iron/common/sequence.py:class OperatorSequence" \
  "iron/operators/gemm/op.py:class GEMM" \
  "iron/operators/gemm/op.py:m_stationary")" || exit 1
echo "[build] IRON on $iron_at (API surface verified)"

export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_DIR:$PATH"
export PEANO_INSTALL_DIR="$VENV_IRON/lib/python3.14/site-packages/llvm-aie"

# Resolve `aie` and aiecc from the FORK INSTANCE at the committed pin, as build_deepc_decode.sh does.
# Without $INST/python first, `aie` resolved through .venv-iron's aie.pth, which hardcodes ONE instance
# dir -- so this script silently built against whatever instance that file last pointed at rather than
# the pin, and said nothing. Measured 2026-08-17: aie.pth -> bf76a7f17b5d while the pin was 185212afd5ca.
INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON:$(dirname "$GEN")${PYTHONPATH:+:$PYTHONPATH}"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
[ -x "$AIECC_PATH" ] || { echo "ERROR: instance aiecc not at $AIECC_PATH (run scripts/toolchain_up.sh)"; exit 1; }

for N in $NS; do
  OUT="${OUT_ROOT}${SUF}_N${N}"
  WORK="$(mktemp -d)"   # amd/IRON writes build/ intermediates under CWD
  mkdir -p "$OUT"
  echo "=== building GEMM probe N=$N cols=$NUM_COLS -> $OUT (work=$WORK) ==="
  ( cd "$WORK" && "$VENV_IRON/bin/python" "$GEN" --weights "$WEIGHTS" --N "$N" --num-cols "$NUM_COLS" --tile-n "$TILE_N" --out "$OUT" ${TILE_M:+--tile-m "$TILE_M"} ${FUSE_RESIDUAL:+--fuse-residual} ${M_STATIONARY:+--m-stationary} )
  rm -rf "$WORK"
  echo "    elf=$(du -h "$OUT/gemmprobe.elf" | cut -f1)"
done
echo "[build] done: ${OUT_ROOT}${SUF}_N{$(echo $NS | tr ' ' ',')}"
