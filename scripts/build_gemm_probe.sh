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
IRON="${IRON:-~/repositories/ns/amd/IRON}"
AIEBU_DIR="${AIEBU_DIR:-~/repositories/ns/amd/XRT-src/src/runtime_src/core/common/aiebu/build/Release/src/cpp/utils/asm}"
WEIGHTS="${WEIGHTS:-$REPO/artifacts/whisper-small/whisper_decoder}"
GEN="$REPO/route_b_kernels/decode_fused/gen_gemm_probe.py"

[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: $VENV_IRON/bin/python missing"; exit 1; }
[ -d "$IRON/iron" ] || { echo "ERROR: amd/IRON not at $IRON"; exit 1; }
[ -x "$AIEBU_DIR/aiebu-asm" ] || { echo "ERROR: aiebu-asm not at $AIEBU_DIR"; exit 1; }
[ -f "$WEIGHTS/L0/fc1.weight.npy" ] || { echo "ERROR: weights not at $WEIGHTS"; exit 1; }

# GEMM fusion-prefix + M-stationary are now COMMITS on the atassis/IRON:xdna2-asr fork branch (not .patch).
# Require the checkout to be on it (gemm-fusion-prefix needed for any GEMM under FusedMLIROperator;
# m-stationary is the opt-in --m-stationary mode, default N-stationary unchanged).
on="$(git -C "$IRON" rev-parse --abbrev-ref HEAD 2>/dev/null)"
[ "$on" = xdna2-asr ] || { echo "ERROR: $IRON must be on xdna2-asr (got '$on'). Run: git -C \"$IRON\" checkout xdna2-asr"; exit 1; }
echo "[build] IRON on xdna2-asr @ $(git -C "$IRON" rev-parse --short HEAD)"

export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_DIR:$PATH"
export PEANO_INSTALL_DIR="$VENV_IRON/lib/python3.14/site-packages/llvm-aie"
export PYTHONPATH="$IRON:$(dirname "$GEN")${PYTHONPATH:+:$PYTHONPATH}"

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
