#!/usr/bin/env bash
# Build every legal GEMM tiling for the qwen3-0.6b prefill shapes, device-free, and emit a
# manifest to time. Nothing here opens /dev/accel: AIE_DEVICE=npu2 pins the target.
#
#   bash scripts/sweep_gemm_tiles.sh                       # census + build, default shapes
#   PLAN=1 bash scripts/sweep_gemm_tiles.sh                # census only, no toolchain needed
#   MAX_PER_SHAPE=0 bash scripts/sweep_gemm_tiles.sh       # build EVERY legal candidate (~550)
#   bash scripts/sweep_gemm_tiles.sh --shape 256x1024x2048 --label q
#
# Any extra argument is passed straight through to designs/decode_fused/sweep_gemm_tiles.py.
#
# Env: BATCH (256), SEQ (2048), OUT, MAX_PER_SHAPE (16), PLAN, VENV_IRON, IRON.
# The default cap takes a stratified stride over the legal set -- see `subsample()` there; it is
# a spread, not a ranking, and MAX_PER_SHAPE=0 removes it.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
BATCH="${BATCH:-256}"; SEQ="${SEQ:-2048}"
OUT="${OUT:-/mnt/data/xdna/scratch/gemm_sweep/$(date +%Y%m%d)_m${BATCH}_s${SEQ}}"
MAX_PER_SHAPE="${MAX_PER_SHAPE:-16}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || VENV_IRON="$WS/xdna-engine/.venv-iron"
. "$REPO/scripts/amd_paths.sh"
IRON="${IRON:-$WS/wt-iron-causal}"
SWEEP="$REPO/designs/decode_fused/sweep_gemm_tiles.py"

# PLAN=1 is pure arithmetic over the tiling rules -- no IRON, no toolchain, no venv.
if [ "${PLAN:-0}" = "1" ]; then
  exec python3 "$SWEEP" --plan --batch "$BATCH" --seq "$SEQ" --out "$OUT" \
       --max-per-shape "$MAX_PER_SHAPE" "$@"
fi

[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: no iron venv at $VENV_IRON"; exit 1; }
iron_require_pin || exit 1
iron_at="$(iron_require_api "gen_gemm_tile_arm.py" \
  "iron/common/sequence.py:class OperatorSequence" \
  "iron/operators/gemm/op.py:class GEMM" \
  "iron/operators/gemm/op.py:b_col_maj" \
  "iron/operators/gemm/op.py:prio_accuracy" \
  "iron/operators/gemm/op.py:round_conv_even")" || exit 1
echo "[sweep] IRON on $iron_at (API surface verified)"

INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON${PYTHONPATH:+:$PYTHONPATH}"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
export AIE_DEVICE="${AIE_DEVICE:-npu2}"   # build off the device lock; see gen_llm_decode.py
[ -x "$AIECC_PATH" ] || { echo "ERROR: instance aiecc missing at $AIECC_PATH"; exit 1; }

# NVMe, never the tmpfs scratchpad. Each arm gets its own work dir under $OUT: IRON's artifact
# cache is filename+mtime keyed and GEMM.name omits the numerics flags, so a shared work dir lets
# one arm link another's design -- see the module docstring in sweep_gemm_tiles.py.
mkdir -p "$OUT"
exec "$VENV_IRON/bin/python" "$SWEEP" \
  --batch "$BATCH" --seq "$SEQ" --out "$OUT" --max-per-shape "$MAX_PER_SHAPE" \
  --python "$VENV_IRON/bin/python" "$@"
