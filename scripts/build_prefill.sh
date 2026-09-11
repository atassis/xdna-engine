#!/usr/bin/env bash
# Build the WHOLE batched-prefill layer stack (designs/decode_fused/gen_llm_prefill.py).
# Compile-only; AIE_DEVICE pins the target so the build never opens /dev/accel.
#
#   bash scripts/build_prefill.sh [LAYERS] [BATCH] [SEQ] [OUT_DIR]
#
# IRON defaults to wt-iron-causal, NOT amd_paths.sh's wt-iron-integ: this build needs BOTH
# OperatorSequence(scratch_order=...) for the shared arena and Softmax(vector_size_source="rows")
# for the causal mask, and branch prefill/causal-softmax is where the two meet.
# Override with IRON=<dir>; the build fails loud if either parameter is absent.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
LAYERS="${1:-1}"; BATCH="${2:-256}"; SEQ="${3:-2048}"
OUT="${4:-/mnt/data/xdna/scratch/prefill/full_l${LAYERS}_m${BATCH}_s${SEQ}}"
CAUSAL="${CAUSAL:-rows}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || VENV_IRON="$WS/xdna-engine/.venv-iron"
. "$REPO/scripts/amd_paths.sh"
IRON="${IRON:-$WS/wt-iron-causal}"
# $REPO/artifacts/<spec>, not "$WS/artifacts-<spec>" -- the latter has a hyphen where a path
# separator belongs and is anchored at the workspace; it resolves to a directory that has
# never existed, so the build died on a missing weight rather than on a clear message.
WEIGHTS="${WEIGHTS:-$REPO/artifacts/qwen3-0.6b/weights}"
DECODE_META="${DECODE_META:-$WS/xdna-engine/artifacts/qwen3-0.6b/decode/meta.json}"

[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: no iron venv at $VENV_IRON"; exit 1; }
iron_require_pin || exit 1
iron_at="$(iron_require_api "gen_llm_prefill.py" \
  "iron/common/sequence.py:class OperatorSequence" \
  "iron/common/sequence.py:scratch_order" \
  "iron/operators/gemm/op.py:b_col_maj" \
  "iron/operators/rope/op.py:angle_rows" \
  "iron/operators/softmax/op.py:vector_size_source" \
  "iron/operators/strided_copy/op.py:output_offset_parameter")" || exit 1
echo "[build] IRON on $iron_at (API surface verified)"

ARENA_ARGS=()
if [ "${NO_ARENA_SHARE:-0}" = "1" ]; then
  ARENA_ARGS+=(--no-arena-share)
  echo "[build] WARNING: NO_ARENA_SHARE=1 -- the ELF will NOT share decode's scratch arena"
elif [ -f "$DECODE_META" ]; then
  ARENA_ARGS+=(--decode-meta "$DECODE_META")
else
  echo "ERROR: no decode artifact at $DECODE_META; set DECODE_META=<path> or NO_ARENA_SHARE=1"
  exit 1
fi
GOLDEN_ARGS=()
# LAYOUT_ONLY=1 stops after the buffer layout + shared-arena assert: seconds, no aiecc.
if [ "${LAYOUT_ONLY:-0}" = "1" ]; then
  ARENA_ARGS+=(--layout-only)
  NO_GOLDEN=1
fi
if [ "${NO_GOLDEN:-0}" = "1" ]; then
  GOLDEN_ARGS+=(--no-golden)
  # Still pass the weights when they are there: --no-golden skips the CPU reference, but the
  # generator also uses this directory to verify that the decode arena holds what the graph
  # assumes (check_shared_weights). Skipping that on the quick build path is how a silently
  # reordered weight reaches the device.
  [ -d "$WEIGHTS" ] && GOLDEN_ARGS+=(--weights "$WEIGHTS")
else
  [ -d "$WEIGHTS" ] || { echo "ERROR: no weights at $WEIGHTS (or set NO_GOLDEN=1)"; exit 1; }
  GOLDEN_ARGS+=(--weights "$WEIGHTS")
fi

INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON${PYTHONPATH:+:$PYTHONPATH}"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
export AIE_DEVICE="${AIE_DEVICE:-npu2}"   # build off the device lock; see gen_llm_decode.py
[ -x "$AIECC_PATH" ] || { echo "ERROR: instance aiecc missing at $AIECC_PATH"; exit 1; }

# Build artifacts go to NVMe, never the tmpfs scratchpad. Per-arm work dir, because IRON keys
# cached operator artifacts by NAME and a shared dir lets one arm link another's binaries.
WORK="${KEEP_WORK:-/mnt/data/xdna/scratch/prefill/build_full_l${LAYERS}_m${BATCH}_s${SEQ}_${CAUSAL}}"
mkdir -p "$WORK" "$OUT"
cd "$WORK"
exec "$VENV_IRON/bin/python" "$REPO/designs/decode_fused/gen_llm_prefill.py" \
  --spec qwen3-0.6b --out "$OUT" --layers "$LAYERS" --batch "$BATCH" --seq "$SEQ" \
  --causal "$CAUSAL" "${ARENA_ARGS[@]}" "${GOLDEN_ARGS[@]}"
