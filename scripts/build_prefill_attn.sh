#!/usr/bin/env bash
# Build the batched-prefill attention block (designs/decode_fused/gen_llm_prefill_attn.py).
# Compile-only; AIE_DEVICE pins the target so the build never opens /dev/accel.
#   bash scripts/build_prefill_attn.sh [BATCH] [SEQ] [OUT_DIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$REPO/scripts/lib/data_root.sh"   # -> XDNA_SCRATCH
BATCH="${1:-256}"; SEQ="${2:-2048}"
OUT="${3:-$XDNA_SCRATCH/prefill/attn_m${BATCH}_s${SEQ}}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
. "$REPO/scripts/amd_paths.sh"
IRON="${IRON:-$IRON_DIR}"
iron_require_pin || exit 1
iron_at="$(iron_require_api "gen_llm_prefill_attn.py" \
  "iron/common/sequence.py:class OperatorSequence" \
  "iron/operators/gemm/op.py:b_col_maj" \
  "iron/operators/softmax/op.py:class Softmax")" || exit 1
echo "[build] IRON on $iron_at (API surface verified)"
INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON${PYTHONPATH:+:$PYTHONPATH}"
aiecc_resolve "$INST" || exit 1
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
export AIE_DEVICE="${AIE_DEVICE:-npu2}"
WORK="${KEEP_WORK:-$XDNA_SCRATCH/prefill/build_attn_m${BATCH}_s${SEQ}}"
mkdir -p "$WORK" "$OUT"; cd "$WORK"
exec "$VENV_IRON/bin/python" "$REPO/designs/decode_fused/gen_llm_prefill_attn.py" \
  --spec qwen3-0.6b --out "$OUT" --batch "$BATCH" --seq "$SEQ"
