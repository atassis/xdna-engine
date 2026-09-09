#!/usr/bin/env bash
# Build the batched-prefill MLP block ELF (designs/decode_fused/gen_llm_prefill_mlp.py).
# Compile-only: AIE_DEVICE pins the target so the build never opens /dev/accel and never queues
# behind whatever holds the single-tenant NPU.
#
#   bash scripts/build_prefill_mlp.sh [BATCH] [OUT_DIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
BATCH="${1:-256}"
OUT="${2:-/mnt/data/xdna-scratch/prefill/mlp_m$BATCH}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || VENV_IRON="$WS/xdna-engine/.venv-iron"
. "$REPO/scripts/amd_paths.sh"
IRON="${IRON:-$IRON_DIR}"
WEIGHTS="${WEIGHTS:-$WS/artifacts-qwen3-0.6b/weights}"
GEN="$REPO/designs/decode_fused/gen_llm_prefill_mlp.py"

[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: no iron venv at $VENV_IRON"; exit 1; }
iron_require_pin || exit 1
iron_at="$(iron_require_api "gen_llm_prefill_mlp.py" \
  "iron/common/sequence.py:class OperatorSequence" \
  "iron/operators/gemm/op.py:class GEMM" \
  "iron/operators/gemm/op.py:b_col_maj")" || exit 1
echo "[build] IRON on $iron_at (API surface verified)"
[ -d "$WEIGHTS" ] || { echo "ERROR: no weights at $WEIGHTS"; exit 1; }

INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON${PYTHONPATH:+:$PYTHONPATH}"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
export AIE_DEVICE="${AIE_DEVICE:-npu2}"   # build off the device lock; see gen_llm_decode.py
[ -x "$AIECC_PATH" ] || { echo "ERROR: instance aiecc missing at $AIECC_PATH"; exit 1; }

# Build artifacts go to NVMe, never the tmpfs scratchpad.
WORK="${KEEP_WORK:-/mnt/data/xdna-scratch/prefill/build_m$BATCH}"
mkdir -p "$WORK" "$OUT"
cd "$WORK"
exec "$VENV_IRON/bin/python" "$GEN" --spec qwen3-0.6b --weights "$WEIGHTS" --out "$OUT" --batch "$BATCH"
