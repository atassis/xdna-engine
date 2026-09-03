#!/usr/bin/env bash
# Build a decoder-LLM fused decode ELF from an LlmSpec (route_b_kernels/decode_fused/gen_llm_decode.py).
# Compile-only, no NPU needed.
#
#   bash scripts/build_llm_decode.sh <spec> [LAYERS] [OUT_DIR]
#     spec    qwen3-0.6b | gemma3-270m
#     LAYERS  default: the spec's full depth
#
# Env overrides: VENV_IRON, IRON (an IRON checkout carrying iron/common/fusion.py), WEIGHTS.
# NOTE the shared workspace IRON checkout is NOT usable by default: its local integration-stack has
# dropped the carried iron/common/fusion.py (upstream deleted it; we carry it). Point IRON at a
# worktree of origin/integration-stack instead of editing the shared checkout.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
SPEC="${1:?usage: build_llm_decode.sh <spec> [LAYERS] [OUT]}"
LAYERS="${2:-}"
OUT="${3:-$REPO/artifacts/$SPEC/decode${LAYERS:+_l$LAYERS}}"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
[ -x "$VENV_IRON/bin/python" ] || VENV_IRON="$WS/xdna-engine/.venv-iron"
IRON="${IRON:-$WS/wt-iron-qwen3}"
WEIGHTS="${WEIGHTS:-$REPO/artifacts/$SPEC/weights}"
GEN="$REPO/route_b_kernels/decode_fused/gen_llm_decode.py"

[ -x "$VENV_IRON/bin/python" ] || { echo "ERROR: no iron venv at $VENV_IRON"; exit 1; }
[ -f "$IRON/iron/common/fusion.py" ] || {
  echo "ERROR: $IRON lacks iron/common/fusion.py (FusedMLIROperator). Use a worktree of"
  echo "       origin/integration-stack -- the shared checkout's local branch dropped the carry."; exit 1; }
[ -d "$WEIGHTS" ] || { echo "ERROR: no weights at $WEIGHTS (run scripts/dump_llm_weights.py)"; exit 1; }

INST="$("$REPO/scripts/toolchain_up.sh")"
export PYTHONPATH="$INST/python:$IRON${PYTHONPATH:+:$PYTHONPATH}"
export AIECC_PATH="${AIECC_PATH:-$INST/bin/aiecc}"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$VENV_IRON/lib/python3.14/site-packages/llvm-aie}"
export MLIR_AIE_INSTANCE="$INST"
. "$REPO/scripts/amd_paths.sh"
export PATH="$VENV_IRON/bin:$VENV_IRON/cc-shim:$AIEBU_ASM_DIR:$PATH"
[ -x "$AIECC_PATH" ] || { echo "ERROR: instance aiecc missing at $AIECC_PATH"; exit 1; }

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT   # IRON writes build/ intermediates under CWD
mkdir -p "$OUT"
echo "[build] spec=$SPEC layers=${LAYERS:-full} inst=$(basename "$INST") iron=$(basename "$IRON")"
( cd "$WORK" && "$VENV_IRON/bin/python" "$GEN" --spec "$SPEC" --weights "$WEIGHTS" \
    --out "$OUT" ${LAYERS:+--layers $LAYERS} ${GEN_EXTRA:-} )
