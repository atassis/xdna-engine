#!/usr/bin/env bash
# Create the venv that generates Gemma-4 host references (llm_hf_bf16_ref.py).
#
# SEPARATE from .venv-export, and not by preference: Gemma-4 needs transformers 5.x -- 4.57.6
# knows gemma through gemma3n and has no gemma4 at all -- while .venv-export is held at 4.57.6 by
# optimum, whose ONNX exporter imports transformers.modeling_utils.get_parameter_dtype. That
# symbol is gone in 5.x and optimum 2.3.0 (latest, 2026-09-14) still imports it, so bumping the
# export venv breaks every ONNX path. Re-check on an optimum release that supports transformers 5;
# until then the two cannot share a venv.
#
#   bash scripts/setup_gemma4_oracle_venv.sh
#   $GEMMA4_ORACLE_VENV/bin/python scripts/llm_hf_bf16_ref.py --model <checkpoint> --layers 6 ...
set -euo pipefail
VENV="${GEMMA4_ORACLE_VENV:-/mnt/data/xdna/venvs/gemma4-oracle}"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
# torch CPU-only by the same policy as the export venv; the +cpu wheel needs the PyTorch index.
uv pip install --python "$VENV" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  transformers==5.17.0 "torch==2.12.0+cpu" numpy==2.4.6 accelerate safetensors
echo "Gemma-4 oracle venv ready at $VENV."
