#!/usr/bin/env bash
# Create the py3.12 model-EXPORT venv used by the ONNX export/extract/convert scripts
# (extract_parakeet_encoder.py, export_gigaam_encoder.py, asr_oracle.py, convert_*.py, ...).
# This is SEPARATE from .venv-iron (py3.14, the AIE toolchain venv) on purpose: the export deps
# (torch/onnx/onnx-asr) must not pollute the toolchain env. Idempotent: safe to re-run.
# Does NOT run any export -- that is task 04.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV="${EXPORT_VENV:-.venv-export}"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"

# CPU-only torch (NPU-first policy: exports never use CUDA). The PyTorch CPU index carries the
# torch==2.12.0+cpu wheel; --extra-index-url keeps PyPI as the primary for the other deps.
uv pip install --python "$VENV" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r scripts/requirements-export.txt

echo "Export venv ready at $VENV."
echo "  activate:  source $VENV/bin/activate"
echo "  then (task 04) run exports, e.g.:  python scripts/extract_parakeet_encoder.py"
