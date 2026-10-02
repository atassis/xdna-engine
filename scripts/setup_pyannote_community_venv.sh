#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV="${PYANNOTE_COMMUNITY_VENV:-.venv-pyannote-community}"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r scripts/requirements-pyannote-community.txt

echo "pyannote community export venv ready at $VENV."
echo "  export: $VENV/bin/python scripts/export_pyannote.py"
