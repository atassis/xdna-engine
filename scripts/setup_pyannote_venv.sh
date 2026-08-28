#!/usr/bin/env bash
# Create the py3.12 venv for scripts/export_pyannote.py.
#
# SEPARATE from .venv-export on purpose, the same way .venv-export is separate from .venv-iron:
# pyannote.audio 3.3.2 does not work against the torch/torchaudio/huggingface_hub versions
# .venv-export pins, and each of those three fails differently and non-obviously:
#   torchaudio >= 2.9  -> pyannote's io.py uses the removed torchaudio.AudioMetaData
#   torchaudio >= 2.2  -> pyannote.audio 3.1.x uses the removed torchaudio.set_audio_backend
#   torch      >= 2.6  -> torch.load defaults to weights_only=True and rejects the checkpoint
#   huggingface_hub >= 0.28 -> hf_hub_download no longer accepts use_auth_token=
# The versions below are the ones actually verified to load the model on this box.
# Idempotent: safe to re-run. Does NOT run the export.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV="${PYANNOTE_VENV:-.venv-pyannote}"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"

uv pip install --python "$VENV" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r scripts/requirements-pyannote.txt

echo "pyannote export venv ready at $VENV."
echo "  export:  HF_TOKEN=hf_... $VENV/bin/python scripts/export_pyannote.py"
