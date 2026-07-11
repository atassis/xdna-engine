#!/usr/bin/env bash
# Download the model inputs the export + ASR pipeline consume, into the HF hub cache
# (~/.cache/huggingface/hub) where every downstream script globs models--*/snapshots/*.
# Idempotent: an already-cached repo is a fast no-op. Everything here is prefetched
# (overnight/PREFETCH-STATE.md), so set HF_HUB_OFFLINE=1 to force a cache-only, network-free run.
# Does NOT export/convert anything -- that is task 04.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

# huggingface-cli ships with huggingface_hub (in the export venv). Allow an override, and fall back
# to the export venv's copy if it is not on PATH.
HF="${HF_CLI:-huggingface-cli}"
command -v "$HF" >/dev/null 2>&1 || HF="$REPO/${EXPORT_VENV:-.venv-export}/bin/huggingface-cli"

# --- ASR pipeline: the load-bearing inputs for the Parakeet/GigaAM NPU engine ---
#   parakeet: encoder-model.onnx(.data), decoder_joint-model.onnx, vocab.txt, encoder-model.int8.onnx
#   gigaam:   v3_rnnt_encoder/decoder/joint.onnx, v3_vocab.txt
# NOTE: the mel preprocessor.onnx is NOT in these repos -- it ships INSIDE the `onnx-asr` pip package
# (onnx_asr/preprocessors/data/gigaam_v3.onnx) and asr_oracle.py copies it from there. Installing
# onnx-asr (setup_export_venv.sh) is what provides the preprocessor; there is no HF download for it.
ASR_REPOS=(
  "istupakov/parakeet-tdt-0.6b-v3-onnx"
  "istupakov/gigaam-v3-onnx"
)

# --- Extended set for the other-arch export/parity tasks (16 esm2, 18 vit/opt/whisper, etc.) ---
EXTRA_REPOS=(
  "facebook/esm2_t6_8M_UR50D"
  "facebook/esm2_t12_35M_UR50D"
  "google/vit-base-patch16-224"
  "facebook/opt-125m"
  "openai/whisper-small"
  "BAAI/bge-base-en-v1.5"
  "openai/clip-vit-base-patch32"
  "facebook/dinov2-base"
  "microsoft/resnet-18"
  "sentence-transformers/all-MiniLM-L6-v2"
  "answerdotai/ModernBERT-base"
)

fetch() { echo "[fetch] $1"; "$HF" download "$1" >/dev/null; }

echo "== ASR pipeline models =="
for r in "${ASR_REPOS[@]}"; do fetch "$r"; done

if [ "${FETCH_EXTRA:-1}" = "1" ]; then
  echo "== extended model set (set FETCH_EXTRA=0 to skip) =="
  for r in "${EXTRA_REPOS[@]}"; do fetch "$r"; done
fi

echo "Done. Models are in ~/.cache/huggingface/hub; downstream scripts glob there."
