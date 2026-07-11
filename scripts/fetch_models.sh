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

# Assemble the flat, serve-ready Parakeet artifact dir the engine loads
# (rust/npu-engine/src/asr/parakeet.rs: artifacts/parakeet/{preprocessor.onnx,decoder_joint.onnx,
# vocab.txt,encoder/}). No HF repo ships this exact layout, so build it here from pinned sources.
# Idempotent. NOTE: the encoder/ arena is produced separately by extract_parakeet_encoder.py and is
# NOT assembled here. Two migration bugs this fixes:
#   - preprocessor.onnx must be onnx_asr's nemo128.onnx (128-mel; Parakeet FastConformer needs 128
#     mels). Do NOT use gigaam_v3.onnx here -- that is the 64-mel preproc, correct only for GigaAM.
#   - the engine loads decoder_joint.onnx, but HF ships it as decoder_joint-model.onnx -> rename.
prep_parakeet_artifacts() {
  local snap dst pyexe data
  snap=$(ls -d "$HOME"/.cache/huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v3-onnx/snapshots/*/ 2>/dev/null | head -n1)
  if [ -z "$snap" ]; then echo "[parakeet] HF snapshot not found -- ASR fetch must run first"; return 1; fi
  dst="$REPO/artifacts/parakeet"; mkdir -p "$dst"

  # locate onnx_asr's bundled 128-mel preprocessor via the export venv python
  pyexe="${EXPORT_PY:-$REPO/${EXPORT_VENV:-.venv-export}/bin/python}"
  data=$("$pyexe" -c "import onnx_asr,os;print(os.path.join(os.path.dirname(onnx_asr.__file__),'preprocessors','data'))")
  cp -f "$data/nemo128.onnx"              "$dst/preprocessor.onnx"   # 128-mel (NOT gigaam_v3 64-mel)
  cp -f "$snap/decoder_joint-model.onnx"  "$dst/decoder_joint.onnx"  # HF name -> engine name
  cp -f "$snap/vocab.txt"                 "$dst/vocab.txt"
  echo "[parakeet] assembled $dst : preprocessor.onnx(nemo128 128-mel) + decoder_joint.onnx + vocab.txt"
}

echo "== ASR pipeline models =="
for r in "${ASR_REPOS[@]}"; do fetch "$r"; done

echo "== assemble parakeet serving artifacts =="
prep_parakeet_artifacts

if [ "${FETCH_EXTRA:-1}" = "1" ]; then
  echo "== extended model set (set FETCH_EXTRA=0 to skip) =="
  for r in "${EXTRA_REPOS[@]}"; do fetch "$r"; done
fi

echo "Done. Models are in ~/.cache/huggingface/hub; downstream scripts glob there."
