#!/usr/bin/env bash
# Recreate the whisper deep-C DECODE build environment from a clean workspace.
#
# The workspace migration provisioned the Parakeet serving path but NOT the whisper
# deep-C decode env, so from a fresh checkout `scripts/build_deepc_decode.sh` cannot
# run (missing torch in the toolchain venv + missing decoder weights). Per the
# reproducibility policy, that recreation must be a checked-in script, not manual
# steps -- this is that script. Captures the 4 steps that provisioned it by hand.
#
# CPU-only, no NPU needed. Idempotent: every step is guarded, so re-runs are safe
# (already-satisfied steps become fast no-ops). Set FORCE=1 to redo the export+extract.
#
#   bash scripts/setup_decode_env.sh
#
# Prerequisites (created elsewhere, NOT by this script):
#   .venv-iron    py3.14 AIE toolchain venv -- built by scripts/toolchain_up.sh /
#                 scripts/fast_build_env.sh. This script only ADDS torch onto it.
#   openai/whisper-small in the HF hub cache -- prefetched by scripts/fetch_models.sh
#                 (step 3 will otherwise download it; network is fine).
#
# After this env is in place the decode build still hits the newstack device-API port
# (a separate toolchain wall) -- this script only removes the ENV gap, not that wall.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"          # py3.14 toolchain venv (gen_decode + iron import torch)
VENV_EXPORT="${EXPORT_VENV:-.venv-export}"          # py3.12 export venv (optimum onnx exporter)
ONNX_DIR="$REPO/artifacts/whisper-small/onnx"
WEIGHTS_DIR="$REPO/artifacts/whisper-small/whisper_decoder"

# --- Step 1: torch (CPU) into the toolchain venv -----------------------------------
# gen_decode.py and iron.common.sequence both `import torch`, but the toolchain venv
# ships without it. CPU-only by policy (NPU-first: the build never touches CUDA). The
# py3.14 wheel that resolves off the PyTorch CPU index is torch 2.13.0+cpu.
echo "== step 1/4: torch (cpu) into $VENV_IRON =="
if [ ! -x "$VENV_IRON/bin/python" ]; then
  echo "ERROR: $VENV_IRON/bin/python missing. Build the toolchain venv first" >&2
  echo "       (scripts/toolchain_up.sh / scripts/fast_build_env.sh), then re-run." >&2
  exit 1
fi
if "$VENV_IRON/bin/python" -c "import torch" 2>/dev/null; then
  echo "[skip] torch already importable in $VENV_IRON"
else
  uv pip install --python "$VENV_IRON" torch --index-url https://download.pytorch.org/whl/cpu
fi

# --- Step 2: optimum-onnx into the export venv -------------------------------------
# optimum 2.x SPLIT the ONNX exporter into a separate `optimum-onnx` package; without
# it `optimum-cli export onnx` is an unrecognized command. Create the export venv from
# its canonical recipe if it is not there yet, then add optimum-onnx onto it.
echo "== step 2/4: optimum-onnx into $VENV_EXPORT =="
if [ ! -x "$VENV_EXPORT/bin/python" ]; then
  echo "[setup] $VENV_EXPORT missing -- creating via scripts/setup_export_venv.sh"
  bash "$REPO/scripts/setup_export_venv.sh"
fi
if uv pip show --python "$VENV_EXPORT" optimum-onnx >/dev/null 2>&1; then
  echo "[skip] optimum-onnx already installed in $VENV_EXPORT"
else
  uv pip install --python "$VENV_EXPORT" optimum-onnx
fi

# --- Step 3: export whisper-small decoder to ONNX ----------------------------------
# CUDA_VISIBLE_DEVICES="" keeps the export on CPU. Produces decoder_model.onnx (+
# decoder_with_past_model.onnx). Needs openai/whisper-small in the HF cache.
echo "== step 3/4: optimum-cli export onnx -> $ONNX_DIR =="
if [ "${FORCE:-0}" != "1" ] && [ -f "$ONNX_DIR/decoder_model.onnx" ]; then
  echo "[skip] $ONNX_DIR/decoder_model.onnx already present (FORCE=1 to re-export)"
else
  CUDA_VISIBLE_DEVICES="" "$VENV_EXPORT/bin/optimum-cli" export onnx \
    --model openai/whisper-small "$ONNX_DIR"
fi

# --- Step 4: extract decoder weights to .npy ---------------------------------------
# extract_whisper_decoder.py reads WHISPER_DECODER_ONNX and writes
# <onnx>/../whisper_decoder/*.npy (it self-prints an npy-vs-onnx equality sanity check).
echo "== step 4/4: extract decoder weights -> $WEIGHTS_DIR =="
if [ "${FORCE:-0}" != "1" ] && [ -f "$WEIGHTS_DIR/proj_out.weight.npy" ]; then
  echo "[skip] $WEIGHTS_DIR already populated (FORCE=1 to re-extract)"
else
  WHISPER_DECODER_ONNX="$ONNX_DIR/decoder_model.onnx" \
    "$VENV_EXPORT/bin/python" "$REPO/scripts/extract_whisper_decoder.py"
fi

echo "Done. Decode env ready:"
echo "  torch in     $VENV_IRON"
echo "  weights in   $WEIGHTS_DIR"
echo "  next:        bash scripts/build_deepc_decode.sh"
