#!/usr/bin/env bash
# Build a Whisper decoder on the LLM decode rail: dump the HF decoder weights (once), build the ELF
# with gen_llm_decode.py, print its sha256 for the scenario's recipe line. Compile-only.
#
#   bash scripts/build_whisper_rail.sh <spec> <OUT_DIR>      # spec: whisper-small | whisper-turbo
#
# MAX_SEQ (default 512) is the self-KV capacity: Whisper's 448 target positions, rounded up to the
# 256 granule check_seq requires.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC="${1:?usage: build_whisper_rail.sh <spec> <OUT_DIR>}"
OUT="${2:?usage: build_whisper_rail.sh <spec> <OUT_DIR>}"
WEIGHTS="${WEIGHTS:-$REPO/artifacts/$SPEC/weights}"
PY="${VENV_IRON:-$REPO/.venv-iron}/bin/python"
if [ ! -f "$WEIGHTS/model.decoder.embed_positions.weight.npy" ]; then
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "$PY" "$REPO/scripts/dump_llm_weights.py" --spec "$SPEC" --out "$WEIGHTS"
fi
WEIGHTS="$WEIGHTS" GEN_EXTRA="--max-seq ${MAX_SEQ:-512}" \
    bash "$REPO/scripts/build_llm_decode.sh" "$SPEC" "" "$OUT"
"$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["sha256"], sys.argv[2])' \
    "$OUT/meta.json" "$OUT/decode.elf"
