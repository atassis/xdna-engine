#!/usr/bin/env bash
# Alternated Whisper decode A/B on device. Each round runs every arm once, as one `npu-dev
# whisper-e2e` process over every WER clip (warmup + 3 timed transcriptions per clip); odd rounds
# run the arms in reverse order. Score the logs with scripts/whisper_decode_ab_report.py.
#
#   scripts/whisper_decode_ab.sh <OUT_DIR> <ROUNDS> <name=scenario.toml[:VAR=VALUE...]>...
#
# The NPU is single-tenant: hold the device alone (stop xdna-engine.service) while it runs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: whisper_decode_ab.sh <OUT_DIR> <ROUNDS> <arm>...}"
ROUNDS="${2:?}"
shift 2
ARMS=("$@")
NPU_DEV_BIN="${NPU_DEV_BIN:-$(cd "$REPO/rust" && cargo metadata --format-version 1 --no-deps \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/release/npu-dev}"
mkdir -p "$OUT"
mapfile -t CLIPS < <(python3 -c 'import json,sys; print("\n".join(sorted(json.load(open(sys.argv[1])))))' \
    "$REPO/artifacts/wer_clips/refs.json")
{ xrt-smi examine 2>/dev/null | grep -i "power mode" || echo "power mode: unreadable"
  powerprofilesctl get 2>/dev/null || true; } > "$OUT/conditions.txt"
cd "$REPO"
for ((r = 0; r < ROUNDS; r++)); do
    order=("${ARMS[@]}")
    if ((r % 2)); then mapfile -t order < <(printf '%s\n' "${ARMS[@]}" | tac); fi
    for arm in "${order[@]}"; do
        name="${arm%%=*}"
        IFS=: read -r -a parts <<< "${arm#*=}"
        echo "[ab] round $r arm $name" >&2
        env WHISPER_SCENARIO="${parts[0]}" WHISPER_TIMING=1 NPU_DEBUG_TOKEN_IDS=1 "${parts[@]:1}" \
            "$NPU_DEV_BIN" whisper-e2e "${CLIPS[@]/#/artifacts/wer_clips/}" > "$OUT/r${r}_$name.log" 2>&1
    done
done
