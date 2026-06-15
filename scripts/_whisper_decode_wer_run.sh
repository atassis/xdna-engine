#!/usr/bin/env bash
# Single-tenant Whisper-small WER head-to-head: ONNX decode vs NPU decode, over wer_clips.
# Stops the NPU services, runs BOTH evals (each launches engine_serve, POSTs the clips, scores
# WER vs refs.json), then ALWAYS restarts the services (trap, even on failure/Ctrl-C).
set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PORT=11434
URL="http://127.0.0.1:${PORT}/v1/audio/transcriptions"
SERVE="rust/target/release/engine_serve"
SCEN="scenarios/asr-whisper-small.toml"
LOGDIR=/tmp/whisper_decode_wer
mkdir -p "$LOGDIR"

restart_services() { echo "[run] restarting npu services"; systemctl --user start npu-asr.service voxd.service; }
trap restart_services EXIT

echo "[run] stopping npu services (single-tenant device)"
systemctl --user stop npu-asr.service voxd.service
sleep 2
if fuser /dev/accel/accel0 2>/dev/null; then echo "[run] ERROR: device still busy"; exit 1; fi
echo "[run] device clear"

run_one() {  # $1=label  $2=extra-env (NPU_DECODE=1 or empty)
  local label="$1" env_kv="$2"
  echo "=================== $label ==================="
  local serve_log="$LOGDIR/serve_${label}.log"
  # shellcheck disable=SC2086
  env LD_LIBRARY_PATH=~/.local/lib/npu-asr $env_kv "$SERVE" "$SCEN" "$PORT" >"$serve_log" 2>&1 &
  local pid=$!
  # readiness poll
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "[run] $label serve died early:"; tail -20 "$serve_log"; return 1; fi
    sleep 1
  done
  echo "[run] $label serve up (pid $pid)"
  python3 scripts/whisper_decode_wer.py --url "$URL" --label "$label" --out "$LOGDIR/wer_${label}.json"
  local rc=$?
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 2
  return $rc
}

run_one onnx ""
run_one npu  "NPU_DECODE=1"

echo "[run] done; logs in $LOGDIR"
