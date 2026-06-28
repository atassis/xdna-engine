#!/usr/bin/env bash
# Phase-0 per-op occupancy run for the Parakeet encoder (brick #8, measure-first gate).
# RUN phase (needs the NPU). Single-tenant: stops npu-asr/voxd, ALWAYS restarts on
# exit, fuser-checks the device first (serialize -- [[npu-timing-check-fuser-first]]).
#
# Pipeline:
#   1. CPU goldens + roofline (parakeet_occupancy_golden.py)            [no NPU]
#   2. production resident xclbins (build_parakeet_kernels.sh, if absent) [no NPU]
#   3. DATA_MOVEMENT_ONLY stub xclbin (build_parakeet_occupancy_stub.sh)  [no NPU]
#   4. free NPU -> A/B occupancy harness -> restore NPU                   [NPU]
# Result: artifacts/parakeet/occupancy/occupancy_results.json (ranked table).
#
# Usage:
#   scripts/run_parakeet_occupancy.sh                 # fast tile 64x32x128 (default)
#   TILE=32x32x32 scripts/run_parakeet_occupancy.sh   # native bf16 (golden-gated)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
TILE="${TILE:-64x32x128}"
ITERS="${ITERS:-50}"
LOG="$REPO/artifacts/parakeet/occupancy/run.log"; mkdir -p "$(dirname "$LOG")"
log(){ echo "$@" | tee -a "$LOG"; }

restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1 || true; log "[svc] npu services restarted"; }

# --- CPU-side prep (no NPU) ---
log "[1/4] CPU goldens + roofline"
.venv-iron/bin/python scripts/parakeet_occupancy_golden.py 2>&1 | tee -a "$LOG"

log "[2/4] production resident xclbins (build if absent)"
WA=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build
if [[ ! -f "$WA/final_512x1024x4096_${TILE}_8c.xclbin" ]]; then
  scripts/build_parakeet_kernels.sh 2>&1 | tee -a "$LOG"
fi

log "[3/4] DATA_MOVEMENT_ONLY stub xclbin ($TILE)"
TILE="$TILE" scripts/build_parakeet_occupancy_stub.sh 2>&1 | tee -a "$LOG"

# --- NPU run (serialize) ---
log "[svc] stopping npu-asr / voxd"
systemctl --user stop npu-asr.service voxd.service >/dev/null 2>&1 || true; sleep 1
trap restart EXIT
if fuser /dev/accel/accel0 >/dev/null 2>&1; then
  log "FATAL: /dev/accel/accel0 busy -- another session holds the NPU. Aborting (serialize)."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"; exit 1
fi

log "[4/4] A/B occupancy harness (tile=$TILE iters=$ITERS)"
.venv-iron/bin/python scripts/parakeet_occupancy_harness.py --tile "$TILE" --iters "$ITERS" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "[done] rc=$rc -- results in artifacts/parakeet/occupancy/occupancy_results.json"
exit "$rc"
