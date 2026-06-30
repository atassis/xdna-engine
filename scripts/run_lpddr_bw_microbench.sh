#!/usr/bin/env bash
# RUN the LPDDR bandwidth microbench sweep on the NPU (needs a window). Single-tenant device
# discipline copied from run_parakeet_dma_occupancy.sh: stop the NPU services, fuser-check the
# device is free, timeout-guard, and ALWAYS restart services on exit (trap). Canonical units =
# npu-serve.service + voxd.service (NOT the stale npu-asr.service). CUDA disabled. Quiesce
# the box first (close anything using LPDDR -- the measurement is bandwidth-sensitive).
#
# Usage:
#   scripts/run_lpddr_bw_microbench.sh                      # rdwr cols=1 default sweep
#   MODE=read COLS=1 scripts/run_lpddr_bw_microbench.sh
#   MODE=rdwr COLS=8 scripts/run_lpddr_bw_microbench.sh     # aggregate (after building cols=8)
set -uo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$(git -C "$SRC" rev-parse --git-common-dir)/.." && pwd)"
export PARAKEET_TOOLROOT="$REPO"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=""
MODE="${MODE:-rdwr}"; COLS="${COLS:-1}"; ITERS="${ITERS:-50}"; TO="${TIMEOUT:-300}"
SWEEP_BYTES="${SWEEP_BYTES:-65536 262144 1048576 4194304 16777216 67108864}"
SWEEP_LINE="${SWEEP_LINE:-1024 4096 16384}"
SWEEP_DEPTH="${SWEEP_DEPTH:-2 4}"
LOG="$REPO/artifacts/parakeet/lpddr_bw/run.log"; mkdir -p "$(dirname "$LOG")"
log(){ echo "$@" | tee -a "$LOG"; }

restart(){ systemctl --user start npu-serve.service voxd.service >/dev/null 2>&1 || true; log "[svc] npu-serve + voxd restarted"; }

log "[svc] stopping npu-serve / voxd (quiesce for clean timing)"
systemctl --user stop npu-serve.service voxd.service >/dev/null 2>&1 || true
trap restart EXIT
sleep 1
for i in 1 2 3 4 5; do
  if ! fuser /dev/accel/accel0 >/dev/null 2>&1; then break; fi
  log "[wait] /dev/accel/accel0 still busy (try $i/5)"; sleep 1
done
if fuser /dev/accel/accel0 >/dev/null 2>&1; then
  log "FATAL: /dev/accel/accel0 still busy after stop -- another session holds the NPU. Aborting."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"; exit 1
fi
log "[ok] device free; running LPDDR-BW microbench (mode=$MODE cols=$COLS iters=$ITERS timeout ${TO}s)"

timeout "$TO" .venv-iron/bin/python "$SRC/scripts/lpddr_bw_microbench_harness.py" \
    --mode "$MODE" --cols "$COLS" --iters "$ITERS" --sweep-bytes $SWEEP_BYTES \
    --sweep-line $SWEEP_LINE --sweep-depth $SWEEP_DEPTH 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "[done] rc=$rc -- results in artifacts/parakeet/lpddr_bw/lpddr_bw_${MODE}_c${COLS}_results.json"
exit "$rc"
