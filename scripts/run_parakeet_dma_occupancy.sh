#!/usr/bin/env bash
# RUN the DMA-occupancy sweep on the NPU (needs a window). Single-tenant device discipline:
# stop the NPU services, fuser-check the device is free, timeout-guard the run, and ALWAYS
# restart the services on exit (trap). Canonical units = npu-serve.service + voxd.service
# (NOT the stale npu-asr.service the older occupancy runner names). CUDA disabled.
#
# Usage:
#   scripts/run_parakeet_dma_occupancy.sh                       # 3 existing N points (no builds)
#   SWEEP_N="1024 2048 3072 4096" scripts/run_parakeet_dma_occupancy.sh   # after building N=3072
set -uo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"           # where this script + harness live (may be a worktree)
# Toolchain root = the MAIN checkout (holds .venv-iron, mlir-aie/build, artifacts/). In a worktree this
# is the git common dir's parent; in MAIN it is SRC itself.
REPO="$(cd "$(git -C "$SRC" rev-parse --git-common-dir)/.." && pwd)"
export PARAKEET_TOOLROOT="$REPO"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=""
SWEEP_N="${SWEEP_N:-1024 2048 4096}"
ITERS="${ITERS:-50}"
TO="${TIMEOUT:-300}"
LOG="$REPO/artifacts/parakeet/occupancy/dma_run.log"; mkdir -p "$(dirname "$LOG")"
log(){ echo "$@" | tee -a "$LOG"; }

restart(){ systemctl --user start npu-serve.service voxd.service >/dev/null 2>&1 || true; log "[svc] npu-serve + voxd restarted"; }

log "[svc] stopping npu-serve / voxd (quiesce for clean timing)"
systemctl --user stop npu-serve.service voxd.service >/dev/null 2>&1 || true
trap restart EXIT
sleep 1
# fuser-check the device is actually free before dispatching (serialize -- npu-timing-check-fuser-first)
for i in 1 2 3 4 5; do
  if ! fuser /dev/accel/accel0 >/dev/null 2>&1; then break; fi
  log "[wait] /dev/accel/accel0 still busy (try $i/5)"; sleep 1
done
if fuser /dev/accel/accel0 >/dev/null 2>&1; then
  log "FATAL: /dev/accel/accel0 still busy after stop -- another session holds the NPU. Aborting."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"; exit 1
fi
log "[ok] device free; running DMA-occupancy sweep (N=$SWEEP_N iters=$ITERS, timeout ${TO}s)"

timeout "$TO" .venv-iron/bin/python "$SRC/scripts/parakeet_dma_occupancy_harness.py" \
    --sweep-N $SWEEP_N --iters "$ITERS" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "[done] rc=$rc -- results in artifacts/parakeet/occupancy/dma_sweep_results.json"
exit "$rc"
