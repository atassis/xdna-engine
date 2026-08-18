#!/usr/bin/env bash
# Device wrapper for gemm_dispatch_pipeline.py (task gemm-offcore-residue-occupancy, item 1).
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free, and ALWAYS
# restores them on exit including on abort.
#
# The free-device check is fuser + pgrep, never `systemctl is-active` (it prints inactive for
# stopped, absent AND running-as-a-plain-process alike). The pgrep pattern is ANCHORED to a path
# boundary and excludes this script's own process tree.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/gemm_dispatch_pipeline${PIPE_TAG:+_$PIPE_TAG}.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

# The pass-10 arms, so the gated numbers are comparable to the banked timing-only ones.
ARMS="${ARMS:-256x64x256_64x32x32_8c_modalid 512x128x1024_64x32x128_8c_modalid 512x1024x1024_64x32x128_8c_modalid}"
DEPTHS="${DEPTHS:-1 2 4 8 16 32 64}"
REPS="${REPS:-21}"
DEPTH_REPS="${DEPTH_REPS:-7}"
# Named output: never overwrite gemm_dispatch_pipeline{,_deep}.json, which passes 10 cites.
OUT="${OUT:-gemm_dispatch_pipeline_gated.json}"

log "===== whole_array GEMM dispatch pipeline, correctness-gated  $(date -Is) ====="
log "[svc] stopping xdna-engine + npu-vox"
systemctl --user stop xdna-engine.service npu-vox.service >/dev/null 2>&1
sleep 2
held=0
fuser /dev/accel/accel0 >/dev/null 2>&1 && held=1
if pgrep -af '(^|/)npu serve' | grep -qv "^$$ "; then
  pgrep -af '(^|/)npu serve' | grep -v "^$$ " >/dev/null && held=1
fi
if [ "$held" = 1 ]; then
  log "FATAL: device still held -- another session has the NPU. Aborting (single-tenant)."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"
  pgrep -af '(^|/)npu serve' | tee -a "$LOG"
  exit 75
fi
log "[svc] device clear"

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
# shellcheck disable=SC2086
.venv-iron/bin/python scripts/gemm_dispatch_pipeline.py \
    --suffixes $ARMS --depths $DEPTHS --reps "$REPS" --depth-reps "$DEPTH_REPS" \
    --out "$OUT" ${PIPE_EXTRA:-} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

log ""
log "log: $LOG   rc=$rc"
exit "$rc"
