#!/usr/bin/env bash
# Device wrapper for gemm_drain_width_isolated.py (task mode-switched-multi-program-xclbin).
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free, and ALWAYS
# restores them on exit including on abort. The free-device check is fuser + pgrep, never
# `systemctl is-active` -- it prints inactive for stopped, absent AND running-as-a-plain-process
# alike, and `npu-asr` was renamed to `xdna-engine`, so the old name returns a false all-clear.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/gemm_drain_width_isolated.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

OUT="${OUT:-artifacts/gemm_drain_width_isolated.json}"

log "===== K=4096 a_panel GEMM drain-width isolation  $(date -Is) ====="
log "[svc] stopping xdna-engine + npu-vox"
systemctl --user stop xdna-engine.service npu-vox.service >/dev/null 2>&1
sleep 2
held=0
fuser /dev/accel/accel0 >/dev/null 2>&1 && held=1
if pgrep -af '(^|/)npu serve' >/dev/null 2>&1; then held=1; fi
if [ "$held" = 1 ]; then
  log "FATAL: device still held -- another session has the NPU. Aborting (single-tenant)."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"
  pgrep -af '(^|/)npu serve' | tee -a "$LOG"
  exit 75
fi
log "[svc] device clear"

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
.venv-iron/bin/python scripts/gemm_drain_width_isolated.py --out "$OUT" \
    ${DRAIN_ARMS:+--arms $DRAIN_ARMS} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

log ""
log "log: $LOG   rc=$rc"
exit "$rc"
