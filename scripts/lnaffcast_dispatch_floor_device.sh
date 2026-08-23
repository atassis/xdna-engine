#!/usr/bin/env bash
# Device wrapper for the dispatch-floor probe (task mode-switched-multi-program-xclbin, `next:` 2).
#
# ONE QUIESCED SESSION, ONE XCLBIN, TWO QUESTIONS. lnaffcast_rows is insts-only and the GEMM stream
# is a mode of the same array program, so every arm runs on the ONE loaded xclbin -- no program
# transition inside any comparison. That lets both open questions share a session:
#   * the ROW SWEEP (128/256/384/512, the complete legal set -- the generator refuses anything that
#     is not a multiple of the 128-row round) re-fits the 151.0 us floor over four points instead
#     of the recorded two;
#   * the DEPTH SWEEP splits each arm's floor into host round-trip and device-serial work, and the
#     GEMM stream rides along as the arm whose command size differs most.
# Same-session matters: this box drifts 8.0% on one artifact ACROSS sessions against 2.2% within
# one, and that drift is persistent rather than thermal, so absolute numbers from different
# sessions cannot be subtracted.
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free with
# fuser + pgrep (never `systemctl is-active` -- it prints inactive for stopped, absent AND
# running-as-a-plain-process alike), and ALWAYS restores them on exit including on abort.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/lnaffcast_dispatch_floor.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

log "===== lnaffcast dispatch floor: host round-trip or its own cost?  $(date -Is) ====="
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
.venv-iron/bin/python scripts/lnaffcast_dispatch_floor.py "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "\nlog: $LOG   rc=$rc"
exit $rc
