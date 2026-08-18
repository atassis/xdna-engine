#!/usr/bin/env bash
# Device wrapper for gemm_command_accounting.py (task gemm-offcore-residue-occupancy, item 3).
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free, and ALWAYS
# restores them on exit including on abort.
#
# The free-device check is fuser + pgrep, never `systemctl is-active` (it prints inactive for
# stopped, absent AND running-as-a-plain-process alike). The pgrep pattern is ANCHORED to a path
# boundary and excludes this script's own process tree: an unanchored `pgrep -f 'npu serve'` matches
# any shell whose command line merely CONTAINS the string, which is a false FATAL that costs a
# service-stop window (hit 2026-08-18).
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/gemm_command_accounting${ACCT_TAG:+_$ACCT_TAG}.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

ARMS="${ARMS:-512x1024x1024_64x32x128_1c_modalid 512x1024x1024_64x32x128_4c_modalid 512x1024x1024_64x32x128_8c_modalid}"
SPAN="${SPAN:-512x1024x1024_64x32x128_4c_modalid=419340}"
REPS="${REPS:-10}"
# Named output: the default json name is the width series' banked artifact, so an arm set that is
# not that series must write elsewhere or it replaces a result other passes cite.
OUT="${OUT:-gemm_command_accounting.json}"

log "===== whole_array GEMM command-time accounting  $(date -Is) ====="
log "[svc] stopping xdna-engine + npu-vox"
systemctl --user stop xdna-engine.service npu-vox.service >/dev/null 2>&1
sleep 2
held=0
fuser /dev/accel/accel0 >/dev/null 2>&1 && held=1
# -a prints the cmdline so a survivor is identifiable; grep -v our own pid tree.
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
.venv-iron/bin/python scripts/gemm_command_accounting.py \
    --suffixes $ARMS --reps "$REPS" --out "$OUT" --span $SPAN 2>&1 | tee -a "$LOG"

log ""
log "log: $LOG"
