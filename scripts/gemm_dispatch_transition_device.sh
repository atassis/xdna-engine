#!/usr/bin/env bash
# Device wrapper for gemm_dispatch_transition.py (task gemm-offcore-residue-occupancy, item 1b).
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free, and ALWAYS
# restores them on exit including on abort. Same shape as gemm_dispatch_chain_device.sh -- the
# free-device check is fuser + pgrep, never `systemctl is-active` (it prints inactive for stopped,
# absent AND running-as-a-plain-process alike).
#
# The two arms are distinct xclbins of the SAME 512x1024x1024 shape on the SAME 4 columns, so a
# command's work and the partition are held fixed and only the program differs. Different tilings are
# fine and deliberate: the counterfactual is (a+b)/2 at the same depth, so the arms are not required
# to cost the same.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/gemm_dispatch_transition${TRANS_TAG:+_$TRANS_TAG}.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

# MODE picks WHICH boundary the alternating arm crosses -- see the probe header. two-context is the
# banked pairing (program and context both change); same-context and dup-context split that floor
# into its two terms, and same-context is the mode-switch mechanism itself, so its arm B must be a
# stream of arm A's OWN design (the modalid/modalsilu pair differs only in the rtp[0] epilogue word).
MODE="${MODE:-two-context}"
ARM_A="${ARM_A:-512x1024x1024_64x32x128_4c_modalidnt}"
if [ "$MODE" = "same-context" ]; then
  ARM_B="${ARM_B:-512x1024x1024_64x32x128_4c_modalsilunt}"
elif [ "$MODE" = "dup-context" ]; then
  ARM_B="${ARM_B:-$ARM_A}"
else
  ARM_B="${ARM_B:-512x1024x1024_64x64x64_4c_modalidnt}"
fi
DEPTHS="${DEPTHS:-1 2 4 8 16 32 64}"
DEPTH_REPS="${DEPTH_REPS:-15}"
OUT="${OUT:-gemm_dispatch_transition.json}"

log "===== whole_array GEMM program-transition dispatch probe [$MODE]  $(date -Is) ====="
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
# shellcheck disable=SC2086
.venv-iron/bin/python scripts/gemm_dispatch_transition.py \
    --arm-a "$ARM_A" --arm-b "$ARM_B" --mode "$MODE" --depths $DEPTHS --depth-reps "$DEPTH_REPS" \
    --out "$OUT" ${TRANS_EXTRA:-} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

log ""
log "log: $LOG   rc=$rc"
exit "$rc"
