#!/usr/bin/env bash
# Device wrapper for gemm_dispatch_chain.py (task gemm-offcore-residue-occupancy, item 1).
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free, and ALWAYS
# restores them on exit including on abort. Same shape as gemm_dispatch_pipeline_device.sh -- the
# free-device check is fuser + pgrep, never `systemctl is-active` (it prints inactive for stopped,
# absent AND running-as-a-plain-process alike).
#
# The arm is the bf16-OUT build, which the chain needs and the shipped f32-out design cannot give:
# C must be re-readable as the next command's A. It is also n=64, not the deployed n=128 -- bf16 out
# carries a separate f32 accumulator AND a bf16 drain buffer, which is 72 KB of L1 against 64 KB at
# n=128 ('Basic sequential allocation failed'). So the ABSOLUTE us/cmd here is NOT comparable to the
# banked 382.3 -> 240.1; the indep arm is measured on this same artifact precisely so the chain has a
# same-build denominator.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/gemm_dispatch_chain${CHAIN_TAG:+_$CHAIN_TAG}.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

ARMS="${ARMS:-512x1024x1024_64x32x64_8c_modalidbf16out}"
DEPTHS="${DEPTHS:-1 2 4 8 16 32 64}"
REPS="${REPS:-21}"
# The depth-1 denominator is the unstable end -- the same arm returned 358.6/360.7/381.7/450.1/454.8
# us in one session -- so the shallow end needs reps, not 7. See the pass-11 method note.
DEPTH_REPS="${DEPTH_REPS:-25}"
OUT="${OUT:-gemm_dispatch_chain.json}"

log "===== whole_array GEMM dependent-chain dispatch probe  $(date -Is) ====="
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
.venv-iron/bin/python scripts/gemm_dispatch_chain.py \
    --suffixes $ARMS --depths $DEPTHS --reps "$REPS" --depth-reps "$DEPTH_REPS" \
    --out "$OUT" ${CHAIN_EXTRA:-} 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

log ""
log "log: $LOG   rc=$rc"
exit "$rc"
