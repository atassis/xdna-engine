#!/usr/bin/env bash
# Device wrapper for the gather-vs-tap arm (task mode-switched-multi-program-xclbin).
#
# TWO GROUPS IN ONE QUIESCED SESSION. The scatter body is a different array program, so the
# candidate cannot share the banked baseline's xclbin. Each group therefore loads its own and the
# comparison between them is cross-xclbin -- which is why each group also runs the GEMM stream on
# its own xclbin as a CONTROL. The GEMM stream is untouched by lnaffcast_scatter_c (its insts are
# 10704 B on both, the shipped resident's), so its two medians agreeing is what says the vehicles
# are comparable and the mode delta is the body rather than the session.
#
#   group A, baseline xclbin: <base>          derived C tap, contiguous write   -> must PASS
#                             <base>ctgc      contiguous tap, contiguous write  -> must FAIL
#                             <base:gemm>     GEMM stream                       -> control
#   group B, scatter xclbin:  <scat>          derived C tap, SCATTERED write    -> must FAIL
#                             <scat>ctgc      contiguous tap, SCATTERED write   -> must PASS
#                             <scat:gemm>     GEMM stream                       -> control
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free, and ALWAYS
# restores them on exit including on abort. The free-device check is fuser + pgrep, never
# `systemctl is-active` -- it prints inactive for stopped, absent AND running-as-a-plain-process
# alike, and `npu-asr` was renamed to `xdna-engine`, so the old name returns a false all-clear.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/lnaffcast_scatter.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

BASE=512x1024x1024_32x32x128_8c_modalidbf16outkrtpkrllnaff1024
REPS="${REPS:-5}"

log "===== lnaffcast: does the core-side C un-permute beat the tap?  $(date -Is) ====="
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
rc=0

run_group(){   # $1 = group name, $2 = host suffix, $3 = must-pass arm, $4.. = arms
  local name="$1" host="$2" pass="$3"; shift 3
  log "\n========== group $name: host $host =========="
  .venv-iron/bin/python scripts/lnaffcast_burst_isolation.py \
      --host "$host" --arms "$@" --parity-must-pass "$pass" --reps "$REPS" \
      --out "artifacts/lnaffcast_scatter_$name.json" 2>&1 | tee -a "$LOG"
  local r=${PIPESTATUS[0]}; [ "$r" = 0 ] || rc=$r
}

run_group base "${BASE}rtp18g4" "${BASE}rtp18g4" \
    "${BASE}rtp18g4" "${BASE}rtp18g4ctgc" "${BASE}"
run_group scat "${BASE}scatrtp18g4" "${BASE}scatrtp18g4ctgc" \
    "${BASE}scatrtp18g4" "${BASE}scatrtp18g4ctgc" "${BASE}scat"

log ""
log "log: $LOG   rc=$rc"
exit $rc
