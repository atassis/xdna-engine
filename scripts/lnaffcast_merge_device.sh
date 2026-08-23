#!/usr/bin/env bash
# Device gate for the lnaffcast mode ON THE ENCODER'S OWN RESIDENT
# (task mode-switched-multi-program-xclbin, its `next:` item 1 -- "the BUILD").
#
# THE QUESTION. lnaffcast_merge_build.sh showed the merge LINKS at the encoder's geometry and costs
# the shipped fc1 nothing at the artifact level: core ELFs identical across the two streams, fc1's
# instruction stream byte-identical to today's, L1 allocation unmoved. None of that is correctness.
# The 1x route's parity was established on `512x1024x1024_..._modalidbf16outkrtpkrl`, and three of
# the things it turns on are geometry-dependent -- x is read out of the B tensor, which is 4x larger
# here; the C drain shares a design with a panel-major GEMM drain; and the branch now falls through
# a silu epilogue rather than an identity one. So parity at N=4096 is a question, not a formality.
#
# ARMS -- all three streams on the ONE loaded xclbin, no program transition inside the comparison:
#   <mode>r512g4ctgc  the merge candidate, 512 rows = the encoder's only lnaffcast row count -> PASS
#   <mode>r512g4      the DERIVED C tap against the scatter-2 body                           -> FAIL
#   <gemm>            the fc1 GEMM stream on the merged xclbin                     -> control, FAIL
# The GEMM arm is expected to fail parity against an LN reference -- it computes a GEMM. What it is
# here for is its WALL CLOCK: it is the fc1 dispatch the encoder actually runs, on the merged
# xclbin, so it is the arm that would show the added branch charging the shipped path.
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies free with fuser + pgrep (never
# `systemctl is-active` -- it prints inactive for stopped, absent AND running-as-a-plain-process
# alike), and ALWAYS restores on exit including on abort.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/lnaffcast_merge.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

SHIPPED=512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024
X1=${SHIPPED}lnaff1024scat21x
REPS="${REPS:-5}"
ROWS="${ROWS:-512}"

log "===== lnaffcast merge: does the mode hold at the encoder's geometry?  $(date -Is) ====="
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

group(){   # $1 = out tag, $2 = host, $3 = must-pass ("" = every arm must FAIL), $4.. = arms
  local tag="$1" host="$2" pass="$3"; shift 3
  log "\n========== group $tag: host $host =========="
  .venv-iron/bin/python scripts/lnaffcast_burst_isolation.py \
      --host "$host" --arms "$@" --parity-must-pass $pass \
      --reps "$REPS" --rows "$ROWS" --one-x \
      --out "artifacts/lnaffcast_merge_$tag.json" 2>&1 | tee -a "$LOG"
  local r=${PIPESTATUS[0]}; [ "$r" = 0 ] || rc=$r
}

# GROUP mode -- correctness of the merge, plus its own cost.
group mode "${X1}rtp18r${ROWS}g4ctgc" "${X1}rtp18r${ROWS}g4ctgc" \
      "${X1}rtp18r${ROWS}g4ctgc" "${X1}rtp18r${ROWS}g4" "${X1}"

# WHAT THE MERGE CHARGES THE SHIPPED PATH, and the reason this is bracketed rather than a single
# pair. fc1's instruction stream is byte-identical on both xclbins, so the only thing that can move
# its wall clock is the core body -- and the body DID change: the GEMM path now takes its first A
# acquire before the rtp branch and runs k_trip-1 in the loop instead of k_trip. That peel is what
# gets the branch behind the rtp visibility lag, so it is not removable, and whether it costs is a
# measurement. Cross-xclbin means a program transition between the two arms, so the shipped arm runs
# BEFORE and AFTER the merged one and the repeat bounds the session drift against the effect.
group gemm_shipped_pre "$SHIPPED" "" "$SHIPPED"
group gemm_merged      "$X1"      "" "$X1"
group gemm_shipped_post "$SHIPPED" "" "$SHIPPED"

log ""
log "log: $LOG   rc=$rc"
exit $rc
