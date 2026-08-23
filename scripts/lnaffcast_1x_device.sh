#!/usr/bin/env bash
# Device wrapper for the 1x route (task mode-switched-multi-program-xclbin, its `next:` item 1).
#
# THE QUESTION. lnaffcast-mode-duplicates-output-across-the-a-broadcast measured that the 8 columns
# of an array row share one A fifo and so compute the SAME output: 4 useful C tiles per round of
# 32. The 1x route swaps the operands -- x onto the per-column B fifo, gb onto the per-row A one --
# and breaks the row axis with a skip. Derived traffic goes 24 -> 6.25 KB per output row. Does the
# wall clock follow, or does the skip's serialisation of a column's n_aie_rows cores eat it?
#
# TWO GROUPS, ONE QUIESCED SESSION. lnaffcast_1x changes the array program, so the candidate cannot
# share the 8x arm's xclbin and the comparison is CROSS-xclbin. Each group therefore runs the GEMM
# stream on its OWN xclbin as the control (gemm-stream-controls-the-cross-xclbin-comparison): the
# GEMM stream is untouched by the route, so its two medians agreeing is what says the vehicles are
# comparable and the delta is the route rather than the session.
#
#   group 8x, scat2 xclbin: <s2>rtp18g4ctgc   the settled best, 1083.8 us at 256 rows -> must PASS
#                           <s2>rtp18g4       derived C tap against the scatter body -> must FAIL
#                           <s2>              GEMM stream                            -> control
#   group 1x, 1x xclbin:    <s2 1x>r256g4ctgc the candidate, same 256 rows           -> must PASS
#                           <s2 1x>r256g4     derived C tap                          -> must FAIL
#                           <s2 1x>           GEMM stream                            -> control
#
# The two groups need DIFFERENT operand layouts (the 1x route reads x off B), which is what
# --one-x is: same values, same borrow, same reference, so both are gated on one parity number.
#
# Single-tenant NPU: stops xdna-engine + npu-vox, verifies the device is actually free, and ALWAYS
# restores them on exit including on abort. The free-device check is fuser + pgrep, never
# `systemctl is-active` -- it prints inactive for stopped, absent AND running-as-a-plain-process
# alike, and `npu-asr` was renamed to `xdna-engine`, so the old name returns a false all-clear.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LOG="$WT/artifacts/lnaffcast_1x.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

BASE=512x1024x1024_32x32x128_8c_modalidbf16outkrtpkrllnaff1024
S2=${BASE}scat2
X1=${BASE}scat21x
REPS="${REPS:-5}"
ROWS="${ROWS:-256}"

log "===== lnaffcast 1x route: does deleting the 8x write duplication pay?  $(date -Is) ====="
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

run_group(){   # $1 = name, $2 = host, $3 = must-pass arm, $4 = extra flags, $5.. = arms
  local name="$1" host="$2" pass="$3" extra="$4"; shift 4
  log "\n========== group $name: host $host =========="
  # shellcheck disable=SC2086
  .venv-iron/bin/python scripts/lnaffcast_burst_isolation.py \
      --host "$host" --arms "$@" --parity-must-pass "$pass" --reps "$REPS" \
      --rows "${ROWS}" $extra \
      --out "artifacts/lnaffcast_1x_$name.json" 2>&1 | tee -a "$LOG"
  local r=${PIPESTATUS[0]}; [ "$r" = 0 ] || rc=$r
}

run_group 8x "${S2}rtp18g4ctgc" "${S2}rtp18g4ctgc" "" \
    "${S2}rtp18g4ctgc" "${S2}rtp18g4" "${S2}"
run_group 1x "${X1}rtp18r${ROWS}g4ctgc" "${X1}rtp18r${ROWS}g4ctgc" "--one-x" \
    "${X1}rtp18r${ROWS}g4ctgc" "${X1}rtp18r${ROWS}g4" "${X1}"

# ROW SWEEP -- the 1x route's cost per row is not its cost. lnaffcast_rows is insts-only, so 256
# and 512 rows are two streams on the ONE xclbin already loaded above, which makes the pair a
# clean two-point fit: marginal = T512 - T256 and floor = 2*T256 - T512, with no transition and
# no second vehicle inside the subtraction. Worth doing here because the 1x arm landed close
# enough to the GEMM stream's own median that a fixed per-dispatch cost could be most of it --
# and if it is, the tap's burst length stops being the lever it was at 8x.
ROWS=512 run_group 1x512 "${X1}rtp18r512g4ctgc" "${X1}rtp18r512g4ctgc" "--one-x" \
    "${X1}rtp18r512g4ctgc" "${X1}"

log ""
log "log: $LOG   rc=$rc"
exit $rc
