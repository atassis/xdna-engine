#!/usr/bin/env bash
# BRACKET the lnaffcast mode ON THE FOLD -- task mode-switched-multi-program-xclbin, `next:` item 1.
#
# THE QUESTION. The -172 ms/clip the mode reads on the fold arm (blocking dispatch 1.511 -> 1.339 s)
# is a SINGLE UNBRACKETED PASS. Its transition count is exact -- 695 -> 575, counted at every
# dispatch, not timed -- but the time is one sample of a statistic this task has already been burned
# by: REPS=5 read a comparison 11% off here before. So the count is evidence and the millisecond
# figure is not, until it is bracketed forward/reversed with >=11 reps.
#
# THE CONTROL THAT IS BUILT IN. `PARAKEET_FOLD_FC1=1` did NOT actually fold until a59b8b5
# (2026-08-23 00:49) resolved the panel's two byte-identical paths to one picker: before it, the arm
# read 743 transitions / 12 hw_contexts -- the same as the unfolded default -- and only
# `NPU_XCLBIN_CACHE_BY_CONTENT=1` merged the duplicate to 695/11. `cache_id`'s own doc states the
# post-fix expectation: "a caller that has [resolved to one path] reads the same hw_contexts count
# with this flag on and off". So the cache flag is carried here as a CONTROL, not as a factor: f1
# must reproduce f0. If it does not, a second duplicate context is still live and every ms in this
# table is measuring that instead of the mode.
#
# ARMS -- all four hold PARAKEET_FOLD_FC1=1; they differ only in the mode and the control:
#   f0   fold                          the prior pass's fold base   (expect 695 trans, 11 ctx)
#   f0m  fold + LN_MODE                the prior pass's fold mode   (expect 575 trans, 11 ctx)
#   f1   fold + cache-by-content       control for f0  -- must match f0
#   f1m  fold + LN_MODE + cache        control for f0m -- must match f0m
#
# THE FOLD ARM'S ENCODER OUTPUT IS WRONG BY DESIGN. `fold_fc1` is timing-only until the bf16
# consumer gap closes, so this script brackets TIME and TRANSITIONS ONLY and is NOT a correctness
# gate. Correctness for the mode is gated on the HYBRID arm by lnaffcast_mode_encoder_gate.sh, which
# reads rel-L2 0.0000e+00 on all 17 clips. Nothing here can move that verdict in either direction.
#
# ORDERING. Each rep runs all four arms; the leading arm rotates by rep AND the traversal direction
# reverses on alternate reps, so neither a within-rep position effect nor a monotone box drift can
# load onto one arm. Pairing is per (rep, clip) -- the probe times each clip and clip 1 carries the
# cold weight-BO load in EVERY arm, so pairing on clip index keeps that cost on both sides of the
# difference rather than averaging it into one arm's mean.
#
# Usage: scripts/lnmode_fold_bracket_ab.sh [reps] [clips]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-11}"
CLIPS="${2:-3}"
OUT="${OUT:-artifacts/lnmode_fold_bracket}"
BIN=rust/target/release/parakeet_encode_npu
MELS=artifacts/wer_mels
ARMS="f0 f0m f1 f1m"

log(){ echo -e "[lnbr] $*"; }
[ -x "$BIN" ] || { log "[ERR] missing $BIN -- cargo build --release -p npu-probes --bin parakeet_encode_npu"; exit 1; }

NPU_LOCK_SH="${NPU_LOCK_SH:-}"
if [ -z "$NPU_LOCK_SH" ] || [ ! -x "$NPU_LOCK_SH" ]; then
  log "[ERR] set NPU_LOCK_SH to the device serialisation wrapper (mode: queue -- <cmd>)"; exit 1
fi

WORK="$(mktemp -d)"
mkdir -p "$WORK/mel" "$OUT"
i=0
for f in $(ls "$MELS"/*.npy | sort); do
  [ "$i" -ge "$CLIPS" ] && break
  ln -sf "$(readlink -f "$f")" "$WORK/mel/$(basename "$f")"
  i=$((i + 1))
done
log "clips/rep = $i, reps = $REPS, arms = $ARMS"
trap 'rm -rf "$WORK"; npu_svc_start' EXIT

npu_svc_stop || exit 1
npu_svc_require_device_free || exit 1

run_arm() { # $1 = arm, $2 = rep
  local arm=$1 rep=$2 rc mode=0 content=0
  local rpt="$OUT/${arm}_rep${rep}.txt"
  case "$arm" in
    f0)  mode=0; content=0 ;;
    f0m) mode=1; content=0 ;;
    f1)  mode=0; content=1 ;;
    f1m) mode=1; content=1 ;;
  esac
  timeout -k 10 900 "$NPU_LOCK_SH" queue -- \
    env PARAKEET_FOLD_FC1=1 PARAKEET_LN_MODE=$mode NPU_XCLBIN_CACHE_BY_CONTENT=$content \
        NPU_DISPATCH_LOG=1 NPU_XCLBIN_ROOT="$PWD" \
        "$BIN" "$WORK/mel" "$WORK/out_$arm" >"$rpt" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "[ERR] arm=$arm rep=$rep exited $rc"; tail -5 "$rpt"; return 1; }
  grep -q '^mean encode' "$rpt" || { log "[ERR] arm=$arm rep=$rep no timing line"; return 1; }
  log "  arm=$arm rep=$rep  $(grep -m1 'hw_contexts:' "$rpt")  $(grep -m1 -oE 'transitions [0-9]+' "$rpt")  $(grep -m1 'mean encode' "$rpt")"
}

for a in $ARMS; do rm -f "$OUT/${a}_rep"*.txt; done
for rep in $(seq 1 "$REPS"); do
  # Rotate the leading arm, and reverse the traversal on alternate reps -- forward/reversed.
  order=""; k=$(( (rep - 1) % 4 )); n=0
  for a in $ARMS; do n=$((n + 1)); [ $n -gt $k ] && order="$order $a"; done
  n=0
  for a in $ARMS; do n=$((n + 1)); [ $n -le $k ] && order="$order $a"; done
  if [ $(( rep % 2 )) -eq 0 ]; then
    rev=""; for a in $order; do rev="$a $rev"; done; order="$rev"
  fi
  log "rep $rep order:$order"
  for arm in $order; do run_arm "$arm" "$rep" || exit 1; done
  python3 scripts/lnmode_fold_bracket_stats.py "$OUT" >"$OUT/summary.txt" 2>&1
done

log "--- results ---"
python3 scripts/lnmode_fold_bracket_stats.py "$OUT" 2>&1 | tee "$OUT/summary.txt"
log "reports under $OUT"
