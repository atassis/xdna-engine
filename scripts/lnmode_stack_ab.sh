#!/usr/bin/env bash
# THE STACKED TOTAL, IN ONE WINDOW -- task mode-switched-multi-program-xclbin, `next:` item b.
#
# WHY THIS RUN EXISTS. The two levers were each bracketed against their OWN baseline in their own
# session: the fold+glu+krtpkrl composition against the shipped default (-624.9 ms/clip wall), and
# the lnaffcast mode against that composition (-211.4 ms/clip wall). Adding them is an ADDITION
# ACROSS WINDOWS, not a measurement -- different sessions, different thermal state, a rebuilt
# binary -- and a stacked figure quoted that way is exactly the kind of number this task has twice
# had to withdraw. Three arms in one quiesced window is the only honest way to say what a caller
# would get.
#
# ARMS:
#   s0   shipped default, no flags        what a caller gets today (expect 743 transitions)
#   k0   FOLD_FC1 + FOLD_GLU + krtpkrl    the composition                            (335)
#   k0m  + LN_MODE                        the composition with the merge             (192)
#
# NOT A CORRECTNESS GATE. Each arm is gated in its own right -- the composition by fold_krtp_ab.sh's
# parity pass against f32 truth, the mode bit-exact on both residents -- and s0's output legitimately
# differs from k0's, since on-core accumulation rounds in a different order than four separately
# rounded partials. This run measures TIME and TRANSITIONS.
#
# ORDERING. The leading arm rotates by rep and the traversal reverses on alternate reps, so neither
# a within-rep position effect nor a monotone drift can load onto one arm. Pairing is per
# (rep, clip) -- clip 1 carries the cold weight-BO load in every arm.
#
# Usage: scripts/lnmode_stack_ab.sh [reps] [clips]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-9}"
CLIPS="${2:-3}"
OUT="${OUT:-artifacts/lnmode_stack}"
BIN=rust/target/release/parakeet_encode_npu
MELS=artifacts/wer_mels
ARMS="s0 k0 k0m"

log(){ echo -e "[stack] $*"; }
[ -x "$BIN" ] || { log "[ERR] missing $BIN"; exit 1; }

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
  local arm=$1 rep=$2 rc; local env_=()
  local rpt="$OUT/${arm}_rep${rep}.txt"
  [ "$arm" != s0 ] && env_=(PARAKEET_FOLD_FC1=1 PARAKEET_FOLD_GLU=1 PARAKEET_MODAL_EPI_SUFFIX=krtpkrl)
  [ "$arm" = k0m ] && env_+=(PARAKEET_LN_MODE=1)
  timeout -k 10 900 "$NPU_LOCK_SH" queue -- \
    env NPU_XCLBIN_CACHE_BY_CONTENT=0 NPU_DISPATCH_LOG=1 NPU_XCLBIN_ROOT="$PWD" "${env_[@]}" \
        "$BIN" "$WORK/mel" "$WORK/out_$arm" >"$rpt" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "[ERR] arm=$arm rep=$rep exited $rc"; tail -8 "$rpt"; return 1; }
  grep -q '^mean encode' "$rpt" || { log "[ERR] arm=$arm rep=$rep no timing line"; return 1; }
  log "  arm=$arm rep=$rep  $(grep -m1 'hw_contexts:' "$rpt")  $(grep -m1 -oE 'transitions [0-9]+' "$rpt")  $(grep -m1 'mean encode' "$rpt")"
}

for a in $ARMS; do rm -f "$OUT/${a}_rep"*.txt; done
for rep in $(seq 1 "$REPS"); do
  arr=($ARMS); n=${#arr[@]}; order=()
  for i in $(seq 0 $((n - 1))); do order+=("${arr[$(( (rep - 1 + i) % n ))]}"); done
  if [ $(( rep % 2 )) -eq 0 ]; then
    rev=(); for a in "${order[@]}"; do rev=("$a" "${rev[@]}"); done; order=("${rev[@]}")
  fi
  log "rep $rep order: ${order[*]}"
  for arm in "${order[@]}"; do run_arm "$arm" "$rep" || exit 1; done
  python3 scripts/lnmode_fold_bracket_stats.py "$OUT" stack >"$OUT/summary.txt" 2>&1
done

log "--- results ---"
python3 scripts/lnmode_fold_bracket_stats.py "$OUT" stack 2>&1 | tee "$OUT/summary.txt"
log "reports under $OUT"
