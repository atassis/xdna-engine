#!/usr/bin/env bash
# Measure the IN-ENCODER restream cost directly, paired per rep, under the quiesce guard.
#
# The question (task in-encoder-restream-direct-measure): the "~0.6-0.7 ms per restream" residual is
# backed out of an end-to-end delta -- 48 deleted transitions worth 74-85 ms against a ~17 ms tile
# penalty predicted a ~57-68 ms win, and the drift-cancelling A/B measured NEUTRAL. Both terms in that
# subtraction are projections, so the residual is an inference, not a measurement.
#
# What this measures instead: `NPU_DISPATCH_LOG=1` splits each (xclbin, instruction stream) row into
# dispatches whose predecessor was the SAME stream and those whose predecessor was a DIFFERENT stream
# on the same xclbin. Within one row the work is identical and the two populations interleave in
# dispatch order inside a single clip, so their difference is the reconfiguration cost and nothing
# else. Reps give the error bar; the arms are interleaved so box drift cancels between them too.
#
# Usage: scripts/restream_in_encoder_ab.sh [reps] [clips]
set -u

# Source the quiesce guard BEFORE any cd: BASH_SOURCE is whatever was typed, and these scripts cd to
# the git common dir, which from a worktree is ANOTHER checkout (see the 2026-08-17 quiesce fact).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1

cd "$HERE/.." || exit 1

REPS="${1:-12}"
CLIPS="${2:-4}"
OUT="${OUT:-artifacts/restream_ab}"
BIN=rust/target/release/parakeet_encode_npu
MELS=artifacts/wer_mels

log(){ echo -e "[restream] $*"; }

[ -x "$BIN" ] || { log "[ERR] missing $BIN -- cargo build --release -p npu-probes --bin parakeet_encode_npu"; exit 1; }

# A warm-clip subset: the dispatch log resets per clip and the report describes the LAST one, so the
# leading clips only exist to get past the cold weight-BO load.
WORK="$(mktemp -d)"
mkdir -p "$WORK/mel" "$OUT"
i=0
for f in $(ls "$MELS"/*.npy | sort); do
  [ "$i" -ge "$CLIPS" ] && break
  ln -sf "$(readlink -f "$f")" "$WORK/mel/$(basename "$f")"
  i=$((i + 1))
done
log "clips/rep = $i, reps = $REPS"
trap 'rm -rf "$WORK"; npu_svc_start' EXIT

npu_svc_stop || exit 1
npu_svc_require_device_free || exit 1

run_arm() {   # $1 = arm name, $2 = rep index
  local arm=$1 rep=$2 rc
  local rpt="$OUT/${arm}_rep${rep}.txt"
  local fold=0
  [ "$arm" = fold ] && fold=1
  PARAKEET_FOLD_FC1=$fold NPU_DISPATCH_LOG=1 NPU_XCLBIN_ROOT="$PWD" \
    "$BIN" "$WORK/mel" "$WORK/out_$arm" >"$rpt" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    log "[ERR] arm=$arm rep=$rep exited $rc; tail:"; tail -5 "$rpt"; return 1
  fi
  grep -q "dispatches by PREDECESSOR" "$rpt" || { log "[ERR] arm=$arm rep=$rep produced no table"; return 1; }
  log "  arm=$arm rep=$rep ok  ($(grep -m1 'mean encode' "$rpt"))"
}

rm -f "$OUT"/base_rep*.txt "$OUT"/fold_rep*.txt
for rep in $(seq 1 "$REPS"); do
  # Alternate which arm leads so a within-rep ordering effect does not load onto one arm.
  if [ $((rep % 2)) -eq 1 ]; then order="base fold"; else order="fold base"; fi
  for arm in $order; do
    run_arm "$arm" "$rep" || exit 1
  done
done

log "--- results ---"
python3 scripts/restream_in_encoder_stats.py --label "(arm=base, shipped 3-xclbin rotation)" "$OUT"/base_rep*.txt
python3 scripts/restream_in_encoder_stats.py --label "(arm=fold, PARAKEET_FOLD_FC1=1)" "$OUT"/fold_rep*.txt
log "reports under $OUT"
