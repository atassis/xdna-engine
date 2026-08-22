#!/usr/bin/env bash
# Re-price the fold's wall clock now that both device defects are closed.
#
# The prior figure, -114.8 ms/clip, was a TIMING-ONLY reading taken while the arm's encoder output
# was still numerically wrong, and it predates the bf16-addend resadd/acc_add arms and the folded GLU
# epilogue (both add or move dispatches). Six arms, the 3 flag settings x cache-by-content, so the
# fold's price and the context cache's price are separable rather than confounded:
#
#   b0 default            b1 default + content-keyed context cache
#   f0 FOLD_FC1           f1 FOLD_FC1 + cache      <- f0 does not actually fold (two contexts)
#   p0 FOLD_FC1+FOLD_GLU  p1 the pair + cache      <- p1 is the shippable arm
#
# All six interleave INSIDE a rep and the leading arm rotates, so box drift pairs out per rep and no
# arm carries the cold-cache slot. Pairing is per (rep, clip): the probe prints each clip's own time.
#
# MEASURED 2026-08-23 (12 reps x 3 clips): p1 is -175.3 ms/clip vs b0, CI [-234.4, -116.2], 34/36
# pairs. The win is the CONTEXT MERGE, not the fold -- b1 is +6.9 [-42.9, 56.7] and f0 is +45.1
# [-11.6, 101.7], so neither alone moves anything, while the cache buys -166.3 on f0 and -171.9 on
# p0. f1/p1 are the only arms reading 11 hw_contexts (the rest read 12): the fold resolves one stem
# to two byte-identical files and thereby CREATES the duplicate context that content-keying merges.
# The -166 ms deletes zero dispatches; the -24/clip GLU deletion is the smaller, weaker half
# (-54.1 [-98.6, -9.5], but only 22/36 pairs -- drift-grade by this instrument's own sign rule).
#
# Usage: scripts/fold_pair_wallclock_ab.sh [reps] [clips]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-10}"
CLIPS="${2:-3}"
OUT="${OUT:-artifacts/fold_pair_wallclock}"
BIN=rust/target/release/parakeet_encode_npu
MELS=artifacts/wer_mels
ARMS="b0 b1 f0 f1 p0 p1"

log(){ echo -e "[foldwc] $*"; }
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
  local arm=$1 rep=$2 rc fc1=0 glu=0 content=0
  local rpt="$OUT/${arm}_rep${rep}.txt"
  case "$arm" in
    b0) fc1=0; glu=0; content=0 ;;
    b1) fc1=0; glu=0; content=1 ;;
    f0) fc1=1; glu=0; content=0 ;;
    f1) fc1=1; glu=0; content=1 ;;
    p0) fc1=1; glu=1; content=0 ;;
    p1) fc1=1; glu=1; content=1 ;;
  esac
  timeout -k 10 900 "$NPU_LOCK_SH" queue -- \
    env PARAKEET_FOLD_FC1=$fc1 PARAKEET_FOLD_GLU=$glu NPU_XCLBIN_CACHE_BY_CONTENT=$content \
        NPU_DISPATCH_LOG=1 NPU_XCLBIN_ROOT="$PWD" \
        "$BIN" "$WORK/mel" "$WORK/out_$arm" >"$rpt" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "[ERR] arm=$arm rep=$rep exited $rc"; tail -5 "$rpt"; return 1; }
  grep -q '^mean encode' "$rpt" || { log "[ERR] arm=$arm rep=$rep no timing line"; return 1; }
  log "  arm=$arm rep=$rep  $(grep -m1 'hw_contexts:' "$rpt")  $(grep -m1 'mean encode' "$rpt")"
}

for a in $ARMS; do rm -f "$OUT/${a}_rep"*.txt; done
for rep in $(seq 1 "$REPS"); do
  # Rotate the leading arm so a within-rep ordering effect does not load onto one arm.
  order=""; k=$(( (rep - 1) % 6 )); n=0
  for a in $ARMS; do
    n=$((n + 1))
    [ $n -gt $k ] && order="$order $a"
  done
  n=0
  for a in $ARMS; do
    n=$((n + 1))
    [ $n -le $k ] && order="$order $a"
  done
  for arm in $order; do run_arm "$arm" "$rep" || exit 1; done
  python3 scripts/fold_pair_wallclock_stats.py "$OUT" >"$OUT/summary.txt" 2>&1
done

log "--- results ---"
python3 scripts/fold_pair_wallclock_stats.py "$OUT" 2>&1 | tee "$OUT/summary.txt"
log "reports under $OUT"
