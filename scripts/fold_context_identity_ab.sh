#!/usr/bin/env bash
# Does PARAKEET_FOLD_FC1 actually fold, or does it only fold in the LOG?
#
# The fold merges fc1 onto the resident by pointing both at "the same xclbin" and relying on
# `Device::load_kernel` caching by PATH. But the resident resolves the stem under the whole_array
# build dir and `Fc1PanelBf16` resolves it under artifacts/parakeet/ln -- byte-identical copies at
# two paths, so the cache misses and the two live in SEPARATE hw_contexts wearing one stem. Every
# fc1<->modal boundary is then a program transition that `dispatch_log` reports as a same-xclbin
# instruction restream, which is where the encoder's "+2.26 ms restream" came from and why seven
# isolated axes refuted it.
#
# Three arms:
#   base  -- shipped 3-xclbin rotation (control)
#   fold  -- PARAKEET_FOLD_FC1=1, the fold as it stands
#   foldc -- the same, plus NPU_XCLBIN_CACHE_BY_CONTENT=1, which keys the context cache on the
#            xclbin's CONTENT so the two copies share one context and the fold actually folds
#
# Reads out per arm: hw_contexts used, per-predecessor dispatch table, mean encode.
#   fold == base on hw_contexts  => the fold never folded (the log's restream was a transition).
#   foldc == base - 1            => the contexts merged; whatever the restream row then costs is
#                                   the price of a REAL same-context stream change in the encoder.
#
# Usage: scripts/fold_context_identity_ab.sh [reps] [clips]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-12}"
CLIPS="${2:-3}"
OUT="${OUT:-artifacts/fold_ctx_identity}"
BIN=rust/target/release/parakeet_encode_npu
MELS=artifacts/wer_mels

log(){ echo -e "[foldctx] $*"; }
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
log "clips/rep = $i, reps = $REPS"
trap 'rm -rf "$WORK"; npu_svc_start' EXIT

npu_svc_stop || exit 1
npu_svc_require_device_free || exit 1

run_arm() { # $1 = arm, $2 = rep
  local arm=$1 rep=$2 rc fold=0 content=0
  local rpt="$OUT/${arm}_rep${rep}.txt"
  case "$arm" in
    base)  fold=0; content=0 ;;
    fold)  fold=1; content=0 ;;
    foldc) fold=1; content=1 ;;
  esac
  timeout -k 10 900 "$NPU_LOCK_SH" queue -- \
    env PARAKEET_FOLD_FC1=$fold NPU_XCLBIN_CACHE_BY_CONTENT=$content \
        NPU_DISPATCH_LOG=1 NPU_XCLBIN_ROOT="$PWD" \
        "$BIN" "$WORK/mel" "$WORK/out_$arm" >"$rpt" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "[ERR] arm=$arm rep=$rep exited $rc"; tail -5 "$rpt"; return 1; }
  grep -q "dispatches by PREDECESSOR" "$rpt" || { log "[ERR] arm=$arm rep=$rep no table"; return 1; }
  log "  arm=$arm rep=$rep  $(grep -m1 'hw_contexts:' "$rpt")  $(grep -m1 'mean encode' "$rpt")"
}

rm -f "$OUT"/base_rep*.txt "$OUT"/fold_rep*.txt "$OUT"/foldc_rep*.txt
# Rotate which arm leads so a within-rep ordering effect does not load onto one arm.
for rep in $(seq 1 "$REPS"); do
  case $((rep % 3)) in
    1) order="base fold foldc" ;;
    2) order="fold foldc base" ;;
    0) order="foldc base fold" ;;
  esac
  for arm in $order; do run_arm "$arm" "$rep" || exit 1; done
done

log "--- results ---"
python3 scripts/fold_context_identity_stats.py "$OUT" 2>&1 | tee "$OUT/summary.txt"
log "reports under $OUT"
