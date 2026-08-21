#!/usr/bin/env bash
# Does the isolated restream price turn on with cross-context or cross-BO ACTIVITY?
#
# Task restream-price-in-encoder-vs-isolated. Dissimilarity and PASSIVE context residency are both
# refuted flat at zero, so the surviving suspects -- BO churn and instruction-BO eviction -- are
# tested on the axis those sweeps held fixed. Each point interleaves N ballast dispatches between the
# measured ones, identically in both arms; the two axes differ only in what the ballast consumes:
#
#   ctx  N ballast streams on N own hw_contexts   -- many contexts in flight
#   bos  N ballast streams on the measured one    -- many instruction BOs in flight, content pinned
#
# Turns on => the mechanism is whichever axis crossed. Stays off => neither, and the next suspects
# are the engine's own BO lifecycle and the weight-arena footprint the zero-filled operands miss.
#
# Usage: scripts/restream_activity_sweep.sh [reps] [total] [xclbin-stem]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-20}"
TOTAL="${2:-256}"
XCLBIN="${3:-512x1024x1024_32x32x128_8c_modalidbf16out}"
POINTS="${POINTS:-0 1 2 3 4 5 6 8}"
OUT="${OUT:-artifacts/restream_activity}"
mkdir -p "$OUT"

# The device is single-tenant, so every timed arm runs under a serialisation wrapper. Refuse rather
# than fall back: an unserialised timed run competes for the device and reports the contention as
# dispatch cost, which is indistinguishable from the effect being measured.
NPU_LOCK_SH="${NPU_LOCK_SH:-}"
if [ -z "$NPU_LOCK_SH" ] || [ ! -x "$NPU_LOCK_SH" ]; then
  echo "[activity] set NPU_LOCK_SH to the device serialisation wrapper (mode: queue -- <cmd>)" >&2
  exit 1
fi

ID1024=insts_512x1024x1024_32x32x128_8c_modalidbf16out:1024:1024
ID2048=insts_512x1024x2048_32x32x128_8c_modalidbf16out:1024:2048
SILU=insts_512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024:1024:4096
# A second hw_context needs a DISTINCT xclbin stem: context() memoises per stem, so @-ing the
# default stem would hand both streams the same context and silently make the control a null arm.
ID2048_OWN=$ID2048@512x1024x2048_32x32x128_8c_modalidbf16out

log(){ echo -e "[activity] $*"; }
trap 'npu_svc_start' EXIT
npu_svc_stop || exit 1
npu_svc_require_device_free || exit 1

# One measured pair per encoder-relevant layout: id2048 repeats the refuted sweep's pair so the
# points are comparable to it, silupanel is the cross-layout pair the encoder priced at +2258 us.
run_point(){ # $1=axis flag  $2=N  $3=tag
  local flag="$1" n="$2" tag="$3"
  log "$tag n=$n"
  timeout -k 10 1800 "$NPU_LOCK_SH" queue -- \
    python3 scripts/restream_similarity_ab.py \
      --root "$PWD" --xclbin "$XCLBIN" \
      --pair "onectx_id1024_id2048=$ID1024,$ID2048" \
      --pair "onectx_id1024_silupanel=$ID1024,$SILU" \
      "$flag" "$n" --total "$TOTAL" --reps "$REPS" \
      --out "$OUT/${tag}_n${n}.json" \
    2>&1 | tee "$OUT/${tag}_n${n}.log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || log "  rc=$rc"
}

# The two-context arm is the instrument's positive control: a harness that cannot resolve +2.3 ms
# here has no power to call any sweep point null.
log "positive control (two hw_contexts, no ballast)"
timeout -k 10 1800 "$NPU_LOCK_SH" queue -- \
  python3 scripts/restream_similarity_ab.py \
    --root "$PWD" --xclbin "$XCLBIN" \
    --pair "twoctx_id1024_id2048=$ID1024,$ID2048_OWN" \
    --total "$TOTAL" --reps "$REPS" --out "$OUT/control_twoctx.json" \
  2>&1 | tee "$OUT/control_twoctx.log"

for n in $POINTS; do run_point --interleave-contexts "$n" ctx; done
for n in $POINTS; do run_point --interleave-bos "$n" bos; done

python3 scripts/restream_activity_summary.py --dir "$OUT" 2>&1 | tee "$OUT/summary.txt"
log "done -> $OUT"
