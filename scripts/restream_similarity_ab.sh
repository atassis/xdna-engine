#!/usr/bin/env bash
# Sweep the restream price against stream dissimilarity, isolated, under the quiesce guard.
#
# Task restream-price-similarity-or-flat. All four streams are 2484 words and link the same core
# objects, so they are mutually dispatchable on one xclbin; they differ in the BD size/stride fields,
# the drain layout, and the epilogue mode their embedded rtp_write selects.
#
# The ladder, by differing words against the id N=1024 stream:
#   self      0 words  -- same file, two BOs: isolates instruction-BO switching from content
#   id2048  224 words  -- row-major drain both sides, differs in N
#   silu    240 words  -- panel-major vs row-major drain: the cross-layout pair the encoder measured
#
# Usage: scripts/restream_similarity_ab.sh [reps] [total] [xclbin-stem]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-30}"
TOTAL="${2:-256}"
XCLBIN="${3:-512x1024x1024_32x32x128_8c_modalidbf16out}"
OUT="${OUT:-artifacts/restream_similarity}"
mkdir -p "$OUT"

ID1024=insts_512x1024x1024_32x32x128_8c_modalidbf16out:1024:1024
ID2048=insts_512x1024x2048_32x32x128_8c_modalidbf16out:1024:2048
SILU=insts_512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024:1024:4096

log(){ echo -e "[similarity] $*"; }
trap 'npu_svc_start' EXIT
npu_svc_stop || exit 1
npu_svc_require_device_free || exit 1

log "xclbin=$XCLBIN reps=$REPS total=$TOTAL"
timeout -k 10 1800 ../xdna-engine-private/journal/scripts/npu_lock.sh run -- \
  python3 scripts/restream_similarity_ab.py \
    --root "$PWD" --xclbin "$XCLBIN" \
    --pair "self_id1024=$ID1024,$ID1024" \
    --pair "id1024_vs_id2048=$ID1024,$ID2048" \
    --pair "id1024_vs_silupanel=$ID1024,$SILU" \
    --pair "id2048_vs_silupanel=$ID2048,$SILU" \
    --total "$TOTAL" --reps "$REPS" \
    --out "$OUT/${XCLBIN}_r${REPS}_t${TOTAL}.json" \
  2>&1 | tee "$OUT/${XCLBIN}_r${REPS}_t${TOTAL}.log"
rc=${PIPESTATUS[0]}
log "rc=$rc"
exit "$rc"
