#!/usr/bin/env bash
# Does the isolated restream price turn on once the LIVE FOOTPRINT reaches the encoder's?
#
# Task restream-price-in-encoder-vs-isolated. Every axis tried so far is refuted flat at zero --
# stream dissimilarity, passive hw_context residency, cross-context activity, live instruction-BO
# count -- and operand IDENTITY is refuted by the self pair. All of them held one thing fixed: HOW
# MUCH IS LIVE. The isolated probe pins ~15 MiB over a dozen zero-filled BOs; the encoder pins
# 1.25 GB (measured pinned BO bytes for the resident encoder) across one multi-MB weight BO per op and
# reads a different one every dispatch. The two surviving mechanisms -- eviction of the next
# dispatch's working set, and address-translation state where each BO is a distinct DMA mapping --
# both scale with footprint and mapping count, and with nothing else that has been swept.
#
# Each point adds N resident NON-ZERO ~7 MiB ballast streams on the measured context, dispatched
# round-robin between the measured dispatches, identically in both arms. N=176 pins ~1.2 GiB over
# ~1056 BOs, i.e. past the encoder on both axes.
#
# TURNS ON at some footprint => the price is an eviction/mapping effect, the in-encoder restream
#   cost is a property of the working set rather than of the stream change, and the mode-switch
#   viability condition needs restating.
# STAYS OFF at encoder-scale footprint => no isolated configuration reproduces it; stop building the
#   probe up and start bisecting the ENCODER down toward the probe by resident footprint.
#
# Read the measured-only absolute (us/dispatch, ballast excluded) as well as the ALT-GRP
# differential: a UNIFORM eviction hitting both arms equally is invisible to the differential, and
# the absolute is what the encoder's per-predecessor-class ms/dispatch is commensurable with.
#
# Usage: scripts/restream_footprint_sweep.sh [reps] [total] [xclbin-stem]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-20}"
TOTAL="${2:-256}"
XCLBIN="${3:-512x1024x1024_32x32x128_8c_modalidbf16out}"
POINTS="${POINTS:-0 8 16 32 64 96 128 160 176}"
OUT="${OUT:-artifacts/restream_footprint}"
mkdir -p "$OUT"

# The device is single-tenant, so every timed arm runs under a serialisation wrapper. Refuse rather
# than fall back: an unserialised timed run competes for the device and reports the contention as
# dispatch cost, which is indistinguishable from the effect being measured.
NPU_LOCK_SH="${NPU_LOCK_SH:-}"
if [ -z "$NPU_LOCK_SH" ] || [ ! -x "$NPU_LOCK_SH" ]; then
  echo "[footprint] set NPU_LOCK_SH to the device serialisation wrapper (mode: queue -- <cmd>)" >&2
  exit 1
fi

ID1024=insts_512x1024x1024_32x32x128_8c_modalidbf16out:1024:1024
ID2048=insts_512x1024x2048_32x32x128_8c_modalidbf16out:1024:2048
SILU=insts_512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024:1024:4096
# A second hw_context needs a DISTINCT xclbin stem: context() memoises per stem, so @-ing the
# default stem would hand both streams the same context and silently make the control a null arm.
ID2048_OWN=$ID2048@512x1024x2048_32x32x128_8c_modalidbf16out

log(){ echo -e "[footprint] $*"; }
trap 'npu_svc_start' EXIT
npu_svc_stop || exit 1
npu_svc_require_device_free || exit 1

# The same two measured pairs the activity sweep used, so its refuted points and these share a
# construction and the ladders can be read against each other point for point.
run_point(){ # $1=N
  local n="$1"
  log "footprint n=$n"
  timeout -k 10 3600 "$NPU_LOCK_SH" queue -- \
    python3 scripts/restream_similarity_ab.py \
      --root "$PWD" --xclbin "$XCLBIN" \
      --pair "onectx_id1024_id2048=$ID1024,$ID2048" \
      --pair "onectx_id1024_silupanel=$ID1024,$SILU" \
      --footprint-bos "$n" --footprint-spec "$ID2048" \
      --total "$TOTAL" --reps "$REPS" \
      --out "$OUT/fp_n${n}.json" \
    2>&1 | tee "$OUT/fp_n${n}.log"
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

for n in $POINTS; do run_point "$n"; done

python3 scripts/restream_activity_summary.py --dir "$OUT" 2>&1 | tee "$OUT/summary.txt"
log "done -> $OUT"
