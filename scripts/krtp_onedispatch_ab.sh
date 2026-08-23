#!/usr/bin/env bash
# Price the one-dispatch fc2 collapse -- the FIRST execution this path has ever had.
#
# `PARAKEET_FC2_ONEDISPATCH` has been default-ON since it landed and had never run: it declines on
# `!self.krtp`, `self.krtp` comes from the resident xclbin filename, and no *krtp* artifact existed
# because build_parakeet_modal_kernels.sh has no `k_loop_rtp=1` arm. So its comment's
# "-336 of 816 dispatches/clip" was a claim, not a measurement.
#
# Two arms on the NON-FOLD default, which is where `c_elem_bytes()` is the constant 4 and the
# bf16-drain conflict with PARAKEET_FOLD_FC1 does not arise:
#
#   k0  default resident            -> fc2 K-split: 4 partials + 4 acc_add per FFN
#   k1  krtp resident (explicit)    -> one-dispatch fc2: all DFF of K in one modal dispatch
#
# The krtp resident is selected with NPU_RESIDENT_XCLBIN and deliberately does NOT live in the
# whole_array build dir: the default branch PREFERS a krtp stem, so dropping one there would
# silently change what the default encoder does on this box.
#
# 17 clips per process, not 3 -- the fold program re-priced itself three times because a per-process
# term divided by 3 is not the same statistic as one divided by 17 (fold-wall-clock-at-the-shipped-
# clip-count). Pairing is per (rep, clip).
#
# NOT bit-identical by construction: on-core accumulation rounds the same products in a different
# order than four separately-rounded partials. Gate with encoder_parity.py, never the 17-clip WER.
#
# Usage: scripts/krtp_onedispatch_ab.sh [reps] [clips]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-4}"
CLIPS="${2:-17}"
OUT="${OUT:-artifacts/krtp_onedispatch}"
BIN=rust/target/release/parakeet_encode_npu
MELS=artifacts/wer_mels
KRTP="${KRTP:-$PWD/artifacts/krtp-probe/final_512x1024x4096_64x32x128_8c_modalsilukrtp.xclbin}"
ARMS="k0 k1"

log(){ echo -e "[krtp] $*"; }
[ -x "$BIN" ] || { log "[ERR] missing $BIN -- cargo build --release -p npu-probes --bin parakeet_encode_npu"; exit 1; }
[ -f "$KRTP" ] || { log "[ERR] missing krtp resident $KRTP"; exit 1; }

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

# f32 truth, once -- the parity gate's reference. Host only, no device, so it runs outside the lock.
if [ ! -d "$OUT/ref_f32" ] || [ -z "$(ls -A "$OUT/ref_f32" 2>/dev/null)" ]; then
  log "generating f32 reference (host encoder)"
  "$BIN" "$WORK/mel" "$OUT/ref_f32" --cpu > "$OUT/ref_f32.log" 2>&1 \
    || { log "[ERR] f32 reference failed"; tail -5 "$OUT/ref_f32.log"; exit 1; }
fi

run_arm() { # $1 = arm, $2 = rep
  local arm=$1 rep=$2 rc
  local rpt="$OUT/${arm}_rep${rep}.txt"
  local resident=()
  [ "$arm" = "k1" ] && resident=(NPU_RESIDENT_XCLBIN="$KRTP")
  timeout -k 10 1800 "$NPU_LOCK_SH" queue -- \
    env PARAKEET_FOLD_FC1=0 PARAKEET_FOLD_GLU=0 NPU_XCLBIN_CACHE_BY_CONTENT=0 \
        NPU_DISPATCH_LOG=1 NPU_XCLBIN_ROOT="$PWD" "${resident[@]}" \
        "$BIN" "$WORK/mel" "$OUT/out_$arm" >"$rpt" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "[ERR] arm=$arm rep=$rep exited $rc"; tail -5 "$rpt"; return 1; }
  grep -q '^mean encode' "$rpt" || { log "[ERR] arm=$arm rep=$rep no timing line"; return 1; }
  log "  arm=$arm rep=$rep  $(grep -m1 'hw_contexts:' "$rpt")  $(grep -m1 'mean encode' "$rpt")"
}

for a in $ARMS; do rm -f "$OUT/${a}_rep"*.txt; done
for rep in $(seq 1 "$REPS"); do
  # Alternate which arm leads so a within-rep ordering effect does not load onto one arm.
  if [ $(( rep % 2 )) -eq 1 ]; then order="k0 k1"; else order="k1 k0"; fi
  for arm in $order; do run_arm "$arm" "$rep" || exit 1; done
done

log "--- wall clock + dispatch ledger ---"
python3 scripts/krtp_onedispatch_stats.py "$OUT" 2>&1 | tee "$OUT/summary.txt"

log "--- numerical parity (k1 candidate vs k0 baseline, against f32 truth) ---"
python3 scripts/encoder_parity.py "$OUT/ref_f32" "$OUT/out_k0" "$OUT/out_k1" \
  --json "$OUT/parity.json" 2>&1 | tee "$OUT/parity.txt"

log "reports under $OUT"
