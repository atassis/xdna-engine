#!/usr/bin/env bash
# Compose the two levers that were only ever measured apart: the fc1 fold (-133.5 ms/clip, a hardware-
# context merge) and the one-dispatch fc2 (-515.2 ms/clip, 336 fewer dispatches). They address
# DIFFERENT boundaries -- 48 fc1 ones and 336 fc2 ones -- so the prediction is additive, and the point
# of this harness is to find out whether that holds on device.
#
# Why it could not be run before: the fold makes fc1's bf16-out build the resident, which moves EVERY
# modal GEMM to a bf16 C drain, and the one-dispatch fc2 needs a krtp resident. One artifact has to be
# both. That set now exists (built CPU-only, `dtype_out=bf16 k_loop_rtp=1` at the fold's m=32 tile):
#
#   resident  512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024krtp
#   streams   512x1024x{1024,2048}_32x32x128_8c_modalidbf16outkrtp
#             512x1024x2048_32x32x128_8c_modalglubf16outkrtp        (pw1 with the GLU fold)
#             512x4096x1024_32x32x128_8c_modalidbf16outkrtpapanel1024  (the fc2 collapse itself)
#
# FOLD_GLU is not optional here. Under a bf16-out resident pw1 hands the standalone glu.cc brick a
# bf16 buffer it reads as `const float *`; the fold applies the gate inside pw1's own epilogue and
# deletes that consumer. Same reason the Macaron residual needs resadd's `_bf16b` arm -- with the fc2
# collapse the FFN output IS the modal C drain, where the K-split handed over acc_add's f32.
#
# NOT bit-identical to either arm alone, and not expected to be: on-core accumulation rounds in a
# different order than four separately-rounded partials, and the bf16 drain rounds again. Gate with
# encoder_parity.py against fresh f32 truth, never the 17-clip WER.
#
# THE f1 ARM IS THE KNOWN-BROKEN CONTROL, f2 IS THE CANDIDATE. f1's resident bakes the k-loop trip
# count read at the TOP of the tile, so mixing fc1's K=1024 with the collapse's K=4096 on one
# resident contracts over a stale length -- measured arithmetically dead at rel-L2 0.9942 on all
# frames. f2 selects the `krl` resident, whose core reads rtp[1] AFTER in_a/in_b.acquire(1); that
# peel took 14/14 mixed-K dispatches bit-exact in isolation. Keeping both arms makes this an A/B:
# f1 must reproduce its own failure in the same session or the run says nothing.
#
# Usage: scripts/fold_krtp_ab.sh [reps] [clips]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-4}"
CLIPS="${2:-17}"
OUT="${OUT:-artifacts/fold_krtp}"
BIN=rust/target/release/parakeet_encode_npu
MELS=artifacts/wer_mels
ARMS="f0 f1 f2"
# Per-arm resident epilogue variant. fc1_panel_bf16_stem() appends this to the panel stem, and
# npu.rs derives `krtp` by substring, so `krtpkrl` selects the peeled resident with no code change.
# The streams are NOT suffixed here: under the fold npu.rs hard-codes `bf16out`+`krtp` for them, and
# the peel is core-ELF only, so all three arms drive byte-identical instruction streams.
declare -A EPI_SFX=([f1]=krtp [f2]=krtpkrl)

log(){ echo -e "[foldkrtp] $*"; }
[ -x "$BIN" ] || { log "[ERR] missing $BIN -- cargo build --release -p npu-probes --bin parakeet_encode_npu"; exit 1; }

WA=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build
for s in 512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024krtp \
         512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024krtpkrl \
         512x1024x1024_32x32x128_8c_modalidbf16outkrtp \
         512x1024x2048_32x32x128_8c_modalidbf16outkrtp \
         512x1024x2048_32x32x128_8c_modalglubf16outkrtp \
         512x4096x1024_32x32x128_8c_modalidbf16outkrtpapanel1024; do
  [ -f "$WA/final_$s.xclbin" ] && [ -f "$WA/insts_$s.txt" ] || { log "[ERR] missing artifact $s"; exit 1; }
done
# The resident must resolve from ONE directory. `fc1_panel_bf16_dir` prefers artifacts/parakeet/ln
# and npu.rs::open() uses the same picker, so a copy in BOTH is two hardware contexts for one
# byte-identical xclbin -- the defect that made the fold's first reading a loss.
for s in 512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024krtp \
         512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024krtpkrl; do
  [ -f "artifacts/parakeet/ln/final_$s.xclbin" ] \
    && { log "[ERR] $s is staged in BOTH ln/ and $WA -- that is two hw_contexts"; exit 1; }
done

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

# f32 truth, once -- host only, so it runs outside the lock.
if [ ! -d "$OUT/ref_f32" ] || [ -z "$(ls -A "$OUT/ref_f32" 2>/dev/null)" ]; then
  log "generating f32 reference (host encoder)"
  "$BIN" "$WORK/mel" "$OUT/ref_f32" --cpu > "$OUT/ref_f32.log" 2>&1 \
    || { log "[ERR] f32 reference failed"; tail -5 "$OUT/ref_f32.log"; exit 1; }
fi

run_arm() { # $1 = arm, $2 = rep
  local arm=$1 rep=$2 rc
  local rpt="$OUT/${arm}_rep${rep}.txt"
  local fold=()
  [ "$arm" != "f0" ] && fold=(PARAKEET_FOLD_FC1=1 PARAKEET_FOLD_GLU=1 PARAKEET_MODAL_EPI_SUFFIX="${EPI_SFX[$arm]}")
  timeout -k 10 1800 "$NPU_LOCK_SH" queue -- \
    env NPU_XCLBIN_CACHE_BY_CONTENT=0 NPU_DISPATCH_LOG=1 NPU_XCLBIN_ROOT="$PWD" "${fold[@]}" \
        "$BIN" "$WORK/mel" "$OUT/out_$arm" >"$rpt" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "[ERR] arm=$arm rep=$rep exited $rc"; tail -15 "$rpt"; return 1; }
  grep -q '^mean encode' "$rpt" || { log "[ERR] arm=$arm rep=$rep no timing line"; return 1; }
  log "  arm=$arm rep=$rep  $(grep -m1 'hw_contexts:' "$rpt")  $(grep -m1 'mean encode' "$rpt")"
}

declare -A FAILED=()
for a in $ARMS; do rm -f "$OUT/${a}_rep"*.txt; done
for rep in $(seq 1 "$REPS"); do
  # Rotate by rep so a within-rep ordering effect does not load onto one arm:
  # rep1 f0 f1 f2, rep2 f1 f2 f0, rep3 f2 f0 f1, ...
  arr=($ARMS); n=${#arr[@]}; order=()
  for i in $(seq 0 $((n - 1))); do order+=("${arr[$(( (rep - 1 + i) % n ))]}"); done
  # A failing arm is DATA, not a reason to abandon the run: f1 is the known-broken control and its
  # mixed-K defect can deadlock the array outright ("kernel run did not complete"), which would
  # otherwise take the candidate's measurement down with it. f0 failing IS fatal -- without the
  # baseline there is nothing to pair against.
  for arm in "${order[@]}"; do
    run_arm "$arm" "$rep" || { FAILED[$arm]=$(( ${FAILED[$arm]:-0} + 1 ))
      [ "$arm" = f0 ] && { log "[ERR] baseline arm f0 failed -- nothing to pair against"; exit 1; }
      log "  arm=$arm rep=$rep FAILED (${FAILED[$arm]} so far) -- continuing"; }
  done
done
for a in $ARMS; do
  [ "${FAILED[$a]:-0}" -gt 0 ] && log "[WARN] arm $a failed ${FAILED[$a]}/$REPS reps"
done

# Both analysis scripts price exactly one pair, so each candidate is read against the same f0.
for cand in f1 f2; do
  log "--- wall clock + dispatch ledger (f0 vs $cand) ---"
  python3 scripts/krtp_onedispatch_stats.py "$OUT" "f0:shipped default,$cand:fold+krtp/${EPI_SFX[$cand]}" \
    2>&1 | tee "$OUT/summary_$cand.txt"

  # A deadlocked arm leaves a PARTIAL out dir; parity would traceback on the first missing clip.
  nref=$(ls "$OUT/ref_f32"/*.npy 2>/dev/null | wc -l)
  ncand=$(ls "$OUT/out_$cand"/*.npy 2>/dev/null | wc -l)
  if [ "$ncand" -ne "$nref" ]; then
    log "--- numerical parity ($cand): SKIPPED -- $ncand/$nref clips written (arm never completed a rep) ---"
    continue
  fi
  log "--- numerical parity ($cand vs f0 baseline, against f32 truth) ---"
  python3 scripts/encoder_parity.py "$OUT/ref_f32" "$OUT/out_f0" "$OUT/out_$cand" \
    --json "$OUT/parity_$cand.json" 2>&1 | tee "$OUT/parity_$cand.txt"
done

log "reports under $OUT"
