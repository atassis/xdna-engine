#!/usr/bin/env bash
# PRICE THE lnaffcast MODE ON THE krtpkrl COMPOSITION -- task mode-switched-multi-program-xclbin,
# `next:` item 2.
#
# WHY THIS IS A DIFFERENT RUN FROM lnmode_fold_bracket_ab.sh. The mode's headline ledger -- 143
# transitions/clip deleted (`lnaffcast -> panel-fc1` x96 plus `panel-fc1 -> lnaffcast` x47) and
# -240.0 ms/clip -- belongs to the f2 composition of fold_krtp_ab.sh: FOLD_FC1 + FOLD_GLU + the
# krtpkrl resident that carries the one-dispatch fc2. The fold-only bracket measures a DIFFERENT
# graph and reads 96 + 24 = 120, so 143 has never been observed end to end. Which composition a
# merge is worth anything in is the open question on this task; this run answers it for the one the
# number was quoted from.
#
# AND THIS COMPOSITION CAN BE GATED FOR CORRECTNESS, which the fold-only bracket cannot. `fold_fc1`
# alone leaves a bf16 consumer gap and its encoder output is wrong by design; FOLD_GLU closes it,
# which is why fold_krtp_ab.sh gates f2 with encoder_parity.py against f32 truth and reads PASS. So
# this script brackets TIME + TRANSITIONS over N reps and then runs ONE 17-clip parity pass on the
# mode arm -- the same shape fold_krtp uses, against the same f32 reference.
#
# ARMS -- both hold FOLD_FC1=1 FOLD_GLU=1 MODAL_EPI_SUFFIX=krtpkrl; they differ only in the mode:
#   k0   composition            expect 335 transitions (fold_krtp f2, 2026-08-23)
#   k0m  composition + LN_MODE  the ledger predicts 335 - 143 = 192
# The cache control that the fold bracket carried (NPU_XCLBIN_CACHE_BY_CONTENT) is not repeated: it
# PASSED there in two independent windows -- the fold folds unaided since a59b8b5 -- so a third
# reading of a settled no-op would buy reps that this run spends on the contrast instead. The
# both-directories guard below is the check that matters for THIS composition.
#
# ORDERING. Leading arm alternates by rep, so a within-rep position effect cannot load onto one arm.
# Pairing is per (rep, clip): clip 1 carries the cold weight-BO load in every arm, and pairing keeps
# that cost on both sides of the difference rather than averaging it into one arm's mean.
#
# Usage: scripts/lnmode_krtpkrl_bracket_ab.sh [reps] [clips]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/_npu_services.sh" || exit 1
cd "$HERE/.." || exit 1

REPS="${1:-11}"
CLIPS="${2:-3}"
PARITY_CLIPS="${PARITY_CLIPS:-17}"
OUT="${OUT:-artifacts/lnmode_krtpkrl_bracket}"
BIN=rust/target/release/parakeet_encode_npu
MELS=artifacts/wer_mels
ARMS="k0 k0m"

log(){ echo -e "[lnkr] $*"; }
[ -x "$BIN" ] || { log "[ERR] missing $BIN -- cargo build --release -p npu-probes --bin parakeet_encode_npu"; exit 1; }

WA=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build
PANEL=512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024krtpkrl
MODEPANEL="${PANEL}lnaff1024scat21x"
for s in "$PANEL" "$MODEPANEL" "${MODEPANEL}rtp18r512g4ctgc" \
         512x1024x1024_32x32x128_8c_modalidbf16outkrtp \
         512x1024x2048_32x32x128_8c_modalidbf16outkrtp \
         512x1024x2048_32x32x128_8c_modalglubf16outkrtp \
         512x4096x1024_32x32x128_8c_modalidbf16outkrtpapanel1024; do
  [ -f "$WA/insts_$s.txt" ] || { log "[ERR] missing insts_$s.txt"; exit 1; }
done
# The mode stream is INSTS-ONLY by design -- it rides the panel's xclbin -- so only the panels need one.
for s in "$PANEL" "$MODEPANEL"; do
  [ -f "$WA/final_$s.xclbin" ] || { log "[ERR] missing final_$s.xclbin"; exit 1; }
done
# One resident must resolve from ONE directory: `load_kernel` keys its context cache on the PATH, so
# byte-identical copies at two paths are two hw_contexts and a program transition per crossing --
# the defect that made the fold's first reading a loss.
for s in "$PANEL" "$MODEPANEL"; do
  [ -f "artifacts/parakeet/ln/final_$s.xclbin" ] \
    && { log "[ERR] $s is staged in BOTH artifacts/parakeet/ln/ and $WA -- that is two hw_contexts"; exit 1; }
done

NPU_LOCK_SH="${NPU_LOCK_SH:-}"
if [ -z "$NPU_LOCK_SH" ] || [ ! -x "$NPU_LOCK_SH" ]; then
  log "[ERR] set NPU_LOCK_SH to the device serialisation wrapper (mode: queue -- <cmd>)"; exit 1
fi

WORK="$(mktemp -d)"
mkdir -p "$WORK/mel" "$WORK/pmel" "$OUT"
i=0; j=0
for f in $(ls "$MELS"/*.npy | sort); do
  [ "$j" -lt "$PARITY_CLIPS" ] && { ln -sf "$(readlink -f "$f")" "$WORK/pmel/$(basename "$f")"; j=$((j + 1)); }
  [ "$i" -lt "$CLIPS" ] && { ln -sf "$(readlink -f "$f")" "$WORK/mel/$(basename "$f")"; i=$((i + 1)); }
done
log "clips/rep = $i, reps = $REPS, arms = $ARMS, parity clips = $j"
trap 'rm -rf "$WORK"; npu_svc_start' EXIT

npu_svc_stop || exit 1
npu_svc_require_device_free || exit 1

comp_env() { # $1 = arm -> the env that selects the composition, mode last
  local mode=0; [ "$1" = k0m ] && mode=1
  echo "PARAKEET_FOLD_FC1=1 PARAKEET_FOLD_GLU=1 PARAKEET_MODAL_EPI_SUFFIX=krtpkrl PARAKEET_LN_MODE=$mode"
}

run_arm() { # $1 = arm, $2 = rep
  local arm=$1 rep=$2 rc
  local rpt="$OUT/${arm}_rep${rep}.txt"
  timeout -k 10 900 "$NPU_LOCK_SH" queue -- \
    env $(comp_env "$arm") NPU_XCLBIN_CACHE_BY_CONTENT=0 NPU_DISPATCH_LOG=1 NPU_XCLBIN_ROOT="$PWD" \
        "$BIN" "$WORK/mel" "$WORK/out_$arm" >"$rpt" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "[ERR] arm=$arm rep=$rep exited $rc"; tail -8 "$rpt"; return 1; }
  grep -q '^mean encode' "$rpt" || { log "[ERR] arm=$arm rep=$rep no timing line"; return 1; }
  log "  arm=$arm rep=$rep  $(grep -m1 'hw_contexts:' "$rpt")  $(grep -m1 -oE 'transitions [0-9]+' "$rpt")  $(grep -m1 'mean encode' "$rpt")"
}

for a in $ARMS; do rm -f "$OUT/${a}_rep"*.txt; done
for rep in $(seq 1 "$REPS"); do
  order="k0 k0m"; [ $(( rep % 2 )) -eq 0 ] && order="k0m k0"
  log "rep $rep order: $order"
  for arm in $order; do run_arm "$arm" "$rep" || exit 1; done
  python3 scripts/lnmode_fold_bracket_stats.py "$OUT" krtpkrl >"$OUT/summary.txt" 2>&1
done

log "--- results ---"
python3 scripts/lnmode_fold_bracket_stats.py "$OUT" krtpkrl 2>&1 | tee "$OUT/summary.txt"

# CORRECTNESS, once, on the full clip set. The f32 truth is host-only, so it runs outside the lock;
# fold_krtp's reference is reused when it is already there and covers the same clips.
REF="artifacts/fold_krtp/ref_f32"
if [ "$(ls "$REF"/*.npy 2>/dev/null | wc -l)" -ne "$j" ]; then
  REF="$OUT/ref_f32"
  if [ "$(ls "$REF"/*.npy 2>/dev/null | wc -l)" -ne "$j" ]; then
    log "generating f32 reference (host encoder, $j clips)"
    "$BIN" "$WORK/pmel" "$REF" --cpu >"$OUT/ref_f32.log" 2>&1 \
      || { log "[ERR] f32 reference failed"; tail -5 "$OUT/ref_f32.log"; exit 1; }
  fi
fi
log "--- parity pass ($PARITY_CLIPS clips): k0 baseline, k0m candidate, truth $REF ---"
for arm in k0 k0m; do
  timeout -k 10 1800 "$NPU_LOCK_SH" queue -- \
    env $(comp_env "$arm") NPU_XCLBIN_CACHE_BY_CONTENT=0 NPU_XCLBIN_ROOT="$PWD" \
        "$BIN" "$WORK/pmel" "$OUT/parity_out_$arm" >"$OUT/parity_run_$arm.txt" 2>&1 \
    || { log "[ERR] parity run $arm failed"; tail -8 "$OUT/parity_run_$arm.txt"; exit 1; }
done
python3 scripts/encoder_parity.py "$REF" "$OUT/parity_out_k0" "$OUT/parity_out_k0m" \
  --json "$OUT/parity_k0m.json" 2>&1 | tee "$OUT/parity_k0m.txt"

log "reports under $OUT"
