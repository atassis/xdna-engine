#!/usr/bin/env bash
# Build the lnaffcast 1x route at 4 columns, so its per-dispatch floor can be IDENTIFIED instead of
# extrapolated (task mode-switched-multi-program-xclbin).
#
# WHY A SECOND AXIS. The recorded ~103 us device-serial floor is the intercept of a ROW sweep, and
# rows drive everything at once on this route -- +128 rows is +20 dma tasks, +20 BDs, +8 awaits,
# +2752 insts bytes and +800 KB of traffic, exactly each time. A fit along one collinear axis cannot
# say which term the slope belongs to, and its intercept sits outside the evidence: 128 rows is the
# smallest command the generator will emit ("must be a multiple of the round 128"). The whole_array
# GEMM's floor is trusted more precisely because its series has a near-zero-work arm below the fit.
#
# WHY COLUMNS ARE THE AXIS. At fixed rows, 8 -> 4 columns moves the control stream OPPOSITE to the
# way rows move it, at constant DDR traffic:
#
#         ctrl_writes   dma tasks   BD issues   awaits   DDR bytes   per-core work
#   8c        96            80        1792        32         X            X
#   4c        48            96        2048        32         X           2X
#
# So a (rows x cols) grid separates terms a row sweep sums. It also re-tests the prologue model that
# died on decode_norm_gemv from the other direction, by HALVING the control writes while the dma and
# BD counts rise.
#
# 2 columns is not available: the generator raises IndexError in lnaffcast_fills (A_prods[row]) --
# a toolchain gap, not a silicon limit.
#
# Rows are insts-only, so all four row arms share one array program and one loaded xclbin; the
# cross-xclbin comparison against the 8c set is what the RTP=0 GEMM stream controls.
#
# Device-free. Run:  bash scripts/lnaffcast_cols_build.sh
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
B="$EX/build"
LOG="$WT/artifacts/lnaffcast_cols_build.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { log "FATAL: iron_env did not set MLIR_AIE_INSTANCE"; exit 1; }
# Ask the resolver which instance the lock names; re-deriving the key here is a second copy of
# toolchain_up.sh's hashing rule, and the copy went stale when the key moved to the lock's fields.
pin="$("$WT/scripts/toolchain_up.sh")"
[ "$MLIR_AIE_INSTANCE" = "$pin" ] || {
  log "FATAL: instance $MLIR_AIE_INSTANCE is not the one toolchain.lock names ($pin)"; exit 1; }

COLS="${COLS:-4}"
SCAT="${SCAT:-2}"
BASE="512x1024x1024_32x32x128_${COLS}c_modalidbf16outkrtpkrl"

log "======== lnaffcast 1x cols=$COLS build  $(date -Is) ========"
log "instance: $MLIR_AIE_INSTANCE (pin $pin)"

# kernels_dir resolves against the INSTANCE src tree, so the .cc has to be overlaid there.
bash "$WT/scripts/sync_kernels.sh" >>"$LOG" 2>&1
log "[sync] sandbox refreshed; our kernels compile from route_b_kernels/aie_kernels"

# $1 = lnaffcast_rows (0 = the GEMM stream's default), $2 = contig taps.
build_arm() {
  local rows="$1" ctg="$2" sfx rtp
  rtp="${RTP:-1}"
  if [ "$rtp" = 1 ]; then
    sfx="${BASE}lnaff1024scat${SCAT}1xrtp18$([ "$rows" = 0 ] || echo "r$rows")g4${ctg:+ctg$ctg}"
  else
    sfx="${BASE}lnaff1024scat${SCAT}1x"
  fi
  log "\n[build] $sfx"
  ( cd "$EX" && WA_C_DEPTH=1 WA_AB_DEPTH=2 WA_L3L2_DEPTH=2 \
      make -f Makefile.modal NPU2=1 M=512 K=1024 N=1024 m=32 k=32 n=128 n_aie_cols="$COLS" \
      emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 no_silu=1 dtype_out=bf16 \
      k_loop_rtp=1 k_read_late=1 \
      mode_lnaffcast=1024 rtp_mode_lnaffcast="$rtp" lnaffcast_1x=1 \
      lnaffcast_rows="$rows" \
      lnaffcast_group_rounds=$([ "$rtp" = 1 ] && echo 4 || echo 1) \
      lnaffcast_scatter_c="$SCAT" lnaffcast_contig_taps="$ctg" \
      "build/final_${sfx}.xclbin" ) >>"$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$B/final_$sfx.xclbin" ]; then
    log "  OK   xclbin $(stat -c%s "$B/final_$sfx.xclbin") B   insts $(stat -c%s "$B/insts_$sfx.txt") B"
  else
    log "  FAIL rc=$rc  (see $LOG)"; return 1
  fi
}

for r in 128 256 384 512; do build_arm "$r" "c" || exit 1; done
build_arm 256 ""     || exit 1   # derived C tap against the scatter body -- must FAIL parity
RTP=0 build_arm 0 "" || exit 1

# Every row arm must share ONE array program, or a program transition sits inside the row fit.
# Hashing everything above aie.runtime_sequence says that directly; matching the differing lines
# against a list of op spellings only says it for the spellings the list happens to name.
design_hash() { awk '/aie\.runtime_sequence/{exit} {print}' "$1" | sha256sum | cut -c1-16; }
a="$(design_hash "$B/aie_${BASE}lnaff1024scat${SCAT}1xrtp18r128g4ctgc.mlir")"
for r in 256 384 512; do
  b="$(design_hash "$B/aie_${BASE}lnaff1024scat${SCAT}1xrtp18r${r}g4ctgc.mlir")"
  [ "$a" = "$b" ] && log "\n[check] r$r: design identical ($a) -- runtime sequence only" \
                  || log "\n[check] r$r: DESIGN DIFFERS ($a vs $b) -- not one xclbin"
done

log "\ncensus:"
python3 "$WT/scripts/dispatch_control_census.py" --design \
  "$B:${BASE}lnaff1024scat${SCAT}1xrtp18r128g4ctgc" \
  "$B:${BASE}lnaff1024scat${SCAT}1xrtp18r256g4ctgc" \
  "$B:${BASE}lnaff1024scat${SCAT}1xrtp18r384g4ctgc" \
  "$B:${BASE}lnaff1024scat${SCAT}1xrtp18r512g4ctgc" 2>&1 | tee -a "$LOG"

log "\nlog: $LOG"
