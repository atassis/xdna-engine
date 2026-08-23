#!/usr/bin/env bash
# Build the lnaffcast mode's 1x route -- x on B, gb on A -- so the 8 columns of an array row stop
# computing the same output rows (task mode-switched-multi-program-xclbin, its `next:` item 1).
#
# WHY. lnaffcast-mode-duplicates-output-across-the-a-broadcast measured that A_L2L1_0 lists EIGHT
# consumer cores, so x on A reaches all 8 columns of a row and they compute bit-identical output:
# 1x on the read, 8x on the WRITE, 4 useful C tiles per round of 32. With the C un-permute settled
# (the-scatter-was-the-datapath-not-the-store), that duplication is the largest remaining
# multiplier between this vehicle and a verdict on the -238.0 ms/clip merge.
#
# THE SWAP. Which operand duplicates is a property of WHICH FIFO it rides, not of the mode: A is
# per-array-row and broadcast across columns, B is per-column and broadcast down rows. gb is a
# weight and WANTS a broadcast; x does not. So x goes on B and gb on A, and the row axis is broken
# by a skip -- a column's B stream carries n_aie_rows C tiles' worth and each core applies only its
# own share. The skip belongs on B for a size reason: a B tile is 4 A tiles, so it is 8 acquire
# pairs per C tile here against 64 for the same skip on A.
#
# Per round the traffic goes 384 KB / 16 rows -> 800 KB / 128 rows, i.e. 24 -> 6.25 KB per output
# row. The A tap carries gb (one offset, every round), B carries x, and C drains 32 DISTINCT
# offsets per 4 rounds where the 8x route drains 128 BDs onto 16.
#
# ARMS. lnaffcast_1x changes the array program (a different apply kernel and the skip), so it is
# its own xclbin and the comparison against the 8x scat2 arm is CROSS-xclbin -- which is what the
# RTP=0 GEMM stream is for (gemm-stream-controls-the-cross-xclbin-comparison). lnaffcast_rows is
# insts-only, so 256 and 512 share one xclbin:
#   <1x>r256ctgc  the candidate at the 8x route's own row count -- must PASS parity
#   <1x>r256      derived C tap against the scatter body -- must FAIL, the both-directions gate
#   <1x>r512ctgc  the same route at what the B tensor holds -- must PASS
#   <1x>          the GEMM stream on this xclbin -- the cross-xclbin control
#
# Device-free. Run:  bash scripts/lnaffcast_1x_build.sh
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
B="$EX/build"
LOG="$WT/artifacts/lnaffcast_1x_build.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { log "FATAL: iron_env did not set MLIR_AIE_INSTANCE"; exit 1; }
pin="$(sha256sum "$WT/toolchain.lock" | cut -c1-12)"
[ "$(basename "$MLIR_AIE_INSTANCE")" = "$pin" ] || {
  log "FATAL: instance $(basename "$MLIR_AIE_INSTANCE") is not the committed pin $pin"; exit 1; }

log "======== lnaffcast 1x-route build  $(date -Is) ========"
log "instance: $MLIR_AIE_INSTANCE (pin $pin)"

# kernels_dir resolves against the INSTANCE src tree, not this repo, so the .cc has to be
# overlaid there or the build compiles the instance's own older copy.
bash "$WT/scripts/sync_kernels.sh" "$MLIR_AIE_INSTANCE/src" >>"$LOG" 2>&1
bash "$WT/scripts/sync_kernels.sh" >>"$LOG" 2>&1
cmp -s "$WT/route_b_kernels/aie_kernels/mm_mode_lnaffcast.cc" \
       "$MLIR_AIE_INSTANCE/src/aie_kernels/aie2p/mm_mode_lnaffcast.cc" || {
  log "FATAL: the instance's mm_mode_lnaffcast.cc is not this repo's"; exit 1; }
log "[sync] kernel overlaid into the instance"

BASE=512x1024x1024_32x32x128_8c_modalidbf16outkrtpkrl
SCAT="${SCAT:-2}"      # the settled C un-permute: arithmetic at 16 lanes, only the store split

# $1 = lnaffcast_rows (0 = the stream's own default), $2 = contig taps, RTP=0 -> the GEMM stream.
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
      make -f Makefile.modal NPU2=1 M=512 K=1024 N=1024 m=32 k=32 n=128 n_aie_cols=8 \
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

build_arm 256 "c"       || exit 1
build_arm 256 ""        || exit 1
build_arm 512 "c"       || exit 1
RTP=0 build_arm 0 ""    || exit 1

# The three mode arms must share an array program -- that is what lets all of them run on one
# loaded xclbin with no program transition inside the comparison. rows and the tap re-stride both
# live in the runtime sequence, so every differing MLIR line must be a BD.
a="$B/aie_${BASE}lnaff1024scat${SCAT}1xrtp18r256g4ctgc.mlir"
for b in "$B/aie_${BASE}lnaff1024scat${SCAT}1xrtp18r256g4.mlir" \
         "$B/aie_${BASE}lnaff1024scat${SCAT}1xrtp18r512g4ctgc.mlir"; do
  n=$(diff "$a" "$b" | grep -c '^[<>]')
  nbd=$(diff "$a" "$b" | grep '^[<>]' | grep -cE 'aie\.dma_bd|aiex\.npu\.')
  log "\n[check] $(basename "$b" .mlir): $n differing lines, $nbd of them BD/npu ops"
  [ "$n" = "$nbd" ] && log "    OK -- runtime sequence only" || log "    WARN -- something else moved"
done

log "\nlog: $LOG"
