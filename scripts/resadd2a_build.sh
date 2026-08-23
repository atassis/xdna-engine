#!/usr/bin/env bash
# Build the both-on-A resadd mode -- `out = a + scale*b` carried as a MODE of the f32-out modal
# GEMM, both operands on the A fifo (task mode-switched-multi-program-xclbin, its `next:` item a).
#
# WHY BOTH ON A. lnaffcast resolved its broadcast by having an operand that WANTED it: gb is a
# weight, so A_L2L1's per-array-row broadcast is what a weight needs. resadd has no weight -- a
# and b are both activations -- so the broadcast is paid in acquires instead: each core takes
# every column's share off the row's stream and applies its own, 2 * n/k tiles per C tile per
# column = 64 pairs (two-activation-mode-broadcast-acquire-cost). The split route is 40 acquires
# but owes a B->A reconciliation across two maps AND two tile extents; both-on-A gets the
# operand pairing free because one map applies to both.
#
# NO TAP DERIVATION. A_L2L1 and C_L2L3 compose into a closed form the core walks --
# (p/(r*k))*(n*r) + (p%(r*k)) + j*(r*k), verified over 112 j-maps -- so both taps here are plain
# row-major block reads and the C drain is the GEMM's own.
#
# ARMS. rows and scale are insts-only, so all of these share ONE xclbin and the comparison
# carries no program transition:
#   <r2a>rtp5      the mode at what the A tensor holds as two 1024-wide operands
#   <r2a>          the GEMM stream on the same xclbin -- the cross-arm control
#
# Device-free. Run:  bash scripts/resadd2a_build.sh
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
B="$EX/build"
LOG="$WT/artifacts/resadd2a_build.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { log "FATAL: iron_env did not set MLIR_AIE_INSTANCE"; exit 1; }
pin="$(sha256sum "$WT/toolchain.lock" | cut -c1-12)"
[ "$(basename "$MLIR_AIE_INSTANCE")" = "$pin" ] || {
  log "FATAL: instance $(basename "$MLIR_AIE_INSTANCE") is not the committed pin $pin"; exit 1; }

log "======== resadd2a build  $(date -Is) ========"
log "instance: $MLIR_AIE_INSTANCE (pin $pin)"

# kernels_dir resolves against the INSTANCE src tree, so the .cc has to be overlaid there or the
# build compiles the instance's own copy.
bash "$WT/scripts/sync_kernels.sh" "$MLIR_AIE_INSTANCE/src" >>"$LOG" 2>&1
bash "$WT/scripts/sync_kernels.sh" >>"$LOG" 2>&1
cmp -s "$WT/route_b_kernels/aie_kernels/mm_mode_resadd2a.cc" \
       "$MLIR_AIE_INSTANCE/src/aie_kernels/aie2p/mm_mode_resadd2a.cc" || {
  log "FATAL: the instance's mm_mode_resadd2a.cc is not this repo's"; exit 1; }
log "[sync] kernel overlaid into the instance"

BASE=512x1024x1024_64x32x128_8c_modalidkrtpkrl

# $1 = rtp_mode_resadd2a
build_arm() {
  local rtp="$1" sfx
  sfx="${BASE}r2a$([ "$rtp" = 1 ] && echo rtp5)"
  log "\n[build] $sfx"
  ( cd "$EX" && WA_C_DEPTH=1 WA_AB_DEPTH=2 WA_L3L2_DEPTH=2 \
      make -f Makefile.modal NPU2=1 M=512 K=1024 N=1024 m=64 k=32 n=128 n_aie_cols=8 \
      emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 no_silu=1 dtype_out=f32 \
      k_loop_rtp=1 k_read_late=1 \
      mode_resadd2a=1 rtp_mode_resadd2a="$rtp" \
      "build/final_${sfx}.xclbin" ) >>"$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$B/final_$sfx.xclbin" ]; then
    log "  OK   xclbin $(stat -c%s "$B/final_$sfx.xclbin") B   insts $(stat -c%s "$B/insts_$sfx.txt") B"
  else
    log "  FAIL rc=$rc  (see $LOG)"; return 1
  fi
}

build_arm 1 || exit 1
build_arm 0 || exit 1

# The two streams must share an array program, or the comparison crosses a program boundary and
# stops being about the mode. Every differing MLIR line has to be a BD.
a="$B/aie_${BASE}r2artp5.mlir"
b="$B/aie_${BASE}r2a.mlir"
n=$(diff "$a" "$b" | grep -c '^[<>]')
nbd=$(diff "$a" "$b" | grep '^[<>]' | grep -cE 'aie\.dma_bd|aiex\.|arith\.constant|issue_token|^[<>] *\}')
log "\n[check] streams differ in $n lines, $nbd of them BD/sequence ops"
[ "$n" = "$nbd" ] && log "    OK -- runtime sequence only" || log "    WARN -- something else moved"

log "\nlog: $LOG"
