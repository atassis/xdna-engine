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
# ARMS. rows is insts-only. SCALE is insts-only only under resadd2a_scale_rtp=1, and the core ELF
# is what says which -- an xclbin carries a per-build UUID, so it is never byte-identical across
# two builds and cannot answer this. The arms:
#   <r2a>rtp5      the mode stream at scale 1.0
#   <r2a>rtp5s05   the mode stream at scale 0.5
#   <r2a>          the GEMM stream -- the cross-arm control
# and the check below is whether the three share one core ELF.
#
# Device-free. Run:  bash scripts/resadd2a_build.sh          # scale baked
#                    R2A_SCALE_RTP=1 bash scripts/resadd2a_build.sh
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
B="$EX/build"
LOG="$WT/artifacts/resadd2a_build.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { log "FATAL: iron_env did not set MLIR_AIE_INSTANCE"; exit 1; }
# Ask the resolver which instance the lock names; re-deriving the key here is a second copy of
# toolchain_up.sh's hashing rule, and the copy went stale when the key moved to the lock's fields.
pin="$("$WT/scripts/toolchain_up.sh")"
[ "$MLIR_AIE_INSTANCE" = "$pin" ] || {
  log "FATAL: instance $MLIR_AIE_INSTANCE is not the one toolchain.lock names ($pin)"; exit 1; }

log "======== resadd2a build  $(date -Is) ========"
log "instance: $MLIR_AIE_INSTANCE (pin $pin)"

# kernels_dir resolves against the INSTANCE src tree, so the .cc has to be overlaid there or the
# build compiles the instance's own copy.
bash "$WT/scripts/sync_kernels.sh" >>"$LOG" 2>&1
log "[sync] sandbox refreshed; our kernels compile from route_b_kernels/aie_kernels"

BASE=512x1024x1024_64x32x128_8c_modalidkrtpkrl
# The arm on record: one pass per (a, b) pair, the vfile detour off, scale*b on the bf16 datapath.
SCRTP="${R2A_SCALE_RTP:-0}"
BODY="r2abf16mnovffus$([ "$SCRTP" = 1 ] && echo scrtp)"

elf_of() {  # core (0,2)'s ELF -- one core is enough, the array is homogeneous
  echo "$B/aie_$1.mlir.prj/elfs_main_core_0_2/elfs_main_core_0_2.elf"
}

# $1 = rtp_mode_resadd2a   $2 = scale
arm_sfx() {
  echo "${BASE}${BODY}$([ "$1" = 1 ] && echo rtp5)$([ "$2" != 1.0 ] && echo "s${2/./}")"
}

build_arm() {
  local rtp="$1" scale="$2" sfx
  sfx=$(arm_sfx "$rtp" "$scale")
  log "\n[build] $sfx"
  ( cd "$EX" && WA_C_DEPTH=1 WA_AB_DEPTH=2 WA_L3L2_DEPTH=2 \
      make -f Makefile.modal NPU2=1 M=512 K=1024 N=1024 m=64 k=32 n=128 n_aie_cols=8 \
      emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 no_silu=1 dtype_out=f32 \
      k_loop_rtp=1 k_read_late=1 \
      mode_resadd2a=1 rtp_mode_resadd2a="$rtp" \
      resadd2a_fused=1 resadd2a_vfile=0 resadd2a_bf16mul=1 \
      resadd2a_scale="$scale" resadd2a_scale_rtp="$SCRTP" \
      "build/final_${sfx}.xclbin" ) >>"$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$B/final_$sfx.xclbin" ]; then
    log "  OK   xclbin $(stat -c%s "$B/final_$sfx.xclbin") B   insts $(stat -c%s "$B/insts_$sfx.txt") B" \
        "  elf $(sha256sum "$(elf_of "$sfx")" | cut -c1-16)"
  else
    log "  FAIL rc=$rc  (see $LOG)"; return 1
  fi
}

gemm_sfx=$(arm_sfx 0 1.0); s100_sfx=$(arm_sfx 1 1.0); s050_sfx=$(arm_sfx 1 0.5)
build_arm 0 1.0 || exit 1
build_arm 1 1.0 || exit 1
build_arm 1 0.5 || exit 1

# THE CHECK. Under resadd2a_scale_rtp=1 the scale is an rtp[2] read, so all three arms must link
# ONE core ELF and the two scales are two instruction streams on one program. At 0 the s05 arm
# carries its own immediate and its ELF must differ -- that is what makes the flag worth having.
log "\n[check] core ELF identity (resadd2a_scale_rtp=$SCRTP)"
h_gemm=$(sha256sum "$(elf_of "$gemm_sfx")" | cut -c1-16)
h_100=$(sha256sum "$(elf_of "$s100_sfx")" | cut -c1-16)
h_050=$(sha256sum "$(elf_of "$s050_sfx")" | cut -c1-16)
# Empty hashes compare EQUAL, so a missing ELF would read as a pass.
[ -n "$h_gemm" ] && [ -n "$h_100" ] && [ -n "$h_050" ] \
  || { log "    FAIL -- a core ELF is missing; the check has nothing to compare"; exit 1; }
log "    gemm $h_gemm   s1.0 $h_100   s0.5 $h_050"
if [ "$SCRTP" = 1 ]; then
  [ "$h_gemm" = "$h_100" ] && [ "$h_100" = "$h_050" ] \
    && log "    OK -- one program serves both scales" \
    || { log "    FAIL -- scale still reaches the ELF"; exit 1; }
else
  [ "$h_100" != "$h_050" ] \
    && log "    OK -- baked: each scale is its own program (the control)" \
    || log "    WARN -- baked arm shares an ELF across scales, which it should not"
fi

# The mode stream and the GEMM stream must share an array program, or the comparison crosses a
# program boundary and stops being about the mode. Every differing MLIR line has to be a BD.
n=$(diff "$B/aie_$s100_sfx.mlir" "$B/aie_$gemm_sfx.mlir" | grep -c '^[<>]')
nbd=$(diff "$B/aie_$s100_sfx.mlir" "$B/aie_$gemm_sfx.mlir" | grep '^[<>]' \
      | grep -cE 'aie\.dma_bd|aiex\.|arith\.constant|issue_token|^[<>] *\}')
log "\n[check] streams differ in $n lines, $nbd of them BD/sequence ops"
[ "$n" = "$nbd" ] && log "    OK -- runtime sequence only" || log "    WARN -- something else moved"

log "\nsuffixes: gemm=$gemm_sfx  s1.0=$s100_sfx  s0.5=$s050_sfx"
log "log: $LOG"
