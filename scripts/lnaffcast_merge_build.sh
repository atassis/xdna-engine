#!/usr/bin/env bash
# Build the lnaffcast 1x mode onto the ENCODER'S OWN resident geometry, not the research vehicle's
# (task mode-switched-multi-program-xclbin, its `next:` item 1 -- "the BUILD").
#
# WHY THIS AND NOT ANOTHER MEASUREMENT. The mode is priced: 633.8 us against the shipped standalone
# lnaffcast_512x1024's 860.5, -226.8 us/dispatch, and at 96 dispatches/clip the merge is worth
# -261.8 ms/clip. Every one of those numbers was taken on `512x1024x1024_32x32x128_8c_modalid...krtpkrl`,
# which is NOT an xclbin the encoder loads. What the encoder loads for the FFN is
#
#   512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024   (build_parakeet_modal_kernels.sh, m=32)
#
# and the merge is real only when the mode is a mode of THAT. Four things differ and each could have
# refused: N=4096 not 1024 (the B tensor the 1x route reads x out of), c_panel_width=1024 (the GEMM
# drains panel-major), modalsilu not modalid (the `< 4` branch must still fall through), and no
# krtp/krl (the research vehicle carried both).
#
# WHAT EACH ARM IS FOR.
#   enc-mode   the lnaffcast stream on the encoder resident -- the artifact the merge needs
#   enc-gemm   the fc1 GEMM stream on THAT SAME xclbin -- the control, and the thing that must not
#              regress: a merge that speeds lnaffcast by slowing every fc1 is not a merge
#   shipped    today's resident, rebuilt here so the GEMM-stream MLIR can be diffed against enc-gemm
#              and the added body shown to change the array program only where it must
#
# C DEPTH IS DELIBERATELY UNSET, unlike the research vehicle's WA_C_DEPTH=1. The shipped recipe does
# not set it, so unset is what "the encoder's xclbin" means -- and Makefile.modal's dep_tag only
# tags WA_C_DEPTH=2, so depth 1 and depth 2 collide on one name. Forcing 1 here would build a
# different xclbin under the shipped one's name.
#
# Device-free. Run:  bash scripts/lnaffcast_merge_build.sh
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
B="$EX/build"
LOG="$WT/artifacts/lnaffcast_merge_build.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { log "FATAL: iron_env did not set MLIR_AIE_INSTANCE"; exit 1; }
pin="$(sha256sum "$WT/toolchain.lock" | cut -c1-12)"
[ "$(basename "$MLIR_AIE_INSTANCE")" = "$pin" ] || {
  log "FATAL: instance $(basename "$MLIR_AIE_INSTANCE") is not the committed pin $pin"; exit 1; }

log "======== lnaffcast merge build (encoder resident geometry)  $(date -Is) ========"
log "instance: $MLIR_AIE_INSTANCE (pin $pin)"

bash "$WT/scripts/sync_kernels.sh" "$MLIR_AIE_INSTANCE/src" >>"$LOG" 2>&1
bash "$WT/scripts/sync_kernels.sh" >>"$LOG" 2>&1
cmp -s "$WT/route_b_kernels/aie_kernels/mm_mode_lnaffcast.cc" \
       "$MLIR_AIE_INSTANCE/src/aie_kernels/aie2p/mm_mode_lnaffcast.cc" || {
  log "FATAL: the instance's mm_mode_lnaffcast.cc is not this repo's"; exit 1; }
log "[sync] kernel overlaid into the instance"

# Exactly the shipped resident's parameters (build_parakeet_modal_kernels.sh, "RESIDENT-FFN: fc1
# bf16-out + panel-major drain"). Nothing here is a research setting.
ENC=(NPU2=1 M=512 K=1024 N=4096 m=32 k=32 n=128 n_aie_cols=8
     dtype_in=bf16 dtype_out=bf16 emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1
     c_panel_width=1024)
SHIPPED=512x1024x4096_32x32x128_8c_modalsilubf16outpanel1024

# KRTPKRL=1 builds the mode on the FOLD composition's resident instead of the shipped one. Both
# vehicles are real and they are not interchangeable: `build_parakeet_modal_kernels.sh` ships
# `...panel1024`, but the f2 ledger every merge VALUE is derived from (96 lnaffcast dispatches, 47
# arrivals from the resident, -240.0 ms/clip of boundary tax) ran `...panel1024krtpkrl`. Gating one
# and quoting the other is how a merge lands on an xclbin nobody loads -- the mistake this whole
# script exists to stop, one level up.
#
# krtp puts the k trip count in rtp[1], which is what makes mixed-K unsound; krl is the peel that
# fixes it, and the pair is why that variant carries BOTH tags. The mode itself needs neither -- its
# slab counts are compile-time constants -- so it rides along either way.
if [ "${KRTPKRL:-0}" = 1 ]; then
  ENC+=(k_loop_rtp=1 k_read_late=1)
  SHIPPED=${SHIPPED}krtpkrl
fi
SCAT="${SCAT:-2}"

build() {   # $1 = artifact suffix, rest = extra make vars
  local sfx="$1"; shift
  log "\n[build] $sfx"
  ( cd "$EX" && make -f Makefile.modal "${ENC[@]}" "$@" "build/final_${sfx}.xclbin" ) >>"$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$B/final_$sfx.xclbin" ]; then
    log "  OK   xclbin $(stat -c%s "$B/final_$sfx.xclbin") B   insts $(stat -c%s "$B/insts_$sfx.txt") B"
  else
    log "  FAIL rc=$rc  (see $LOG)"; return 1
  fi
}

# The body knobs -- what goes IN the xclbin. group_rounds and contig_taps are stream knobs and tag
# the artifact name even when rtp is off, so the GEMM arm must leave them at their defaults or it
# asks make for a target whose name says it carries a mode stream it does not.
MODE_KNOBS=(mode_lnaffcast=1024 lnaffcast_1x=1 lnaffcast_scatter_c="$SCAT")

build "$SHIPPED"                               || exit 1
build "${SHIPPED}lnaff1024scat${SCAT}1x"       "${MODE_KNOBS[@]}" || exit 1
build "${SHIPPED}lnaff1024scat${SCAT}1xrtp18r512g4ctgc" "${MODE_KNOBS[@]}" \
      rtp_mode_lnaffcast=1 lnaffcast_rows=512 lnaffcast_group_rounds=4 lnaffcast_contig_taps=c || exit 1
# The both-directions control: the DERIVED C tap against a scatter-2 body, which must FAIL parity.
# Insts-only (taps live in the runtime sequence), so it is a third stream on the same xclbin -- and
# without it a mode arm that passed for the wrong reason would read as a clean run.
build "${SHIPPED}lnaff1024scat${SCAT}1xrtp18r512g4" "${MODE_KNOBS[@]}" \
      rtp_mode_lnaffcast=1 lnaffcast_rows=512 lnaffcast_group_rounds=4 || exit 1

# CHECK 1 -- the two mode arms must be ONE array program, two instruction streams, or the merge has
# not deleted the transition it exists to delete. The authority is the CORE ELF, not the MLIR: the
# xclbin hashes differ on an embedded uuid even when the program is identical, and an MLIR diff
# counts runtime-sequence ops nobody can enumerate correctly by regex (the first version of this
# check missed aiex.dma_start_task/await_task/free_task and reported a false WARN).
a="$B/aie_${SHIPPED}lnaff1024scat${SCAT}1x.mlir"
b="$B/aie_${SHIPPED}lnaff1024scat${SCAT}1xrtp18r512g4ctgc.mlir"
same=1
for c in 0_2 3_3 7_5; do
  cmp -s "${a%.mlir}.mlir.prj/elfs_main_core_$c/elfs_main_core_$c.elf" \
         "${b%.mlir}.mlir.prj/elfs_main_core_$c/elfs_main_core_$c.elf" || same=0
done
[ "$same" = 1 ] && log "\n[check] mode vs GEMM stream: core ELFs IDENTICAL -- one xclbin, two streams" \
                || log "\n[check] mode vs GEMM stream: core ELFs DIFFER -- NOT one xclbin"

# CHECK 2 -- what the merge charges today's fc1. Two things must hold: the GEMM's instruction stream
# is unchanged (else every fc1 dispatch runs different BDs), and the L1 allocation is unchanged
# (else the added body displaced the GEMM's buffers, which is what the 4768-byte ceiling was about).
s="$B/aie_${SHIPPED}.mlir"
cmp -s "$B/insts_${SHIPPED}.txt" "$B/insts_${SHIPPED}lnaff1024scat${SCAT}1x.txt" \
  && log "[check] fc1 GEMM instruction stream: BYTE-IDENTICAL to the shipped resident" \
  || log "[check] fc1 GEMM instruction stream: DIFFERS from the shipped resident"
diff <(grep "aie.buffer" "$s") <(grep "aie.buffer" "$a") >/dev/null \
  && log "[check] L1 allocation: IDENTICAL (same addresses, same sizes -- the mode declares nothing)" \
  || log "[check] L1 allocation: MOVED -- the added body displaced the GEMM's buffers"
log "[check] core body delta: $(diff "$s" "$a" | grep '^[<>]' | grep -c 'scf.if') scf.if (the rtp branch, one per core)"

log "\nlog: $LOG"
