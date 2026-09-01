#!/usr/bin/env bash
# Build the lnaffcast mode with the C un-permute moved off the shim tap and onto the core
# (task mode-switched-multi-program-xclbin -- the gather-vs-tap arm its `next:` prescribed).
#
# WHY. lnaffcast-mode-contiguous-taps-recover-61-percent measured what the taps cost: giving all
# three the contiguous strides for their own sizes takes the mode 2428 -> 949 us, and the C tap
# alone is 93.2% of that. It is a PRICE, not a saving -- the walk it deletes IS the un-permute, so
# a contiguous-tap arm returns the permuted output. This build is the other side of that choice:
# the core writes C scattered (LNA_SCATTER_C=1) so the contiguous tap becomes the CORRECT one.
#
# The core-side form is a SCATTER, not the 993-run gather the A-side map was priced at. C_L2L3's
# innermost digit is (t, 1), so output element q -> L1 slot C_L2L3[q] lands in 509 runs of 8 with
# every base t-aligned -- 512 16 B stores per C tile against 256 32 B ones, no scalar addressing.
#
# TWO ARMS ON ONE XCLBIN. lnaffcast_scatter_c changes the core ELF, so it tags the body and needs
# its own xclbin; lnaffcast_contig_taps is insts-only, so both arms below run on the FIRST one's
# loaded xclbin and no program transition sits inside the comparison:
#   <scat>        scatter body + the DERIVED C tap -- WRONG output, holds core work, isolates the tap
#   <scat>ctgc    scatter body + the CONTIGUOUS C tap -- the candidate, must PASS parity
#
# The banked baseline artifacts are NOT rebuilt: at lnaffcast_scatter_c=0 the EPI_DEFINES string is
# unchanged, so .epi_defines.stamp stays clean and the default body is the same preprocessor output.
#
# Device-free. Run:  bash scripts/lnaffcast_scatter_build.sh   (SCAT=2 for the split-store form)
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
B="$EX/build"
LOG="$WT/artifacts/lnaffcast_scatter_build.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { log "FATAL: iron_env did not set MLIR_AIE_INSTANCE"; exit 1; }
# Ask the resolver which instance the lock names; re-deriving the key here is a second copy of
# toolchain_up.sh's hashing rule, and the copy went stale when the key moved to the lock's fields.
pin="$("$WT/scripts/toolchain_up.sh")"
[ "$MLIR_AIE_INSTANCE" = "$pin" ] || {
  log "FATAL: instance $MLIR_AIE_INSTANCE is not the one toolchain.lock names ($pin)"; exit 1; }

log "======== lnaffcast scatter-C build  $(date -Is) ========"
log "instance: $MLIR_AIE_INSTANCE (pin $pin)"

# kernels_dir resolves against the INSTANCE src tree, not this repo, so the .cc has to be
# overlaid there or the build compiles the instance's own older copy.
bash "$WT/scripts/sync_kernels.sh" >>"$LOG" 2>&1
log "[sync] sandbox refreshed; our kernels compile from route_b_kernels/aie_kernels"

BASE=512x1024x1024_32x32x128_8c_modalidbf16outkrtpkrl
# 1 = the whole write pass narrowed to t lanes; 2 = arithmetic left at 16 lanes, only the store
# split. Both make the contiguous C tap the correct one; they differ in what the scatter costs.
SCAT="${SCAT:-1}"
TAG="scat$([ "$SCAT" = 1 ] && echo "" || echo "$SCAT")"
# $1 = ctg taps (may be empty); RTP=0 builds the GEMM stream on this same xclbin instead of the
# mode stream. The GEMM arm is the CROSS-XCLBIN control: the scatter body is a different array
# program from the banked baseline, so a bare mode-vs-mode comparison spans two loaded xclbins.
# The GEMM stream is untouched by lnaffcast_scatter_c, so its median matching the baseline
# xclbin's is what says the two vehicles are comparable.
build_arm() {
  local ctg="$1" sfx rtp
  rtp="${RTP:-1}"
  if [ "$rtp" = 1 ]; then sfx="${BASE}lnaff1024${TAG}rtp18g4${ctg:+ctg$ctg}"
  else                    sfx="${BASE}lnaff1024${TAG}"; fi
  log "\n[build] $sfx"
  ( cd "$EX" && WA_C_DEPTH=1 WA_AB_DEPTH=2 WA_L3L2_DEPTH=2 \
      make -f Makefile.modal NPU2=1 M=512 K=1024 N=1024 m=32 k=32 n=128 n_aie_cols=8 \
      emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 no_silu=1 dtype_out=bf16 \
      k_loop_rtp=1 k_read_late=1 \
      mode_lnaffcast=1024 rtp_mode_lnaffcast="$rtp" \
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

build_arm ""            || exit 1
build_arm "c"           || exit 1
RTP=0 build_arm ""      || exit 1

# The two arms must differ in the C BD's STRIDES and in nothing else, and must share an array
# program -- that is what lets both streams run on one loaded xclbin.
a="$B/aie_${BASE}lnaff1024${TAG}rtp18g4.mlir"
b="$B/aie_${BASE}lnaff1024${TAG}rtp18g4ctgc.mlir"
log "\n[check] differing MLIR lines between the two arms:"
diff "$a" "$b" | grep '^[<>]' | sed 's/^/    /' | head -8 >>"$LOG"
n=$(diff "$a" "$b" | grep -c '^[<>]')
nbd=$(diff "$a" "$b" | grep '^[<>]' | grep -c 'aie\.dma_bd')
log "    $n differing lines, $nbd of them aie.dma_bd"
[ "$n" = "$nbd" ] && log "    OK -- BD strides only" || log "    WARN -- something other than a BD moved"

log "\nlog: $LOG"
