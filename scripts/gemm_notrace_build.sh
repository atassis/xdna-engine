#!/usr/bin/env bash
# Build NO-TRACE whole_array GEMM arms as the control for the command-time accounting
# (task gemm-offcore-residue-occupancy, item 3).
#
# WHY: the accounting run compares cols=1/4/8, but only cols=8 was built without trace -- it is the
# one width that cannot route a trace flow. The 1c and 4c designs carry an appended trace BO (64 KB
# and 1 MB), and 1 MB at the ~6 GB/s this box moves intermediates is ~167 us, the same order as the
# width-independent floor the comparison is trying to extract. So the width series is confounded
# with trace cost until the same widths are built trace-free.
#
# NAME COLLISION, and why this script moves files: target_suffix (Makefile.modal:132) has no
# trace-on/off component, so a PROFILE-less build writes the SAME artifact names as the traced build
# and would silently replace banked artifacts eight passes of measurements refer to. That is the
# exact failure the sweep script's own comments record for the depth tags. So each arm is built
# with the traced artifacts moved aside, its products renamed to a `nt` suffix, and the traced
# artifacts put back.
#
# Device-free. Run:  bash scripts/gemm_notrace_build.sh [k:n:cols ...]
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
B="$EX/build"
ARMS="${1:-32:128:1 32:128:4}"
LOG="$WT/artifacts/gemm_notrace_build.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { log "FATAL: iron_env did not set MLIR_AIE_INSTANCE"; exit 1; }

log "======== whole_array NO-TRACE control build  $(date -Is) ========"
log "instance: $MLIR_AIE_INSTANCE"
log ""
log "  cols |   k |   n | build | operands | xclbin bytes | suffix"
log "  -----+-----+-----+-------+----------+--------------+-------"

# Move one arm's four artifacts between name <-> name.stash, so a build cannot land on a banked name.
stash(){ for f in "final_$1.xclbin" "insts_$1.txt" "aie_$1.mlir" "aie_$1.mlir.prj"; do
           [ -e "$B/$f" ] && mv "$B/$f" "$B/$f.stash"; done; return 0; }
unstash(){ for f in "final_$1.xclbin" "insts_$1.txt" "aie_$1.mlir" "aie_$1.mlir.prj"; do
             [ -e "$B/$f.stash" ] && mv "$B/$f.stash" "$B/$f"; done; return 0; }

for arm in $ARMS; do
  IFS=: read -r k n c <<< "$arm"
  sfx="512x1024x1024_64x${k}x${n}_${c}c_modalid"
  nt="${sfx}nt"
  stash "$sfx"
  # No PROFILE=trace and no wa_trace_* -- that is the whole difference from gemm_k_sweep_build.sh.
  # WA_L3L2_DEPTH must be passed even at its default 2: l3dep_tag (Makefile.modal:115) is
  # `$(if $(filter 2,...),,l3$(...))`, so an UNSET value fails the filter and appends a bare `l3`,
  # building `...modalidl3` and failing the explicit target. This is the same empty-value-yields-a-
  # bare-tag defect that wrk_tag two lines below was fixed for with filter-out, still live here.
  ( cd "$EX" && WA_C_DEPTH=1 WA_AB_DEPTH=2 WA_L3L2_DEPTH=2 \
      make -f Makefile.modal NPU2=1 M=512 K=1024 N=1024 m=64 k="$k" n="$n" n_aie_cols="$c" \
      emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 no_silu=1 "build/final_${sfx}.xclbin" ) >>"$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$B/final_$sfx.xclbin" ]; then
    # Read the TOP-LEVEL generated MLIR: PROFILE=production does not retain the .prj's
    # input_with_addresses.mlir that the traced builds are read from.
    ops=$(grep -m1 'aie.runtime_sequence(' "$B/aie_$sfx.mlir" 2>/dev/null | grep -o 'memref<' | wc -l)
    bytes=$(stat -c%s "$B/final_$sfx.xclbin")
    for f in "final_$sfx.xclbin:final_$nt.xclbin" "insts_$sfx.txt:insts_$nt.txt" \
             "aie_$sfx.mlir:aie_$nt.mlir" "aie_$sfx.mlir.prj:aie_$nt.mlir.prj"; do
      src="${f%%:*}"; dst="${f##*:}"; rm -rf "$B/$dst"; [ -e "$B/$src" ] && mv "$B/$src" "$B/$dst"
    done
    log "$(printf '  %4s | %3s | %3s | %5s | %8s | %12s | %s' "$c" "$k" "$n" "OK" "$ops" "$bytes" "$nt")"
    [ "$ops" = "3" ] || log "  WARN: $nt still declares $ops operands -- expected 3 (A,B,C) with no trace BO."
  else
    log "$(printf '  %4s | %3s | %3s | %5s | %8s | %12s | rc=%s' "$c" "$k" "$n" "FAIL" "-" "-" "$rc")"
  fi
  unstash "$sfx"
done

log ""
log "log: $LOG"
