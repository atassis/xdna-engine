#!/usr/bin/env bash
# Build the whole_array GEMM K-tile sweep at cols=4, traced (task gemm-offcore-residue-occupancy).
#
# A static compile sweep predicted 19% -> 51% of bfp16 peak on ISSUE SLOTS as the core K tile grows
# k=32 -> k=512, because the C accumulator's fixed 16 vlda + 16 vst per (z,j) only amortizes over K.
# Device trace at k=32 confirmed the mechanism -- INSTR_LOAD 41.60% vs INSTR_VECTOR 22.28% -- but not
# the prediction.
#
# MEASURED: raising k alone does not build, and hits TWO different walls.
#   k=64,128  -> L1 capacity, 'allocated buffers exceeded available memory'
#   k=256,512 -> 'aie.dma_bd length (16384 32-bit words) exceeds the maximum of 16383', a hard BD
#                limit the B L2 tile (k*n*2 B) crosses by exactly one word at k=256.
# So the static prediction cannot be tested by raising k. The only way up is to spend the L1 that
# A/B double-buffering holds, which WA_AB_DEPTH=1 does.
#
# ROUND 1 (n=128) MEASURED: K amortization is real -- k=32->64 at constant depth cut INSTR_LOAD
# -39.4% and INSTR_STORE -47.8% with INSTR_VECTOR flat -- but it cannot be cashed, because
# single-buffering A/B costs +35.2% span against the K doubling's -23.9%, leaving k=64/d1 2.9%
# SLOWER than the deployed k=32/d2. That trade is forced ONLY because C is 32 KB at n=128.
#
# ROUND 2 halves n. C drops to 16 KB and B with it, so k=64 fits in 48 KB WITH the double buffer
# retained -- K amortization bought without selling the overlap. ARMS are k:ab_depth:n.
#   k=32  d2 n=128  baseline, the deployed tile and the already-traced run
#   k=32  d2 n=64   control,  isolates the n retiling at constant k and depth
#   k=64  d2 n=64   test,     K doubled, double buffer kept
#   k=128 d1 n=64   second point, K doubled again by spending the buffer
# The n=64 control is load-bearing: at n=64, N=1024 over 4 cols is 4 output tiles per core instead
# of 2, i.e. MORE inner-loop entries -- the thing K amortization is trying to remove. Without it a
# result at n=64 cannot be attributed between the K tile and the retiling.
# k=256 at n=64 clears the BD limit (8192 words) but not L1 (80 KB at d1), so k=128 is the ceiling.
#
# L1 accounting (64 KB): A = depth*m*k*2, B = depth*k*n*2, C = m*n*4 single-buffered (WA_C_DEPTH=1).
#
# Device-free. Run:  bash scripts/gemm_k_sweep_build.sh
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
ARMS="${1:-32:2:128 32:2:64 64:2:64 128:1:64}"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
LOG="$WT/artifacts/gemm_k_sweep_build.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

# INSTR_VECTOR is held in every set as the cross-run anchor: under 8 high-rate sources the trace
# drops events roughly uniformly, and a stable MEAN duration against a falling COUNT is what tells a
# lost event apart from a changed execution.
EVENTS="INSTR_EVENT_0,INSTR_EVENT_1,INSTR_VECTOR,ACTIVE,DISABLED,INSTR_LOAD,INSTR_STORE,INSTR_LOCK_ACQUIRE_REQ"

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { log "FATAL: iron_env did not set MLIR_AIE_INSTANCE"; exit 1; }

log "=========== whole_array K-tile sweep build  cols=4  $(date -Is) ==========="
log "instance: $MLIR_AIE_INSTANCE"
log "arms (k:ab_depth): $ARMS"
log ""
log "    k | ab |   n | L1 A+B+C KB | build | xclbin bytes | suffix"
log "  ----+----+-----+-------------+-------+--------------+-------"

for arm in $ARMS; do
  IFS=: read -r k d n <<< "$arm"; n="${n:-128}"
  kb="$(awk -v k="$k" -v d="$d" -v n="$n" 'BEGIN{ printf "%.0f", (d*64*k*2 + d*k*n*2 + 64*n*4)/1024 }')"
  abtag=""; [ "$d" = "1" ] && abtag="ab1"
  sfx="512x1024x1024_64x${k}x${n}_4c_modalid${abtag}"
  out="$EX/build/final_${sfx}.xclbin"
  rm -f "$out"
  # Name the xclbin target. The DEFAULT goal also builds whole_array_modal.exe, the C++ host test,
  # which fails here for reasons unrelated to the design -- gating on make's exit code instead of the
  # artifact reads every arm as FAIL while all three xclbins are sitting in build/.
  ( cd "$EX" && PROFILE=trace WA_C_DEPTH=1 WA_AB_DEPTH="$d" wa_trace_size=1048576 wa_trace_events="$EVENTS" \
      make -f Makefile.modal NPU2=1 M=512 K=1024 N=1024 m=64 k="$k" n="$n" n_aie_cols=4 \
      emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 no_silu=1 "build/final_${sfx}.xclbin" ) >>"$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$out" ]; then
    log "$(printf '  %4s | %2s | %3s | %11s | %5s | %12s | %s' "$k" "$d" "$n" "$kb" "OK" "$(stat -c%s "$out")" "$sfx")"
  else
    log "$(printf '  %4s | %2s | %3s | %11s | %5s | %12s | rc=%s' "$k" "$d" "$n" "$kb" "FAIL" "-" "$rc")"
  fi
done

log ""
log "READ: compare the three traced runs' INSTR_LOAD / INSTR_VECTOR split. Baseline k=32 d2 measured"
log "INSTR_LOAD 41.60%, INSTR_VECTOR 22.28%, span 407710 cyc. If K amortization transfers, k=64 d1"
log "shifts issue AWAY from load; if the d1 control alone moves it, the trade is what moved, not K."
log "log: $LOG"
