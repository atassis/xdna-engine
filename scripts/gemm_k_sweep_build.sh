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
# Arms are k:ab_depth:n[:l3l2_depth[:trace_worker]]; l3l2_depth defaults to the shipped 2 and
# trace_worker to the generator default 0.
ARMS="${1:-32:2:128 32:2:64 64:2:64 128:1:64}"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
LOG="$WT/artifacts/gemm_k_sweep_build.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

# INSTR_VECTOR is held in every set as the cross-run anchor: under 8 high-rate sources the trace
# drops events roughly uniformly, and a stable MEAN duration against a falling COUNT is what tells a
# lost event apart from a changed execution.
# WA_EVENTS overrides the set. Set it EMPTY to fall back to the generator default, which spends
# the 8 slots on the stall question instead (MEMORY/STREAM/LOCK_STALL + PORT_RUNNING). The two
# sets are complementary and neither is complete: the issue set below leaves stall dark, the
# default leaves scalar/load-store dark (66.95% of span at the deployed tile). INSTR_VECTOR is
# in both, and is the anchor that lets two runs of the same arm be composed.
EVENTS="${WA_EVENTS-INSTR_EVENT_0,INSTR_EVENT_1,INSTR_VECTOR,ACTIVE,DISABLED,INSTR_LOAD,INSTR_STORE,INSTR_LOCK_ACQUIRE_REQ}"

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
[ -n "${MLIR_AIE_INSTANCE:-}" ] || { log "FATAL: iron_env did not set MLIR_AIE_INSTANCE"; exit 1; }

log "=========== whole_array K-tile sweep build  cols=4  $(date -Is) ==========="
log "instance: $MLIR_AIE_INSTANCE"
log "arms (k:ab_depth:n:l3l2_depth): $ARMS"
log ""
log "    k | ab | l3 |   n |  w | L1 A+B+C KB | build | xclbin bytes | suffix"
log "  ----+----+----+-----+----+-------------+-------+--------------+-------"

for arm in $ARMS; do
  IFS=: read -r k d n l3 w <<< "$arm"; n="${n:-128}"; l3="${l3:-2}"; w="${w:-0}"
  kb="$(awk -v k="$k" -v d="$d" -v n="$n" 'BEGIN{ printf "%.0f", (d*64*k*2 + d*k*n*2 + 64*n*4)/1024 }')"
  # Depth 2 is the generator default and stays untagged, so existing artifact names survive;
  # EVERY other depth must tag. Tagging only d=1 meant d=3 and d=4 built into the d=2 name and
  # silently overwrote it -- caught mid-build on 2026-08-18.
  abtag=""; [ "$d" = "2" ] || abtag="ab$d"
  # Same rule for the L3->L2 hop -- a resized shim->memtile buffer is a different xclbin.
  l3tag=""; [ "$l3" = "2" ] || l3tag="l3$l3"
  # And for the traced WORKER. It changes only which core carries the trace packets, but that is a
  # different xclbin AND a different measurement. A multicasts down a column and B across a row, so
  # each core sits at the intersection of two different 4-core rendezvous, and MEASURED 2026-08-18
  # over 27 runs / 3 sessions the binding feed differs by core: tile (3,5) stalls against the B
  # channel (7/7 reps) where (0,2) and (1,3) stall against A (20/20). One core is not representative
  # of the array's feed behaviour. Worker 0 is the generator default and stays untagged so banked
  # artifact names survive.
  wtag=""; [ "$w" = "0" ] || wtag="w$w"
  sfx="512x1024x1024_64x${k}x${n}_4c_modalid${abtag}${l3tag}${wtag}"
  out="$EX/build/final_${sfx}.xclbin"
  # Drop the generated MLIR too, not just the xclbin. make's dependency is the .mlir FILE, so it
  # cannot see that wa_trace_events (or any other generator env) changed: a rerun with a different
  # event set silently relinks the SAME traced design and you measure the old set. Caught on
  # 2026-08-18 -- a stall-set rebuild produced a byte-identical xclbin against a .mlir from the
  # previous day. Regenerating costs seconds; the failure is silent and produces wrong numbers.
  rm -f "$out" "$EX/build/aie_${sfx}.mlir"
  # Name the xclbin target. The DEFAULT goal also builds whole_array_modal.exe, the C++ host test,
  # which fails here for reasons unrelated to the design -- gating on make's exit code instead of the
  # artifact reads every arm as FAIL while all three xclbins are sitting in build/.
  ( cd "$EX" && PROFILE=trace WA_C_DEPTH=1 WA_AB_DEPTH="$d" WA_L3L2_DEPTH="$l3" wa_trace_size=1048576 wa_trace_events="$EVENTS" wa_trace_worker="$w" \
      make -f Makefile.modal NPU2=1 M=512 K=1024 N=1024 m=64 k="$k" n="$n" n_aie_cols=4 \
      emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 no_silu=1 "build/final_${sfx}.xclbin" ) >>"$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$out" ]; then
    log "$(printf '  %4s | %2s | %2s | %3s | %2s | %11s | %5s | %12s | %s' "$k" "$d" "$l3" "$n" "$w" "$kb" "OK" "$(stat -c%s "$out")" "$sfx")"
  else
    log "$(printf '  %4s | %2s | %2s | %3s | %2s | %11s | %5s | %12s | rc=%s' "$k" "$d" "$l3" "$n" "$w" "$kb" "FAIL" "-" "$rc")"
  fi
done

log ""
log "READ: compare the three traced runs' INSTR_LOAD / INSTR_VECTOR split. Baseline k=32 d2 measured"
log "INSTR_LOAD 41.60%, INSTR_VECTOR 22.28%, span 407710 cyc. If K amortization transfers, k=64 d1"
log "shifts issue AWAY from load; if the d1 control alone moves it, the trade is what moved, not K."
log "log: $LOG"
