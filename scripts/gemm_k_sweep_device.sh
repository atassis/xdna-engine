#!/usr/bin/env bash
# Trace the whole_array GEMM K-sweep arms on device (task gemm-offcore-residue-occupancy).
#
# Pairs with gemm_k_sweep_build.sh, which explains the arms and why k=64 needs WA_AB_DEPTH=1 to fit.
# Reads the per-event issue split so the three arms can be compared against the k=32 d2 baseline
# (INSTR_LOAD 41.60%, INSTR_VECTOR 22.28%, span 407710 cyc).
#
# Single-tenant NPU: stops xdna-engine + npu-vox, fuser-checks, and ALWAYS restores them on exit
# including on abort. Serialize with any other device session.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
EX="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array"
ARMS="${1:-512x1024x1024_64x32x128_4c_modalid 512x1024x1024_64x32x128_4c_modalidab1 512x1024x1024_64x64x128_4c_modalidab1}"
LOG="$WT/artifacts/gemm_k_sweep_device${KSWEEP_TAG:+_$KSWEEP_TAG}.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }

restore(){ systemctl --user start xdna-engine.service npu-vox.service >/dev/null 2>&1; log "[svc] restored"; }
trap restore EXIT

log "=========== whole_array K-sweep on device  $(date -Is) ==========="
log "[svc] stopping xdna-engine + npu-vox"
systemctl --user stop xdna-engine.service npu-vox.service >/dev/null 2>&1
sleep 2
# systemctl is-active is NOT the check -- it prints inactive for stopped, absent AND
# running-as-a-plain-process alike. fuser + pgrep is.
if fuser /dev/accel/accel0 >/dev/null 2>&1 || pgrep -f 'npu serve' >/dev/null; then
  log "FATAL: device still held -- another session has the NPU. Aborting (single-tenant)."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"; pgrep -af 'npu serve' | tee -a "$LOG"
  exit 75
fi
log "[svc] device clear"

source "$WT/scripts/iron_env.sh" >/dev/null 2>&1
# REPS loops the WHOLE arm list, so reps stay interleaved inside this one service-stop window --
# the box's per-session shift lands on acquire wait and would otherwise be read as an arm effect.
# Each rep gets its own artifact dir: the trace JSON is named from the suffix alone, so a second rep
# used to overwrite the first and only the summary line in this log survived. The raw trace is what
# gemm_trace_overlap.py needs, so a per-rep dir is what makes the overlap split repeatable.
REPS="${REPS:-1}"
for rep in $(seq 1 "$REPS"); do
  for sfx in $ARMS; do
    log ""
    log "---------- $sfx ----------"
    outdir="$WT/artifacts/ksweep_${KSWEEP_TAG:+${KSWEEP_TAG}_}$sfx"
    [ "$REPS" = "1" ] || outdir="${outdir}_r${rep}"
    mkdir -p "$outdir"
    .venv-iron/bin/python scripts/gemm_trace_probe.py \
        --build-dir "$EX/build" --suffix "$sfx" \
        --M 512 --K 1024 --N 1024 --trace-size 1048576 \
        --artifacts "$outdir" 2>&1 | tee -a "$LOG"
  done
done

log ""
log "log: $LOG"
