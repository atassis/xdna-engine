#!/usr/bin/env bash
# =============================================================================================
# Subsystem-B perf: M=1 DECODE J/token + tok/s, canonical full e2e.
# RUN-ONLY (binaries are prebuilt). Single-tenant NPU: quiesces the NPU services, ALWAYS restarts, beeps.
#
#   bash scripts/bench_batched_decode.sh
#
# Produces (-> artifacts/bench_batched_<ts>.log):
#  npu-dev whisper-e2e (M=1, 1 clip): canonical full e2e line (preproc/encoder/decode ms, ms/tok,
#  dispatches/tok) + RAPL energy + [FUSED_PHASE] per-token decode breakdown.
#
# RAPL energy: if J shows n/a, run ONCE first:  sudo chmod -R a+r /sys/class/powercap/intel-rapl*/
# =============================================================================================
set -u
. "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_npu_services.sh" || exit 1   # unit names + asserted quiesce
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LDLIB=~/.local/lib/npu-asr
M1_DIR="$WT/artifacts/fused_decode12"                       # shipped M=1 deep-C ELF
E2E="$WT/rust/target/release/npu-dev"
CLIPDIR="$WT/artifacts/wer_clips"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="$WT/artifacts/bench_batched_${TS}.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ npu_svc_start; }
beep(){ ( speaker-test -t sine -f 1000 -l 1 >/dev/null 2>&1 & local p=$!; sleep 1; kill -9 "$p" >/dev/null 2>&1 ); }
trap 'restart; beep; log "[done] log: $LOG"' EXIT

# preflight (run-only: artifacts + bins must already exist)
for f in "$E2E" "$M1_DIR/decode.elf"; do
  [ -e "$f" ] || { log "FATAL missing (prebuild first): $f"; exit 1; }
done

log "================ M=1 DECODE BENCH  $TS ================"
log "host: $(uname -srm)"
log "M=1 dir:   $M1_DIR"

log "[svc] quiescing (single-tenant)"
npu_svc_stop || exit 1
if fuser /dev/accel/accel0 >/dev/null 2>&1; then
  log "FATAL: /dev/accel/accel0 busy — another session holds the NPU. Aborting."; fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"; exit 1
fi
log "[svc] device clear"

log "\n========== M=1 canonical full e2e (npu-dev whisper-e2e, en_01) =========="
env NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR="$M1_DIR" \
    WHISPER_TIMING=1 FUSED_PHASE_TIMING=1 LD_LIBRARY_PATH=$LDLIB \
    "$E2E" whisper-e2e "$CLIPDIR/en_01.wav" 2>&1 | tee -a "$LOG"

log "\n[bench] complete."
