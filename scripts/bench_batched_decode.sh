#!/usr/bin/env bash
# =============================================================================================
# Subsystem-B perf: batched (B=16) vs M=1 DECODE J/token + tok/s, with full per-step breakout.
# RUN-ONLY (binaries are prebuilt). Single-tenant NPU: stops npu-asr/voxd, ALWAYS restarts, beeps.
#
#   bash scripts/bench_batched_decode.sh
#
# Produces (-> artifacts/bench_batched_<ts>.log):
#  1. bench_batched_decode : encode stage (preproc+encoder) + DECODE-only batched vs M=1
#     (decode_ms, tokens, ms/tok, tok/s, J/tok) + per-dispatch phase breakdown
#     ([BATCHED_PHASE]/[FUSED_PHASE] on stderr) + full e2e split.
#  2. whisper_e2e_timing (M=1, 1 clip): canonical full e2e line (preproc/encoder/decode ms, ms/tok,
#     dispatches/tok) + RAPL energy + [FUSED_PHASE] per-token decode breakdown.
#
# RAPL energy: if J shows n/a, run ONCE first:  sudo chmod -R a+r /sys/class/powercap/intel-rapl*/
# =============================================================================================
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LDLIB=~/.local/lib/npu-asr
M1_DIR="$WT/artifacts/fused_decode12"                       # shipped M=1 deep-C ELF
BATCH_DIR="$WT/artifacts/decode_batched_B16_L12_sp"         # batched B=16 prod ELF
BENCH="$WT/rust/target/release/bench_batched_decode"
E2E="$WT/rust/target/release/whisper_e2e_timing"
CLIPDIR="$WT/artifacts/wer_clips"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="$WT/artifacts/bench_batched_${TS}.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; log "[svc] npu services restarted"; }
beep(){ ( speaker-test -t sine -f 1000 -l 1 >/dev/null 2>&1 & local p=$!; sleep 1; kill -9 "$p" >/dev/null 2>&1 ); }
trap 'restart; beep; log "[done] log: $LOG"' EXIT

# preflight (run-only: artifacts + bins must already exist)
for f in "$BENCH" "$E2E" "$M1_DIR/decode.elf" "$BATCH_DIR/decode_b.elf"; do
  [ -e "$f" ] || { log "FATAL missing (prebuild first): $f"; exit 1; }
done
CLIPS=(); for c in en_01 en_02 en_03 en_04 ru_01 ru_02 ru_03 ru_04 ru_05 ru_06 ru_07 ru_08 ru_09 ru_10 ru_11 ru_12; do
  p="$CLIPDIR/$c.wav"; [ -f "$p" ] || { log "FATAL missing clip $p"; exit 1; }; CLIPS+=("$p"); done

log "================ BATCHED vs M=1 DECODE BENCH  $TS ================"
log "host: $(uname -srm)   B=16   clips=${#CLIPS[@]}"
log "M=1 dir:   $M1_DIR"
log "batch dir: $BATCH_DIR  ($(python3 -c "import json;print('scratch %.0f MB'%(json.load(open('$BATCH_DIR/meta.json'))['scratch_size']/1e6))" 2>/dev/null))"

log "[svc] stopping npu-asr / voxd"
systemctl --user stop npu-asr.service voxd.service >/dev/null 2>&1; sleep 1
if fuser /dev/accel/accel0 >/dev/null 2>&1; then
  log "FATAL: /dev/accel/accel0 busy — another session holds the NPU. Aborting."; fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"; exit 1
fi
log "[svc] device clear"

log "\n========== (1) DECODE bench: batched B=16 vs M=1, same 16 clips, one session =========="
env NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR="$M1_DIR" \
    NPU_DECODE_FUSED_BATCH=1 NPU_DECODE_FUSED_BATCH_DIR="$BATCH_DIR" \
    FUSED_PHASE_TIMING=1 LD_LIBRARY_PATH=$LDLIB \
    "$BENCH" "${CLIPS[@]}" 2>&1 | tee -a "$LOG"

log "\n========== (2) M=1 canonical full e2e (whisper_e2e_timing, en_01) =========="
env NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR="$M1_DIR" \
    WHISPER_TIMING=1 FUSED_PHASE_TIMING=1 LD_LIBRARY_PATH=$LDLIB \
    "$E2E" "$CLIPDIR/en_01.wav" 2>&1 | tee -a "$LOG"

log "\n[bench] complete."
