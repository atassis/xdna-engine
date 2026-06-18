#!/usr/bin/env bash
# E2E before/after A/B for the validated V-transpose coalescing flags (INDIVIDUAL green flags).
#   baseline = artifacts/fused_decode12        (deep-C, no coalesce)
#   cross    = artifacts/fd12_cross            (--coalesce-cross)
#   self     = artifacts/fd12_self             (--coalesce-self)
# Reuses the deep-C resident engine path (only the transposes differ). Measures e2e ms, ms/token,
# per-phase decode breakdown (FUSED_PHASE), and RAPL pkg energy J/transcription (if readable).
# Single-tenant: stops npu-asr+voxd, restarts on exit. Correctness already gated in
# coalesce-cross-self-validated.md (all = WER 0.1172). This run is TIMING/ENERGY only.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
W3="$WT/rust/target/release/whisper_e2e_timing"
LDLIB=~/.local/lib/npu-asr
CLIP="$WT/artifacts/wer_clips/en_01.wav"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="$WT/artifacts/coalesce_e2e_ab_${TS}.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; echo "[svc] npu services restarted" | tee -a "$LOG"; }
trap 'restart; echo "[done] log: $LOG"' EXIT

[ -x "$W3" ] || { log "[ERR] whisper_e2e_timing missing — build: (cd rust && cargo build -p npu-engine --release --bin whisper_e2e_timing)"; exit 1; }
ENC="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build/final_512x800x3072_64x32x96_8c_modalsilu.xclbin"
[ -f "$ENC" ] || { log "[ERR] encoder xclbin missing: $ENC"; exit 1; }
for d in fused_decode12 fd12_cross fd12_self; do
  [ -f "$WT/artifacts/$d/decode.elf" ] || { log "[ERR] missing ELF: artifacts/$d/decode.elf"; exit 1; }
done

# RAPL for energy (best-effort, no interactive sudo).
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
if cat "$RAPL" >/dev/null 2>&1; then log "[rapl] readable — energy measured"
else sudo -n chmod -R a+r /sys/class/powercap/intel-rapl*/ 2>/dev/null || true
  cat "$RAPL" >/dev/null 2>&1 && log "[rapl] readable after sudo -n" \
    || log "[rapl] NOT readable -> energy N/A (enable once: sudo chmod -R a+r /sys/class/powercap/intel-rapl*/)"
fi

log "================ COALESCE E2E A/B  $TS ================"
log "[svc] stopping npu-asr + voxd (single-tenant) ..."
systemctl --user stop npu-asr.service voxd.service; sleep 2
fuser /dev/accel/accel0 2>/dev/null && { log "[ERR] device busy — aborting"; exit 1; }
log "[svc] device clear"

run(){  # $1 label  $2 dir
  log "\n----------------- [$1]  ($2) -----------------"
  env WHISPER_TIMING=1 FUSED_PHASE_TIMING=1 NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR="$WT/artifacts/$2" \
      LD_LIBRARY_PATH="$LDLIB" "$W3" "$CLIP" 2>&1 \
    | grep -E "WHISPER_TIMING|WHISPER_ENERGY|FUSED_PHASE\] (steps|per-token)|dispatch |lm_head |warmup text" \
    | tee -a "$LOG"
}
run baseline fused_decode12
run cross    fd12_cross
run self     fd12_self

log "\n################  SUMMARY (e2e_ms / ms-per-tok / J-per-transcription)  ################"
grep -E "^----|WHISPER_TIMING|WHISPER_ENERGY" "$LOG" | tee -a "$LOG" >/dev/null
log "[done] full log: $LOG"
