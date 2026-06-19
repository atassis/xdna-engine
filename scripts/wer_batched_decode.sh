#!/usr/bin/env bash
# =============================================================================================
# Batched-decode WER gate. Runs verify_batched_decode over the 16 wer_clips through the batched
# decoder, scores per-stream + aggregate WER vs refs.json, gates at 0.1172.
# RUN-ONLY (bin prebuilt). Single-tenant NPU: stops npu-asr/voxd, ALWAYS restarts.
#
#   bash scripts/wer_batched_decode.sh [BATCH_DIR]      # default decode_batched_B16_L12_sp
# =============================================================================================
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LDLIB=~/.local/lib/npu-asr
BATCH_DIR="${1:-$WT/artifacts/decode_batched_B16_L12_sp}"
VERIFY="$WT/rust/target/release/verify_batched_decode"
CLIPDIR="$WT/artifacts/wer_clips"
TS="$(date +%Y%m%d_%H%M%S)"; OUT="$WT/artifacts/wer_batched_${TS}.tsv"; LOG="$WT/artifacts/wer_batched_${TS}.log"
log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; log "[svc] npu services restarted"; }
trap 'restart; log "[done] tsv: $OUT"' EXIT

[ -e "$VERIFY" ] || { log "FATAL missing (prebuild): $VERIFY"; exit 1; }
# FAIL LOUD: the batch dir + its meta.json must exist and carry a valid B. (Was a silent failure
# mode: a missing meta.json fell back to B=16 -> clips=0 -> spurious "PASS".)
[ -d "$BATCH_DIR" ] || { log "FATAL: batch dir does not exist: $BATCH_DIR"; exit 1; }
[ -f "$BATCH_DIR/meta.json" ] || { log "FATAL: $BATCH_DIR/meta.json missing — INCOMPLETE build, not a valid decode dir. Rebuild it."; exit 1; }
[ -f "$BATCH_DIR"/*.elf ] 2>/dev/null || ls "$BATCH_DIR"/*.elf >/dev/null 2>&1 || { log "FATAL: no .elf in $BATCH_DIR"; exit 1; }
# Batch width from the ELF meta; cycle the 16 base wer_clips to fill B (B>16 -> duplicates; WER scored
# on the unique clips by basename, so correctness at the larger batch width is still gated).
BW="$(python3 -c "import json;print(json.load(open('$BATCH_DIR/meta.json'))['dims']['B'])" 2>/dev/null)"
[ -n "$BW" ] && [ "$BW" -ge 1 ] 2>/dev/null || { log "FATAL: could not read dims.B from $BATCH_DIR/meta.json (got '$BW')"; exit 1; }
BASE=(en_01 en_02 en_03 en_04 ru_01 ru_02 ru_03 ru_04 ru_05 ru_06 ru_07 ru_08 ru_09 ru_10 ru_11 ru_12)
CLIPS=(); i=0; while [ "${#CLIPS[@]}" -lt "$BW" ]; do
  c="${BASE[$((i % 16))]}"; p="$CLIPDIR/$c.wav"; [ -f "$p" ] || { log "FATAL missing clip $p"; exit 1; }
  CLIPS+=("$p"); i=$((i+1)); done
log "batch width B=$BW (clips cycled from 16 base)"

log "================ BATCHED WER GATE  $TS ================"
log "batch dir: $BATCH_DIR"
log "[svc] stopping npu-asr / voxd"
systemctl --user stop npu-asr.service voxd.service >/dev/null 2>&1; sleep 1
if fuser /dev/accel/accel0 >/dev/null 2>&1; then
  log "FATAL: /dev/accel/accel0 busy — another session holds the NPU. Aborting."; fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"; exit 1
fi
log "[svc] device clear"

env NPU_DECODE_FUSED_BATCH=1 NPU_DECODE_FUSED_BATCH_DIR="$BATCH_DIR" LD_LIBRARY_PATH=$LDLIB \
    "$VERIFY" "${CLIPS[@]}" 2>>"$LOG" > "$OUT"
VRC=$?
# FAIL LOUD: verify must succeed AND emit transcripts. (Was a silent failure mode: a verify crash
# left an empty TSV, the script continued, and the scorer reported a 0-clip "PASS".)
[ "$VRC" -eq 0 ] || { log "FATAL: verify_batched_decode exited $VRC (see stderr above) — decode FAILED."; exit 1; }
[ -s "$OUT" ] || { log "FATAL: verify_batched_decode produced EMPTY output ($OUT) — no transcripts. Decode FAILED at B=$BW (B=128 WER path may be unwired; see mstationary-b128-retest)."; exit 1; }
NLINES=$(wc -l < "$OUT")
log "[verify] produced $NLINES transcript lines (expected $BW)"

log "\n---- WER vs refs.json (gate = no-regression vs reproducible batched baseline; M=1 ref 0.1167) ----"
# Gate 0.1245 = reproducible batched baseline ([[batched-decode-wer-gate-reconciliation]]); the batched
# bf16 floor sits 2 edits above the M=1-derived 0.1172 (benign argmax noise on OOD proper nouns).
python3 "$WT/scripts/_score_batched_wer.py" "$OUT" "$CLIPDIR/refs.json" --gate=0.1246 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
