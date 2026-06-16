#!/usr/bin/env bash
# =============================================================================================
# LEVER #3 — coalescing CORRECTNESS isolation (run with other NPU sessions stopped; single-tenant).
#
#   bash scripts/lever3_isolate.sh
#
# The combined coalescing is numerically broken (lever3_ab WER 0.996). This isolates WHICH of the two
# transposes broke it, by pooled-WER-testing each in isolation against the deep-C baseline. Correctness
# only (no timing/energy). Build the variant ELFs first (already built by the isolation step):
#   fd12_cross : --coalesce-cross  (i) Venc pre-transpose, self stays per-head num_batches=1
#   fd12_self  : --coalesce-self   (1) batched self-V transpose num_batches=H, cross stays per-head
#
# EXPECTED (hypothesis): baseline & cross = 0.1172 (cross also proves my IRON num_batches=1 path);
# self = ~0.99 (the batched-transpose IRON TAP port is the bug). WER ~0.1172 = WER-safe.
# Everything -> artifacts/lever3_isolate_<ts>.log.
# =============================================================================================
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
LDLIB=~/.local/lib/npu-asr
SERVE="$WT/rust/target/release/engine_serve"
SCEN="$WT/scenarios/asr-whisper-small.toml"
CLIP="$WT/artifacts/wer_clips/en_01.wav"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="$WT/artifacts/lever3_isolate_${TS}.log"; WERDIR="$WT/artifacts/lever3_iso_${TS}"
PORT=11434; URL="http://127.0.0.1:${PORT}/v1/audio/transcriptions"
mkdir -p "$WT/artifacts" "$WERDIR"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; echo "[svc] npu services restarted" | tee -a "$LOG"; }
beep(){ ( speaker-test -t sine -f 1000 -l 1 >/dev/null 2>&1 & local p=$!; sleep 2; kill -9 "$p" >/dev/null 2>&1 ); }
trap 'restart; beep; echo "[done] log: $LOG"' EXIT

log "================ LEVER-3 COALESCE ISOLATION  $TS ================"
# preflight: encoder xclbin (worktree mlir-aie must symlink a checkout WITH built xclbins)
ENC="$WT/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build/final_512x800x3072_64x32x96_8c_modalsilu.xclbin"
[ -f "$ENC" ] || { log "[ERR] encoder xclbin missing: $ENC — fix: rm -rf mlir-aie && ln -s <main>/mlir-aie mlir-aie"; exit 1; }
[ -x "$SERVE" ] || { log "[build] engine_serve missing — building ..."; ( cd "$WT/rust" && cargo build -p npu-engine --release --bin engine_serve ) >>"$LOG" 2>&1 || { log "[ERR] build failed"; exit 1; }; }
for d in fused_decode12 fd12_cross fd12_self; do
  [ -f "$WT/artifacts/$d/decode.elf" ] || { log "[ERR] missing ELF: artifacts/$d/decode.elf (build the isolation variants first)"; exit 1; }
done

log "[svc] stopping npu-asr + voxd ..."; systemctl --user stop npu-asr.service voxd.service; sleep 2
fuser /dev/accel/accel0 2>/dev/null && { log "[ERR] device busy — another session running. Aborting."; exit 1; }
log "[svc] device clear — single-tenant"

run_wer(){  # $1 label ; $2.. extra env KV
  local label="$1"; shift; local sl="$WERDIR/serve_${label}.log"
  log "\n----------------- WER [$label] -----------------"
  env LD_LIBRARY_PATH="$LDLIB" "$@" "$SERVE" "$SCEN" "$PORT" >"$sl" 2>&1 &
  local pid=$! up=0
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { up=1; break; }
    kill -0 "$pid" 2>/dev/null || { log "[$label] serve died early:"; tail -12 "$sl" | tee -a "$LOG"; return 1; }
    sleep 1
  done
  [ "$up" = 1 ] || { log "[$label] serve never healthy"; kill "$pid" 2>/dev/null; return 1; }
  python3 "$WT/scripts/whisper_decode_wer.py" --url "$URL" --label "$label" --out "$WERDIR/wer_${label}.json" 2>&1 | tee -a "$LOG"
  curl -sf -F "file=@${CLIP}" -F "response_format=json" "$URL" 2>/dev/null > "$WERDIR/text_${label}.json" || true
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 2
}

run_wer onnx
run_wer baseline NPU_DECODE_FUSED=1
run_wer cross    NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR="$WT/artifacts/fd12_cross"
run_wer self     NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR="$WT/artifacts/fd12_self"

log "\n################  ISOLATION SUMMARY  ################"
SNAP="$WERDIR/_snap.txt"; cp "$LOG" "$SNAP"
grep -E "=== label=|ALL pooled WER" "$SNAP" | sed 's/^/  /' | tee -a "$LOG"
log "\n-- en_01 transcription per variant --"
for v in baseline cross self; do
  t="$WERDIR/text_${v}.json"; [ -s "$t" ] && log "  [$v] $(python3 -c "import json;print(json.load(open('$t')).get('text','')[:80])" 2>/dev/null)"
done
log "\n[interpret] baseline≈0.1172. cross≈0.1172 ⇒ (i) Venc + IRON num_batches=1 OK. self broken ⇒ the"
log "            batched-transpose (num_batches=H) IRON TAP port is the bug → keep (i), debug/drop (1)."
log "[done] full log: $LOG"
