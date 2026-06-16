#!/usr/bin/env bash
# =============================================================================================
# IDLE full-NPU ASR perf sweep — run on a quiesced box, browser/video closed, so the
# machine is TRULY IDLE. The numbers it produces are the canonical energy/latency comparison the
# full-NPU-ASR-perf program needs (P3-capstone-grade), free of background CPU/LPDDR contention.
#
#   bash scripts/idle_perf_sweep.sh
#
# It is fully unattended: stops npu-asr/voxd (single-tenant NPU), builds the binaries, runs the
# sweep, ALWAYS restarts the services on exit (even on error/Ctrl-C), and beeps when finished.
# Everything is logged to  artifacts/idle_perf_sweep_<timestamp>.log  with a SUMMARY at the end.
#
# Backends compared (all use the NPU encoder; decode differs):
#   onnx            — CPU decode (the baseline)
#   fused_npucross  — full-NPU fused decode + cross-K/V fold on NPU   (current default, lever #2)
#   fused_hostcross — full-NPU fused decode + cross-K/V fold on host  (pre-lever-2, for the A/B)
# Metrics: per-stage latency (encoder/decode/ms-per-token), per-phase decode breakdown
# (FUSED_PHASE), RAPL package energy (J/transcription, avg W, RTF) on en_01, and pooled 17-clip WER.
#
# RAPL note: /sys/.../intel-rapl:0/energy_uj is root-only and resets on reboot. If energy shows N/A,
# run ONCE before this script:   sudo chmod -R a+r /sys/class/powercap/intel-rapl*/
# =============================================================================================
set -u

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WT"
LDLIB=~/.local/lib/npu-asr
W3="$WT/rust/target/release/whisper_e2e_timing"
SERVE="$WT/rust/target/release/engine_serve"
SCEN="$WT/scenarios/asr-whisper-small.toml"
CLIP="$WT/artifacts/wer_clips/en_01.wav"
RAPL=/sys/class/powercap/intel-rapl:0/energy_uj
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$WT/artifacts/idle_perf_sweep_${TS}.log"
WERDIR="$WT/artifacts/idle_wer_${TS}"
PORT=11434
URL="http://127.0.0.1:${PORT}/v1/audio/transcriptions"
mkdir -p "$WT/artifacts" "$WERDIR"; : > "$LOG"

log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; echo "[svc] npu services restarted" | tee -a "$LOG"; }
beep(){ ( speaker-test -t sine -f 1000 -l 1 >/dev/null 2>&1 & local p=$!; sleep 2; kill -9 "$p" >/dev/null 2>&1 ); }
trap 'restart; beep; echo "[done] log: $LOG" ' EXIT

log "================ IDLE PERF SWEEP  $TS ================"
log "host: $(uname -srm)"
log "uptime:$(uptime)"
log "-- top CPU processes (should be near-idle; investigate if anything is hot) --"
ps -eo comm,pcpu,pmem --sort=-pcpu 2>/dev/null | head -8 | tee -a "$LOG"

# ---- RAPL readability ----
if cat "$RAPL" >/dev/null 2>&1; then
  log "[rapl] energy_uj readable — energy will be measured"
else
  sudo -n chmod -R a+r /sys/class/powercap/intel-rapl*/ 2>/dev/null || true
  if cat "$RAPL" >/dev/null 2>&1; then
    log "[rapl] readable after 'sudo -n chmod'"
  else
    log "[rapl] NOT readable — ENERGY WILL BE N/A."
    log "       To enable, run this ONCE then re-run the sweep:"
    log "         sudo chmod -R a+r /sys/class/powercap/intel-rapl*/"
  fi
fi

# ---- binaries (PREBUILT) ----
# Prebuild before running so the idle measurement does NOT spend CPU compiling (a parallel cargo build
# spikes every core and warms the package — not part of an idle run). Build manually first with:
#   ( cd rust && cargo build -p npu-engine --release --bin whisper_e2e_timing --bin engine_serve )
# This script only compiles if a binary is missing (safety net for a fresh checkout).
if [ -x "$W3" ] && [ -x "$SERVE" ]; then
  log "\n[build] binaries present — skipping compile (idle run uses prebuilt)"
else
  log "\n[build] a binary is missing — compiling (prebuild next time to keep the run idle) ..."
  if ( cd "$WT/rust" && cargo build -p npu-engine --release --bin whisper_e2e_timing --bin engine_serve ) >>"$LOG" 2>&1; then
    log "[build] ok"
  else
    log "[build] FAILED — see log above. Aborting."; exit 1
  fi
fi

# ---- single-tenant ----
log "\n[svc] stopping npu-asr + voxd for single-tenant NPU ..."
systemctl --user stop npu-asr.service voxd.service; sleep 2
if fuser /dev/accel/accel0 2>/dev/null; then
  log "[ERR] /dev/accel/accel0 still busy after stopping services — aborting (services will be restarted)."; exit 1
fi
log "[svc] device clear — single-tenant"

# ===========================================================================================
log "\n################  E2E LATENCY + PER-PHASE + RAPL ENERGY (en_01, warmup+3 passes)  ################"
run_e2e(){  # $1 = label ; $2.. = extra env KV pairs
  local label="$1"; shift
  log "\n----------------- [$label] -----------------"
  env WHISPER_TIMING=1 FUSED_PHASE_TIMING=1 "$@" LD_LIBRARY_PATH="$LDLIB" "$W3" "$CLIP" 2>&1 \
    | grep -E "FUSED_PHASE\] (steps|per-token)|  [a-z_]+ +[0-9]|WHISPER_TIMING|WHISPER_ENERGY|warmup text|cross-K/V" \
    | tee -a "$LOG"
}
run_e2e onnx
# fused_npucross now runs the PIPE path by DEFAULT (async-prefetch the position-only ELF registration
# under the dispatch → load_elf leaves step_sum, moves into prefetch_ms). To A/B PIPE vs the old
# synchronous path, build the pre-PIPE commit (parent of the PIPE landing) into a second binary and
# run it here — PIPE is no longer an env flag (it had no regime where the sync path wins).
run_e2e fused_npucross  NPU_DECODE_FUSED=1
run_e2e fused_hostcross NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_HOSTCROSS=1

# ===========================================================================================
log "\n################  POOLED WER (17 clips)  ################"
run_wer(){  # $1 = label ; $2.. = extra env KV pairs
  local label="$1"; shift
  local serve_log="$WERDIR/serve_${label}.log"
  log "\n----------------- WER [$label] -----------------"
  env LD_LIBRARY_PATH="$LDLIB" "$@" "$SERVE" "$SCEN" "$PORT" >"$serve_log" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then up=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then log "[wer:$label] serve died early:"; tail -15 "$serve_log" | tee -a "$LOG"; return 1; fi
    sleep 1
  done
  [ "$up" = 1 ] || { log "[wer:$label] serve never became healthy"; kill "$pid" 2>/dev/null; return 1; }
  python3 "$WT/scripts/whisper_decode_wer.py" --url "$URL" --label "$label" --out "$WERDIR/wer_${label}.json" 2>&1 | tee -a "$LOG"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 2
}
run_wer onnx
run_wer fused_npucross  NPU_DECODE_FUSED=1
run_wer fused_hostcross NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_HOSTCROSS=1

# ===========================================================================================
log "\n################  SUMMARY  ################"
# Snapshot the key lines (read from a copy so we don't grep the log while appending to it).
SNAP="$WERDIR/_snapshot.txt"; cp "$LOG" "$SNAP"
log "-- per-backend e2e timing (encoder / decode / ms-per-token / dispatches) --"
grep -E "WHISPER_TIMING" "$SNAP" | sed 's/^/  /' | tee -a "$LOG"
log "-- per-backend energy (RAPL pkg J/transcription, avg W, RTF) --"
grep -E "WHISPER_ENERGY" "$SNAP" | sed 's/^/  /' | tee -a "$LOG"
log "-- fused per-phase breakdown (mean ms/token; cross_fold_ms is per-utterance) --"
grep -E "FUSED_PHASE\] steps" "$SNAP" | sed 's/^/  /' | tee -a "$LOG"
log "-- pooled WER (per backend; onnx baseline ~0.1136) --"
grep -E "ALL pooled WER|=== label=" "$SNAP" | sed 's/^/  /' | tee -a "$LOG"

log "\n[done] full log: $LOG"
log "[done] idle sweep complete — compare fused_npucross vs onnx for the energy headline."
