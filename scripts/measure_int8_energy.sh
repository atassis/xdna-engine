#!/usr/bin/env bash
# Batched energy + timing A/B of the int8 dataflow variants vs baseline, on a QUIESCED box.
#
# Run this ONCE the laptop is quiesced (no game/video/heavy CPU) — RAPL package energy is whole-SoC, so a
# loaded box invalidates J/transcription (cf. docs memory quiesce-before-energy-measurement). WER/correctness
# is load-independent and already gated elsewhere; THIS script is for the energy+latency the byte cuts buy.
#
#   bash scripts/measure_int8_energy.sh [clip.wav]
#
# For each ELF variant it runs whisper_e2e_timing (warmup + 3 timed passes, RAPL package energy) and prints
# pkg_J_per_transcription + e2e_ms + decode_ms. The byte cut is certain (meta layout); this measures whether
# it converts to real J + wall-time at M=1 (decode is launch-overhead-bound, so the conversion is the unknown).
set -u
cd "$(dirname "$0")/.."
REPO="$(pwd)"
CLIP="${1:-$REPO/artifacts/wer_clips/en_01.wav}"
LIBDIR="$REPO/rust/target/release/build/npu-onnx-d479791e01d0bb48/out"
BIN="$REPO/rust/target/release/whisper_e2e_timing"
# baseline first, then the byte-cut variants in increasing aggressiveness.
VARIANTS=(fused_decode12 fused_decode12_int8kv fused_decode12_int8ffn fused_decode12_int8sweet fused_decode12_int8all)

[ -x "$BIN" ] || { echo "build first: (cd rust && cargo build -p npu-engine --release --bin whisper_e2e_timing)"; exit 1; }

# ---- quiesce gate ----
LOAD=$(cut -d' ' -f1 /proc/loadavg)
echo "[measure] load avg (1m) = $LOAD"
if awk "BEGIN{exit !($LOAD > 3.0)}"; then
  echo "[measure] WARNING: load $LOAD > 3.0 — box is NOT quiesced. RAPL energy will be CONTAMINATED."
  echo "[measure] Quit the game/video and re-run, or pass through anyway (timing still indicative, energy NOT)."
fi
for p in parakeet_serve; do pgrep -af "$p" | grep -qv 'grep' && echo "[measure] NOTE: $p running — will be stopped below"; done

restart() { echo "[measure] restarting npu services"; systemctl --user start npu-asr voxd 2>/dev/null; }
trap restart EXIT
echo "[measure] stopping services + clearing device (single-tenant)"
systemctl --user stop npu-asr voxd 2>/dev/null
pkill -f 'parakeet[_]serve' 2>/dev/null   # bracket avoids self-match of this script's own cmdline
sleep 3
if fuser /dev/accel/accel0 2>/dev/null; then echo "[measure] ERROR: device still busy"; exit 1; fi
echo "[measure] device clear; clip=$CLIP"
echo

printf '%-28s %14s %10s %11s %9s\n' VARIANT J/transcription e2e_ms decode_ms tokens
for V in "${VARIANTS[@]}"; do
  D="$REPO/artifacts/$V"
  [ -d "$D" ] || { printf '%-28s %14s\n' "$V" "(missing)"; continue; }
  OUT=$(env WHISPER_TIMING=1 NPU_DECODE_FUSED=1 NPU_DECODE_FUSED_DIR="artifacts/$V" \
            LD_LIBRARY_PATH="$LIBDIR" "$BIN" "$CLIP" 2>&1)
  J=$(printf '%s\n' "$OUT" | sed -n 's/.*pkg_J_per_transcription=\([0-9.]*\).*/\1/p' | tail -1)
  # last timed pass's e2e/decode/tokens
  LINE=$(printf '%s\n' "$OUT" | grep '\[WHISPER_TIMING\]' | tail -1)
  E2E=$(printf '%s\n' "$LINE" | sed -n 's/.*e2e_ms=\([0-9.]*\).*/\1/p')
  DEC=$(printf '%s\n' "$LINE" | sed -n 's/.*decode_ms=\([0-9.]*\).*/\1/p')
  TOK=$(printf '%s\n' "$LINE" | sed -n 's/.*tokens=\([0-9]*\).*/\1/p')
  printf '%-28s %14s %10s %11s %9s\n' "$V" "${J:-?}" "${E2E:-?}" "${DEC:-?}" "${TOK:-?}"
done
echo
echo "[measure] done. (J/transcription is the energy headline; decode_ms is the M=1 latency. Byte cuts:"
echo "  int8kv -28.3 / int8ffn -56.6 / int8sweet -84.9 / int8all -127 MB/token vs baseline.)"
