#!/usr/bin/env bash
# =============================================================================================
# Lever-3 --coalesce-cross DETERMINISM GATE (the correctness half of the default-flip case).
#
#   bash scripts/lever3_determinism_gate.sh
#
# The latency half is closed and positive (measured): break-even
# 11.3 tokens, CI [5.1, 15.0], below every clip in the shipped bank. What was still owed is the
# CORRECTNESS gate, and the standing rule is that it is NOT rel-L2 and NOT the
# 17-clip WER (chaotic at ~1e-5) -- it is 1:1 determinism
# against the CPU reference at temperature 0.
#
# Decode here is greedy/argmax, so "temperature 0" is structural, not a knob: two arms agree iff
# every argmax matches. This runs three arms over the whole 17-clip bank and compares emitted text
# EXACTLY (byte equality, no normalization -- normalization is what a WER gate does, and hiding a
# token flip is exactly what we do not want):
#
#   onnx     CPU decoder, NPU encoder            -- the reference
#   base     resident fused decode, no flag      -- artifacts/fused_decode12
#   cross    resident fused decode --coalesce-cross -- artifacts/fused_decode12_xcross
#
# Two reps for each NPU arm (run-to-run determinism is part of the claim), one for the CPU
# reference. Single-tenant: stops xdna-engine, restarts on exit.
#
# PASS requires all three: base==cross on every clip (the flip-relevant question), each NPU arm
# equal to itself across reps, and each NPU arm == onnx (the 1:1 gate). Any clip failing any of
# the three is printed with both strings.
# =============================================================================================
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
W3="$WT/rust/target/release/whisper_e2e_timing"
LDLIB="$HOME/.local/lib/npu-asr"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$WT/artifacts/lever3_determinism_${TS}"
LOG="$OUT/gate.log"
mkdir -p "$OUT"
log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start xdna-engine.service >/dev/null 2>&1; echo "[svc] xdna-engine restarted"; }
trap 'restart; echo "[done] $OUT"' EXIT

[ -x "$W3" ] || { echo "[ERR] missing $W3"; exit 1; }
for d in fused_decode12 fused_decode12_xcross; do
  [ -f "$WT/artifacts/$d/decode.elf" ] || { echo "[ERR] missing artifacts/$d/decode.elf"; exit 1; }
done

log "[svc] stopping xdna-engine (single-tenant)"
systemctl --user stop xdna-engine.service >/dev/null 2>&1; sleep 2
fuser /dev/accel/accel0 >/dev/null 2>&1 && { log "[ERR] device busy -- aborting"; exit 1; }
log "[svc] device clear"

# arm <label> <rep> <clip> -> writes "$OUT/<label>.r<rep>.<stem>.txt" holding the emitted text.
arm(){
  local label="$1" rep="$2" clip="$3" stem; stem="$(basename "$clip" .wav)"
  local envs=(WHISPER_TIMING=1 "LD_LIBRARY_PATH=$LDLIB")
  case "$label" in
    base)  envs+=(NPU_DECODE_FUSED=1 "NPU_DECODE_FUSED_DIR=$WT/artifacts/fused_decode12") ;;
    cross) envs+=(NPU_DECODE_FUSED=1 "NPU_DECODE_FUSED_DIR=$WT/artifacts/fused_decode12_xcross") ;;
    onnx)  : ;;
  esac
  local raw="$OUT/$label.r$rep.$stem.raw"
  env "${envs[@]}" "$W3" "$clip" >"$raw" 2>&1
  # The binary prints the transcription once, from the untimed warmup pass.
  sed -n 's/^\[bench\] warmup text: //p' "$raw" > "$OUT/$label.r$rep.$stem.txt"
  sed -n 's/.*tokens=\([0-9]*\).*/\1/p' "$raw" | head -1 > "$OUT/$label.r$rep.$stem.ntok"
}

CLIPS=( $(ls "$WT"/artifacts/wer_clips/*.wav | sort) )
log "[gate] ${#CLIPS[@]} clips x (onnx 1 rep + base 2 + cross 2)"
for clip in "${CLIPS[@]}"; do
  stem="$(basename "$clip" .wav)"
  arm onnx 1 "$clip"; arm base 1 "$clip"; arm cross 1 "$clip"
  arm base 2 "$clip"; arm cross 2 "$clip"
  log "[gate] $stem done ($(cat "$OUT/base.r1.$stem.ntok" 2>/dev/null) tok)"
done

log "\n################ VERDICT ################"
fail=0; n=0
for clip in "${CLIPS[@]}"; do
  stem="$(basename "$clip" .wav)"; n=$((n+1))
  b1="$OUT/base.r1.$stem.txt"; b2="$OUT/base.r2.$stem.txt"
  c1="$OUT/cross.r1.$stem.txt"; c2="$OUT/cross.r2.$stem.txt"; o1="$OUT/onnx.r1.$stem.txt"
  msg=""
  cmp -s "$b1" "$b2" || msg="$msg base-nondeterministic"
  cmp -s "$c1" "$c2" || msg="$msg cross-nondeterministic"
  cmp -s "$b1" "$c1" || msg="$msg base!=cross"
  cmp -s "$b1" "$o1" || msg="$msg base!=onnx"
  cmp -s "$c1" "$o1" || msg="$msg cross!=onnx"
  if [ -n "$msg" ]; then
    fail=$((fail+1))
    log "FAIL $stem ($(cat "$OUT/base.r1.$stem.ntok" 2>/dev/null) tok):$msg"
    log "  onnx : $(cat "$o1")"
    log "  base : $(cat "$b1")"
    log "  cross: $(cat "$c1")"
  else
    log "PASS $stem ($(cat "$OUT/base.r1.$stem.ntok" 2>/dev/null) tok)"
  fi
done
log "\n[gate] $((n-fail))/$n clips PASS, $fail FAIL"
[ "$fail" -eq 0 ] && log "[gate] DETERMINISM GATE: PASS" || log "[gate] DETERMINISM GATE: FAIL"
