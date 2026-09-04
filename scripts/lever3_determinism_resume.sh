#!/usr/bin/env bash
# Continuation for lever3_determinism_gate.sh: re-runs the NPU arms for clips whose fused-decode
# invocations died on DRM_IOCTL_AMDXDNA_CREATE_HWCTX (err=-22). That failure is CUMULATIVE, not a
# property of the clip -- the same clip runs clean from a fresh process after the device settles --
# so this pass adds a settle + device-clear check between clips. ONNX results are reused.
# Usage: bash scripts/lever3_determinism_resume.sh <existing artifacts/lever3_determinism_* dir>
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
OUT="${1:?usage: $0 <lever3_determinism dir>}"
W3="$WT/rust/target/release/whisper_e2e_timing"; LDLIB="$HOME/.local/lib/npu-asr"
LOG="$OUT/resume.log"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start xdna-engine.service >/dev/null 2>&1; }
trap 'restart; echo "[done] $OUT"' EXIT
systemctl --user stop xdna-engine.service >/dev/null 2>&1; sleep 2

arm(){
  local label="$1" rep="$2" clip="$3" stem; stem="$(basename "$clip" .wav)"
  local envs=(WHISPER_TIMING=1 "LD_LIBRARY_PATH=$LDLIB" NPU_DECODE_FUSED=1)
  case "$label" in
    base)  envs+=("NPU_DECODE_FUSED_DIR=$WT/artifacts/fused_decode12") ;;
    cross) envs+=("NPU_DECODE_FUSED_DIR=$WT/artifacts/fused_decode12_xcross") ;;
  esac
  local raw="$OUT/$label.r$rep.$stem.raw"
  env "${envs[@]}" "$W3" "$clip" >"$raw" 2>&1
  sed -n 's/^\[bench\] warmup text: //p' "$raw" > "$OUT/$label.r$rep.$stem.txt"
  sed -n 's/.*tokens=\([0-9]*\).*/\1/p' "$raw" | head -1 > "$OUT/$label.r$rep.$stem.ntok"
  grep -q 'CREATE_HWCTX' "$raw" && echo "HWCTX" || echo "ok"
}

for clip in "$WT"/artifacts/wer_clips/*.wav; do
  stem="$(basename "$clip" .wav)"
  # Only clips still missing a complete NPU set.
  complete=1
  for f in base.r1 base.r2 cross.r1 cross.r2; do [ -s "$OUT/$f.$stem.txt" ] || complete=0; done
  [ "$complete" = 1 ] && continue
  for a in base cross; do for r in 1 2; do
    st=$(arm "$a" "$r" "$clip")
    [ "$st" = "HWCTX" ] && log "[warn] $stem $a.r$r hit CREATE_HWCTX -- settling 15s and retrying once" && { sleep 15; st=$(arm "$a" "$r" "$clip"); }
    log "[resume] $stem $a.r$r $st"
    sleep 3   # let the driver reclaim the hw context before the next one
  done; done
done
log "[resume] done"
