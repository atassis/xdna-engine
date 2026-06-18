#!/usr/bin/env bash
# =============================================================================================
# Array-fill measurement: per-token NPU dispatch at B=16 (g_cols=1, 4/32 cores) vs B=128
# (g_cols=8, 32/32 cores) for the SAME layer count, via the NL-aware BatchedFusedDecoder.
# Output is garbage (short ELF) but the DISPATCH timing is valid — that's the array-fill number.
# RUN-ONLY (ELFs+bins prebuilt). Single-tenant NPU: stops npu-asr/voxd, ALWAYS restarts.
#
#   bash scripts/measure_arrayfill.sh <B16_dir> <B128_dir>
#   e.g. bash scripts/measure_arrayfill.sh artifacts/decode_batched_B16_L1_sp_nopdi artifacts/decode_batched_B128_L1_sp_nopdi
# =============================================================================================
set -u
WT="$REPO"; cd "$WT"
LDLIB=~/.local/lib/npu-asr
VERIFY="$WT/rust/target/release/verify_batched_decode"
CLIP="$WT/artifacts/wer_clips/en_04.wav"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="$WT/artifacts/arrayfill_${TS}.log"
log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; }
trap 'restart; log "[done] $LOG"' EXIT

measure_one(){  # $1=dir -> echoes "B<nl> dispatch_ms per_tok"
  local dir="$1"
  local B; B="$(python3 -c "import json;print(json.load(open('$dir/meta.json'))['dims']['B'])")"
  local NL; NL="$(python3 -c "import json;print(json.load(open('$dir/meta.json'))['dims']['layers'])")"
  local ph="$WT/artifacts/af_$(basename "$dir")_${TS}.phase"
  env NPU_DECODE_FUSED_BATCH=1 NPU_DECODE_FUSED_BATCH_DIR="$dir" FUSED_PHASE_TIMING=1 LD_LIBRARY_PATH=$LDLIB \
      timeout 300 "$VERIFY" --replicate "$B" "$CLIP" >/dev/null 2>"$ph"
  local disp; disp="$(grep -oE "dispatch_ms=[0-9.]+" "$ph" | head -1 | cut -d= -f2)"
  local pertok; pertok="$(python3 -c "print('%.4f'%(${disp:-0}/$B))")"
  log "  B=$B NL=$NL : dispatch=${disp} ms/dispatch  per-token=${pertok} ms"
  echo "$B ${disp:-0} $pertok"
}

[ $# -eq 2 ] || { echo "usage: measure_arrayfill.sh <B16_dir> <B128_dir>"; exit 1; }
log "================ ARRAY-FILL  $TS ================"
log "[svc] stopping npu services"; systemctl --user stop npu-asr.service voxd.service >/dev/null 2>&1; sleep 1
if fuser /dev/accel/accel0 >/dev/null 2>&1; then log "FATAL device busy"; fuser -v /dev/accel/accel0 2>&1|tee -a "$LOG"; exit 1; fi
log "[svc] device clear"

log "\n-- low-B (g_cols=1) --"; r1=($(measure_one "$1"))
log "\n-- high-B (g_cols=8) --"; r2=($(measure_one "$2"))

log "\n================ ARRAY-FILL RESULT ================"
python3 -c "
b1,d1,p1=${r1[0]},${r1[1]},${r1[2]}
b2,d2,p2=${r2[0]},${r2[1]},${r2[2]}
print(f'  per-token dispatch: B={b1} {p1} ms  vs  B={b2} {p2} ms')
print(f'  ARRAY-FILL (per-token): {p1/p2:.2f}x faster at B={b2}' if p2>0 else '  (no data)')
print(f'  dispatch_ms grew {d1}->{d2} = {d2/d1:.2f}x for {b2//b1}x the streams (ideal flat=1.0x)' if d1>0 else '')
" | tee -a "$LOG"
