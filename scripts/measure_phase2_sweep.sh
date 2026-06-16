#!/usr/bin/env bash
# =============================================================================================
# Phase-2 measurement sweep — RUN THIS ON THE IDLE BOX. Single-tenant NPU: stops npu-asr/voxd,
# ALWAYS restarts (trap), fuser-checks, beeps when done. RUN-ONLY (ELFs + bins prebuilt by the agent).
#
#   bash $REPO/scripts/measure_phase2_sweep.sh
#
# For every prebuilt batched-decode ELF config it measures, in ONE session:
#   - per-token NPU dispatch (FUSED_PHASE_TIMING) + lm_head
#   - per-stream WER vs refs.json (correctness gate, <= 0.1246)
#   - decode tok/s + J/token (RAPL package energy)
# Skips any config whose ELF isn't built yet. Writes everything to artifacts/phase2_sweep_<ts>.log
# AND prints a final summary table. Paste the summary table (or the log path) back to the agent.
# =============================================================================================
set -u
WT="$REPO"; cd "$WT"
LDLIB=~/.local/lib/npu-asr
CLIPDIR="$WT/artifacts/wer_clips"
VERIFY="$WT/rust/target/release/verify_batched_decode"
BULK="$WT/rust/target/release/bench_bulk_decode"
TS="$(date +%Y%m%d_%H%M%S)"; LOG="$WT/artifacts/phase2_sweep_${TS}.log"
: > "$LOG"; log(){ echo -e "$*" | tee -a "$LOG"; }

# configs to sweep: "label:dirname"  (skipped if the ELF is missing)
CONFIGS=(
  "baseline-B16:decode_batched_B16_L12_sp"
  "O18occ-B16:decode_batched_B16_L12_sp_occ"
  "liftB-B32:decode_batched_B32_L12_sp"
  "liftB-B48:decode_batched_B48_L12_sp"
  "arrayfill-B128:decode_batched_B128_L12_sp"
  "arrayfill+occ-B128:decode_batched_B128_L12_sp_occ"
)
BASE=(en_01 en_02 en_03 en_04 ru_01 ru_02 ru_03 ru_04 ru_05 ru_06 ru_07 ru_08 ru_09 ru_10 ru_11 ru_12)

restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; log "[svc] npu services restarted"; }
beep(){ ( speaker-test -t sine -f 1000 -l 1 >/dev/null 2>&1 & local p=$!; sleep 1; kill -9 "$p" >/dev/null 2>&1 ); }
trap 'restart; beep; log "\n[done] full log: $LOG"; print_summary' EXIT

declare -a SUMMARY
for f in "$VERIFY" "$BULK"; do [ -e "$f" ] || { log "FATAL missing bin (prebuild): $f"; exit 1; }; done

log "[svc] stopping npu-asr / voxd"; systemctl --user stop npu-asr.service voxd.service >/dev/null 2>&1; sleep 1
if fuser /dev/accel/accel0 >/dev/null 2>&1; then
  log "FATAL: /dev/accel/accel0 busy — another session holds the NPU. Aborting."; fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"; exit 1
fi
log "[svc] device clear\n================ PHASE-2 SWEEP  $TS ================"

clips_for_B(){ local B="$1"; local -n _out="$2"; _out=(); local i=0
  while [ "${#_out[@]}" -lt "$B" ]; do _out+=("$CLIPDIR/${BASE[$((i % 16))]}.wav"); i=$((i+1)); done; }

for cfg in "${CONFIGS[@]}"; do
  label="${cfg%%:*}"; dir="$WT/artifacts/${cfg#*:}"
  if [ ! -e "$dir/decode_b.elf" ]; then log "\n---- SKIP $label (no ELF at $dir) ----"; continue; fi
  B="$(python3 -c "import json;print(json.load(open('$dir/meta.json'))['dims']['B'])" 2>/dev/null || echo 16)"
  scr="$(python3 -c "import json;print('%.0f'%(json.load(open('$dir/meta.json'))['scratch_size']/1e6))" 2>/dev/null || echo '?')"
  declare -a CL; clips_for_B "$B" CL
  log "\n========== $label  (B=$B, scratch ${scr} MB) =========="

  # (1) dispatch breakdown + WER (FUSED_PHASE_TIMING verify)
  tsv="$WT/artifacts/sweep_${label}_${TS}.tsv"; ph="$WT/artifacts/sweep_${label}_${TS}.phase"
  env NPU_DECODE_FUSED_BATCH=1 NPU_DECODE_FUSED_BATCH_DIR="$dir" FUSED_PHASE_TIMING=1 LD_LIBRARY_PATH=$LDLIB \
      "$VERIFY" "${CL[@]}" 2>"$ph" >"$tsv"
  disp="$(grep -oE "dispatch_ms=[0-9.]+" "$ph" | head -1 | cut -d= -f2)"
  lmh="$(grep -oE "lm_head_ms=[0-9.]+" "$ph" | head -1 | cut -d= -f2)"
  pertok="$(python3 -c "print('%.2f'%(${disp:-0}/$B))" 2>/dev/null || echo '?')"
  werline="$(python3 "$WT/scripts/_score_batched_wer.py" "$tsv" "$CLIPDIR/refs.json" --gate=0.1246 2>/dev/null | grep WER-GATE)"
  wer="$(echo "$werline" | grep -oE "WER [0-9.]+" | head -1 | awk '{print $2}')"
  log "  dispatch=${disp} ms/dispatch  per-token=${pertok} ms  lm_head=${lmh} ms  | $werline"

  # (2) tok/s + J/token (bench_bulk, N=B one batch)
  bulk="$(env NPU_DECODE_FUSED_BATCH=1 NPU_DECODE_FUSED_BATCH_DIR="$dir" LD_LIBRARY_PATH=$LDLIB \
      "$BULK" "${CL[@]}" 2>/dev/null | grep -E "unsorted" | head -1)"
  log "  perf: $bulk"
  toks="$(echo "$bulk" | grep -oE "tok/s=[ ]*[0-9.]+" | grep -oE "[0-9.]+$")"
  jtok="$(echo "$bulk" | grep -oE "J/tok=[0-9.]+" | cut -d= -f2)"
  SUMMARY+=("$(printf '%-20s B=%-4s disp=%-8s ms/tok=%-7s WER=%-7s tok/s=%-7s J/tok=%-7s' "$label" "$B" "${disp:-?}" "${pertok:-?}" "${wer:-?}" "${toks:-?}" "${jtok:-?}")")
done

print_summary(){ echo; echo "================ SUMMARY (paste this back to the agent) ================" | tee -a "$LOG"
  for r in "${SUMMARY[@]}"; do echo "$r" | tee -a "$LOG"; done
  echo "baseline ref: B=16 disp=258.1 ms/tok=16.13 WER=0.1245 tok/s~23 J/tok~0.50" | tee -a "$LOG"; }
