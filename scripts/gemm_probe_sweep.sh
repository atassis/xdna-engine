#!/usr/bin/env bash
# =============================================================================================
# LEVER #3 vector-(b) Milestone-0 — batched-GEMM dispatch-amortisation sweep (single-tenant NPU).
#
#   bash scripts/gemm_probe_sweep.sh ["16 32 64 128"]
#
# Decisive question (design spec internal notes): does the
# per-token cost (dispatch_ms / N) FALL as the batch width N grows? Each ELF is one fc1 GEMM
# out[M=3072,N]=W[M,K]@X[K,N] with the SAME 4.72 MB weight read once. If dispatch_ms grows sub-linearly
# in N (per-token cost falls) => batching amortises the weight read+launch => GO for the full batched
# decode. If dispatch_ms ~ linear in N (per-token cost flat) => no amortisation on this HW => KILL.
#
# Builds the probe ELFs (device-free) if missing, then dispatches each on the NPU via fused_elf_probe
# FUSED_TIME (reports "dispatch alone" ms) and the rel-L2 <= 0.08 correctness gate. Fully unattended:
# stops npu-asr/voxd, ALWAYS restarts them on exit, fuser-checks the device, beeps when done.
# RAPL note: energy not measured here (dispatch-only microbench); ms/tok is the signal.
# =============================================================================================
# Env: NUM_COLS (default 1) selects the array config / artifact dir tag. Full-array sweep:
#   NUM_COLS=8 bash scripts/gemm_probe_sweep.sh "128 256 512"
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
NS="${1:-16 32 64 128}"
NUM_COLS="${NUM_COLS:-1}"
SUF=""; [ "$NUM_COLS" != "1" ] && SUF="_c${NUM_COLS}"
LDLIB=~/.local/lib/npu-asr
PROBE="$WT/rust/target/release/fused_elf_probe"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$WT/artifacts/gemm_probe_sweep_${TS}.log"
mkdir -p "$WT/artifacts"; : > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }
restart(){ systemctl --user start npu-asr.service voxd.service >/dev/null 2>&1; log "[svc] npu services restarted"; }
beep(){ ( speaker-test -t sine -f 1000 -l 1 >/dev/null 2>&1 & local p=$!; sleep 1; kill -9 "$p" >/dev/null 2>&1 ); }
trap 'restart; beep; log "[done] log: $LOG"' EXIT

log "================ GEMM batching probe sweep  $TS ================"
log "host: $(uname -srm)   N set: $NS"

# 1) device-free: ensure ELFs exist
for N in $NS; do
  [ -f "$WT/artifacts/gemm_probe${SUF}_N${N}/gemmprobe.elf" ] || { log "[build] gemm_probe${SUF}_N${N} missing -> building (cols=$NUM_COLS)"; NUM_COLS="$NUM_COLS" bash "$WT/scripts/build_gemm_probe.sh" "$N" >>"$LOG" 2>&1; }
done

# 2) build the probe host bin
log "[build] fused_elf_probe (release)"
( cd "$WT/rust" && cargo build --release -p npu-asr --bin fused_elf_probe ) >>"$LOG" 2>&1 || { log "FATAL: probe build failed"; exit 1; }

# 3) claim the single-tenant NPU
log "[svc] stopping npu-asr / voxd"
systemctl --user stop npu-asr.service voxd.service >/dev/null 2>&1
sleep 1
if fuser /dev/accel/accel0 >/dev/null 2>&1; then
  log "FATAL: /dev/accel/accel0 still busy — another session holds the NPU. Aborting (serialize, see [[npu-timing-check-fuser-first]])."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"
  exit 1
fi
log "[svc] device clear"

# 4) sweep
log ""
log "  N | weightMB | dispatch_ms | per-tok ms (=disp/N) | rel-L2 | gate"
log "  --+----------+-------------+----------------------+--------+-----"
for N in $NS; do
  OUT="$WT/artifacts/gemm_probe${SUF}_N${N}"
  res="$(cd "$WT" && FUSED_TIME=1 LD_LIBRARY_PATH=$LDLIB "$PROBE" "$OUT" 2>&1)"
  echo "$res" >>"$LOG"
  disp="$(echo "$res" | sed -n 's/.*dispatch alone (1 NPU dispatch): *\([0-9.]*\) ms.*/\1/p' | head -1)"
  rel="$(echo "$res"  | sed -n 's/.*rel-L2 = \([0-9.]*\).*/\1/p' | head -1)"
  gate="$(echo "$res" | grep -q 'PASS' && echo PASS || echo FAIL)"
  pt="$(awk -v d="${disp:-0}" -v n="$N" 'BEGIN{ if(n>0 && d!="") printf "%.4f", d/n; else print "?" }')"
  log "$(printf '  %3s| %8s | %11s | %20s | %6s | %s' "$N" "4.72" "${disp:-?}" "$pt" "${rel:-?}" "$gate")"
done
log ""
log "READ: per-tok ms falling as N grows => weight read amortises => GO. Flat => KILL."
log "(All N share ONE 4.72 MB weight read; X/out grow with N. Compare per-tok across rows.)"
