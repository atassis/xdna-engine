#!/usr/bin/env bash
# Time every arm of a GEMM tile sweep on the NPU, then write the winners into gemm_tiles.json.
# THE DEVICE HALF -- scripts/sweep_gemm_tiles.sh builds the arms without ever touching /dev/accel.
#
#   bash scripts/time_gemm_tiles.sh <sweep-dir>/manifest.json
#   PREFER=accuracy bash scripts/time_gemm_tiles.sh <manifest>   # winner rule at ingest
#   NO_INGEST=1     bash scripts/time_gemm_tiles.sh <manifest>   # timings only, registry untouched
#
# Single-tenant NPU: quiesces npu-asr/voxd, asserts the device is free, ALWAYS restarts on exit.
# The loop itself lives in designs/decode_fused/sweep_gemm_tiles.py --time-manifest, which gates on
# a PINNED power mode before the first dispatch and interleaves the arms across shapes so a drifting
# box drifts across the whole set rather than across one shape.
set -u
. "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_npu_services.sh" || exit 1
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$WT"
MANIFEST="${1:?usage: time_gemm_tiles.sh <sweep-dir>/manifest.json}"
[ -f "$MANIFEST" ] || { echo "ERROR: no manifest at $MANIFEST"; exit 1; }
LOG="$(dirname "$MANIFEST")/time.log"
PROBE="$WT/rust/target/release/fused_elf_probe"
LDLIB="${LDLIB:-$HOME/.local/lib/npu-asr}"
: > "$LOG"
log(){ echo -e "$*" | tee -a "$LOG"; }
trap 'npu_svc_start; log "[done] log: $LOG"' EXIT

log "[build] fused_elf_probe (release)"
( cd "$WT/rust" && cargo build --release -p npu-probes --bin fused_elf_probe ) >>"$LOG" 2>&1 \
  || { log "FATAL: probe build failed"; exit 1; }

log "[svc] quiescing (single-tenant NPU)"
npu_svc_stop || exit 1
pkill -f parakeet_serve >/dev/null 2>&1 || true
sleep 1
if fuser /dev/accel/accel0 >/dev/null 2>&1; then
  log "FATAL: /dev/accel/accel0 busy -- another session holds the NPU."
  fuser -v /dev/accel/accel0 2>&1 | tee -a "$LOG"
  exit 1
fi
log "[svc] device clear"

python3 "$WT/designs/decode_fused/sweep_gemm_tiles.py" \
    --time-manifest "$MANIFEST" --probe "$PROBE" --ld-library-path "$LDLIB" \
    --warmup "${WARMUP:-20}" --iters "${ITERS:-200}" --prefer "${PREFER:-time}" \
    ${NO_INGEST:+--no-ingest} 2>&1 | tee -a "$LOG"
