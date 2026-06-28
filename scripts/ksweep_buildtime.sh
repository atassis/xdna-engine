#!/usr/bin/env bash
# K-sweep: build the fused decode at several layer counts with the fully-optimized
# the toolchain_up instance aiecc, recording end-to-end (gen+aiecc) wall + per-phase + ELF sha.
# Build-only (engine-only), no NPU. K>1 has no canonical byte-gate (GATE INFO).
#   bash scripts/ksweep_buildtime.sh "1 4 6 8 12"
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KS="${1:-1 4 6 8 12}"
SUMMARY="$REPO/artifacts/ksweep_summary.txt"
: > "$SUMMARY"
echo "K-sweep (optimized aiecc, engine-only B=128) — $(printf '%s' "$KS")" | tee -a "$SUMMARY"
for K in $KS; do
  D="/tmp/ksweep_K${K}"
  echo "=== building K=$K -> $D ===" | tee -a "$SUMMARY"
  # bound each build (K=12 is the largest); 50 min ceiling
  if NL="$K" FROZEN_DIR="$D" timeout 3000 bash "$REPO/scripts/gate_freeze_and_build.sh" > "/tmp/ksweep_K${K}.log" 2>&1; then
    wall=$(grep -oE "gen\+build wall: [0-9]+ s" "/tmp/ksweep_K${K}.log" | grep -oE "[0-9]+")
    sha=$(grep -oE "ELF sha256=[0-9a-f]+" "/tmp/ksweep_K${K}.log" | cut -d= -f2)
    npu=$(grep "npu-insts" "$D/phase_timers.log" 2>/dev/null | grep -v "+=" | grep -oE ": [0-9.]+ s" | grep -oE "[0-9.]+" | head -1)
    pc=$(grep "per-core-compile" "$D/phase_timers.log" 2>/dev/null | grep -v "+=" | grep -oE ": [0-9.]+ s" | grep -oE "[0-9.]+" | head -1)
    ts=$(grep "timed-sum" "$D/phase_timers.log" 2>/dev/null | grep -oE ": [0-9.]+ s" | grep -oE "[0-9.]+" | head -1)
    echo "K=$K  gen+build=${wall}s  aiecc-timed-sum=${ts}s  npu-insts=${npu}s  per-core=${pc}s  sha=${sha:0:12}" | tee -a "$SUMMARY"
  else
    echo "K=$K  BUILD FAILED/TIMEOUT (see /tmp/ksweep_K${K}.log)" | tee -a "$SUMMARY"
  fi
  rm -rf "$D"   # free disk between K (each work/ is large)
done
echo "=== K-sweep done ===" | tee -a "$SUMMARY"
