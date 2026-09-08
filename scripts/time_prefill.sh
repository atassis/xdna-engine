#!/usr/bin/env bash
# Whole-stack prefill timing, batched vs per-token, ALTERNATED.
#
# The rail has never had this number -- the ~90x in the batched-prefill architecture note is a cost
# model. The two arms are two processes (the env flag resolves once per process), interleaved
# round by round, because this box drifts ~33% over hours and an A-then-B ordering measures the
# drift as well as the change. Paired medians per prompt length; the DELTA is the result.
#
#   bash scripts/time_prefill.sh [rounds] [reps] [lens]
#
# Single-tenant: stop `npu serve` (pkill -x npu, never -f) and run under npu_lock.sh.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
ROUNDS="${1:-2}"; REPS="${2:-2}"; LENS="${3:-256,512,1024}"
DEC="${DECODE_ART:-$WS/xdna-engine/artifacts/qwen3-0.6b/decode}"
PRE="${PREFILL_ART:-/mnt/data/xdna-scratch/prefill/full_l28_m256_s2048}"
OUT="${TIME_OUT:-/mnt/data/xdna-scratch/prefill/timing}"
BIN="$REPO/rust/target/release/prefill_time_probe"
[ -x "$BIN" ] || { echo "ERROR: build it first: cargo build --release -p npu-probes --bin prefill_time_probe"; exit 2; }
mkdir -p "$OUT"
echo "[time] decode=$DEC"; echo "[time] prefill=$PRE"; echo "[time] rounds=$ROUNDS reps=$REPS lens=$LENS"
for r in $(seq 1 "$ROUNDS"); do
  for arm in 0 1; do
    echo "--- round $r arm NPU_LLM_PREFILL_BATCHED=$arm"
    NPU_LLM_PREFILL_BATCHED=$arm "$BIN" "$DEC" "$PRE" --reps "$REPS" --lens "$LENS" \
      2>&1 | tee "$OUT/r${r}_arm${arm}.log"
  done
done
echo "[time] logs in $OUT"
