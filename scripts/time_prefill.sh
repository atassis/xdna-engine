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
# Auto-ingests the batched-vs-per-token CROSSOVER into $PRE/meta.json afterward
# (scripts/ingest_prefill_break_even.py) unless NO_INGEST=1 -- see that script's own header for
# why this exists: a hand-edited Rust constant for this number went stale the first time an
# artifact rebuilt without a matching re-sweep (2026-09-11). Include at least one length under
# ~30 tokens in $LENS to actually exercise the region a wrong crossover would bite in; the
# ingest script itself refuses to write (loud, not a guess) if either arm is not flat.
#
# Single-tenant: stop `npu serve` (pkill -x npu, never -f) and run under npu_lock.sh.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
ROUNDS="${1:-2}"; REPS="${2:-2}"; LENS="${3:-256,512,1024}"
DEC="${DECODE_ART:-$WS/xdna-engine/artifacts/qwen3-0.6b/decode}"
PRE="${PREFILL_ART:-/mnt/data/xdna/scratch/prefill/full_l28_m256_s2048}"
OUT="${TIME_OUT:-/mnt/data/xdna/scratch/prefill/timing}"
# Ask cargo where it puts binaries: rust/.cargo/config.toml redirects target-dir off /home, and a
# stale rust/target/ directory survives there, so a hardcoded path finds a directory and no binary.
TGT="$(cd "$REPO/rust" && cargo metadata --format-version 1 --no-deps 2>/dev/null \
       | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])' 2>/dev/null)"
BIN="${TGT:-$REPO/rust/target}/release/prefill_time_probe"
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
if [ "${NO_INGEST:-0}" != "1" ]; then
  python3 "$REPO/scripts/ingest_prefill_break_even.py" --timing-dir "$OUT" --prefill-art "$PRE"
fi
