#!/usr/bin/env bash
# Phase-1a clean re-run: host-glue bandwidth-contention probe, AFTER the mha softmax vectorization
# (commit 32e2b46). CPU-ONLY — no NPU, no service stop needed. Run on an IDLE machine.
#
#   bash scripts/run_glue_contention.sh
#
# Then send the logfile printed at the end (/tmp/glue_contention_clean.log) for review.

set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
LOG=/tmp/glue_contention_clean.log
REPS="${REPS:-80}"
RUNS="${RUNS:-3}"

# tee everything (stdout+stderr) to the log
exec > >(tee "$LOG") 2>&1

echo "################ glue_contention clean run ################"
echo "date:   $(date)"
echo "host:   $(uname -srm)"
echo "cores:  $(nproc)"
echo "REPS=$REPS  RUNS=$RUNS"
echo
echo "==== machine cleanliness BEFORE (want low load, no music/other sessions) ===="
uptime
ps -eo pid,pcpu,comm --sort=-pcpu | head -8
echo
echo "==== git state (confirm mha fix 32e2b46 present) ===="
git -C "$R" rev-parse --short HEAD
git -C "$R" log --oneline -3
if git -C "$R" merge-base --is-ancestor 32e2b46 HEAD 2>/dev/null; then
  echo "OK: mha vectorized-softmax fix (32e2b46) IS in this build"
else
  echo "WARNING: mha fix 32e2b46 NOT found in history — results would be pre-fix!"
fi
echo
echo "==== build (release, CPU-only crate) ===="
( cd "$R/rust" && cargo build --release -p npu-asr-host --bin glue_contention 2>&1 | tail -3 )
echo
BIN="$R/rust/target/release/glue_contention"
for i in $(seq 1 "$RUNS"); do
  echo "================== RUN $i / $RUNS  (REPS=$REPS) =================="
  REPS="$REPS" "$BIN"
  echo
done
echo "==== machine cleanliness AFTER ===="
uptime
echo
echo "################ DONE — save this file: $LOG ################"
