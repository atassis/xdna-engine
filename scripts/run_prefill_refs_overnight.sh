#!/usr/bin/env bash
# Generates the remaining gemma4-12b Tier-2 prefill references (P=64 and P=255 already exist from
# the 2026-09-14 session). Meant to be run standalone, unattended -- no device, no IRON, CPU only.
#
#   scripts/run_prefill_refs_overnight.sh
#
# Logs to scripts/run_prefill_refs_overnight.log (timestamped, appended). Safe to Ctrl-C and rerun:
# each length is its own subprocess and only writes its .json on success, so a partial run never
# leaves a corrupt file -- rerunning just repeats whatever didn't finish.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG="scripts/run_prefill_refs_overnight.log"
PY="../xdna-engine/.venv-export/bin/python3"
WEIGHTS="/mnt/data/xdna/artifacts/gemma4-12b/weights"
REMAINING_LENGTHS="256,257,512,600,768"

if pgrep -f "make_prefill_refs.py|gate_llm_reference.py" >/dev/null 2>&1; then
  echo "[overnight] another prefill-refs run is already active -- not starting a second one" >&2
  exit 1
fi

{
  echo "=== $(date -Iseconds) starting, lengths=${REMAINING_LENGTHS} tokens=8 ==="
  free -h
  "$PY" scripts/make_prefill_refs.py --spec gemma4-12b --weights "$WEIGHTS" \
    --lengths "$REMAINING_LENGTHS" --tokens 8
  echo "=== $(date -Iseconds) DONE ==="
} >>"$LOG" 2>&1

echo "[overnight] finished -- see $LOG and tests/refs/gemma4-12b/prefill/"
