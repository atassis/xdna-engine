#!/usr/bin/env bash
# ESM e2e latency via engine_serve (RELEASE, idle). Args: scenario port label
set -u
export LD_LIBRARY_PATH=~/.local/lib/npu-asr:${LD_LIBRARY_PATH:-}
cd "$(dirname "$0")/.."
SCEN="$1"; PORT="$2"; LABEL="$3"
PROT="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
LOG="/tmp/esm_sv_${PORT}.log"
./rust/target/release/engine_serve "$SCEN" "$PORT" >"$LOG" 2>&1 &
PID=$!
i=0; until grep -q "ready on" "$LOG" 2>/dev/null || ! kill -0 "$PID" 2>/dev/null || [ $i -ge 180 ]; do i=$((i+1)); sleep 0.5; done
if grep -q "ready on" "$LOG" 2>/dev/null; then
  RESP=$(curl -s localhost:$PORT/v1/embeddings -d "{\"input\":\"$PROT\"}")
  NF=$(echo "$RESP" | grep -o '[-0-9.eE]\+' | wc -l)
  echo "[$LABEL] embedding floats ~$NF (expect ~dim)"
  curl -s -o /dev/null localhost:$PORT/v1/embeddings -d "{\"input\":\"$PROT\"}"   # warm
  echo "[$LABEL] 8 timed POSTs (time_total s):"
  for n in $(seq 1 8); do
    curl -s -o /dev/null -w "  %{time_total}\n" localhost:$PORT/v1/embeddings -d "{\"input\":\"$PROT\"}"
  done
else
  echo "[$LABEL] NOT READY"; tail -20 "$LOG"
fi
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
