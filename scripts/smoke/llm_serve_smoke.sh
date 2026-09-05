#!/usr/bin/env bash
# End-to-end smoke for llm-serve-openai-surface. NOT run yet -- device is held for a separate
# owner-gated experiment (patched amdxdna map_wc module) as of 2026-09-05 18:31. Run this only
# after the owner confirms the device is free AND the module is back to stock
# (srcversion 76D3502C9538984CD074984), so the result is reported against the shipped driver.
set -uo pipefail

WORKTREE="<workspace>/wt-llm-serve"
NPU_LOCK="<workspace>/xdna-engine-private/journal/scripts/npu_lock.sh"
# Rescued from a session scratchpad 2026-09-05. Everything that was an absolute path into that
# ephemeral dir is now derived or overridable, so this survives the session that wrote it.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
SMOKE_DIR="${SMOKE_DIR:-$(mktemp -d -t llm-serve-smoke-XXXXXX)}"
# The engine config template ships beside this script; the scenario it names must point at a
# decode artifact built against the CURRENT toolchain pin. A stale artifact will now be REFUSED
# at load by LlmArtifact's freshness gate rather than silently answering with a wrong token.
ENGINE_TOML="${ENGINE_TOML:-$SCRIPT_DIR/llm_serve_smoke.engine.toml}"
CFG="$SMOKE_DIR/engine.toml"
PORT=18434
BIN="$WORKTREE/rust/target/debug/npu"
export LD_LIBRARY_PATH=$HOME/.local/lib/xdna-engine
export XDNA_ENGINE_ROOT="${XDNA_ENGINE_ROOT:-$(cd -- "$REPO/.." >/dev/null 2>&1 && pwd)}"

TRANSCRIPT="$SMOKE_DIR/transcript.txt"
: > "$TRANSCRIPT"
log() { echo "$@" | tee -a "$TRANSCRIPT"; }
record() { echo "--- $1 ---" >> "$TRANSCRIPT"; shift; "$@" | tee -a "$TRANSCRIPT"; echo >> "$TRANSCRIPT"; }

echo "amdxdna srcversion: $(cat /sys/module/amdxdna/srcversion 2>/dev/null)" | tee -a "$TRANSCRIPT"

log "== starting npu serve under npu_lock.sh queue =="
"$NPU_LOCK" queue -- "$BIN" --config "$CFG" serve --port "$PORT" > "$SMOKE_DIR/serve.log" 2>&1 &
SERVE_PID=$!

for i in $(seq 1 90); do
  if curl -s "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then break; fi
  sleep 1
done
if ! curl -s "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  log "FAIL: server did not come up (or exited early)"
  log "--- /healthz (raw, may be empty) ---"; curl -s "http://127.0.0.1:$PORT/healthz" | tee -a "$TRANSCRIPT"
  log "--- serve.log ---"; cat "$SMOKE_DIR/serve.log" | tee -a "$TRANSCRIPT"
  kill "$SERVE_PID" 2>/dev/null; wait "$SERVE_PID" 2>/dev/null
  exit 1
fi
record "/healthz" curl -s "http://127.0.0.1:$PORT/healthz"
record "/v1/models" curl -s "http://127.0.0.1:$PORT/v1/models"

record "chat/completions non-stream" curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":32,"temperature":0}'

log "--- chat/completions stream (raw SSE) ---"
curl -s -N "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"Count from one to three."}],"max_tokens":32,"temperature":0,"stream":true}' \
  | tee -a "$TRANSCRIPT"
echo >> "$TRANSCRIPT"

record "/v1/completions" curl -s "http://127.0.0.1:$PORT/v1/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","prompt":"The capital of France is","max_tokens":8,"temperature":0}'

log "--- temperature:0 determinism (two identical requests) ---"
R1=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16,"temperature":0,"seed":1}')
R2=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16,"temperature":0,"seed":1}')
echo "$R1" >> "$TRANSCRIPT"; echo "$R2" >> "$TRANSCRIPT"
if [ "$R1" = "$R2" ]; then log "PASS: byte-identical"; else log "FAIL: outputs differ"; fi

record "unsupported n:2 -> expect 400" curl -s -o /dev/null -w "%{http_code}\n" \
  "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"hi"}],"n":2}'

log "== stopping server =="
kill "$SERVE_PID" 2>/dev/null
wait "$SERVE_PID" 2>/dev/null

log "== npu generate CLI (one-shot, streams to stdout) =="
"$NPU_LOCK" queue -- "$BIN" --config "$CFG" generate "Say hello in one short sentence." --max-tokens 32 --temperature 0 \
  2>&1 | tee -a "$TRANSCRIPT"

log "== done, transcript at $TRANSCRIPT =="
