#!/usr/bin/env bash
# End-to-end smoke for the LLM serving surface: /healthz, /v1/models, chat completions buffered and
# streamed, /v1/completions, temperature-0 determinism, the 400 on an unsupported param, and the
# `npu generate` CLI. Opens the NPU, so it is single-tenant: set NPU_LOCK to a serializing wrapper
# (`NPU_LOCK=/path/to/lock.sh queue --`) or make sure nothing else holds /dev/accel/accel0.
#
# Report a result together with the driver it ran against: `cat /sys/module/amdxdna/srcversion`.
set -uo pipefail

# Every path is derived from this script's own location or overridable. It was written in a session
# scratchpad and then in a worktree, and both spellings named a directory that no longer exists.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." >/dev/null 2>&1 && pwd)"
SMOKE_DIR="${SMOKE_DIR:-$(mktemp -d -t llm-serve-smoke-XXXXXX)}"
# Serializing wrapper for device access, as an argv PREFIX. Empty runs unserialized, which is fine
# on an idle box and wrong on a shared one -- hence the warning rather than a silent default.
read -r -a NPU_LOCK_ARGV <<< "${NPU_LOCK:-}"
[ "${#NPU_LOCK_ARGV[@]}" -gt 0 ] || echo "WARN: NPU_LOCK unset -- running unserialized on a single-tenant device" >&2
# The engine config template ships beside this script; the scenario it names must point at a
# decode artifact built against the CURRENT toolchain pin. A stale artifact will now be REFUSED
# at load by LlmArtifact's freshness gate rather than silently answering with a wrong token.
ENGINE_TOML="${ENGINE_TOML:-$SCRIPT_DIR/llm_serve_smoke.engine.toml}"
CFG="$SMOKE_DIR/engine.toml"
mkdir -p "$SMOKE_DIR"
[ -f "$ENGINE_TOML" ] || { echo "FATAL: no engine config template at $ENGINE_TOML" >&2; exit 1; }
cp "$ENGINE_TOML" "$CFG"
PORT="${PORT:-18434}"
# Release first: it is what install.sh ships, so a debug-only smoke tests a binary nobody runs.
BIN="${BIN:-}"
if [ -z "$BIN" ]; then
  for cand in "$REPO/rust/target/release/npu" "$REPO/rust/target/debug/npu"; do
    [ -x "$cand" ] && BIN="$cand" && break
  done
fi
[ -x "$BIN" ] || { echo "FATAL: no npu binary -- build it, or set BIN=" >&2; exit 1; }
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$HOME/.local/lib/xdna-engine}"
# The repo IS the engine root here: the smoke config names its scenario root-relative, exactly as
# the installed engine.toml does, so scenarios/ and artifacts/ resolve out of this checkout.
export XDNA_ENGINE_ROOT="${XDNA_ENGINE_ROOT:-$REPO}"

TRANSCRIPT="$SMOKE_DIR/transcript.txt"
: > "$TRANSCRIPT"
log() { echo "$@" | tee -a "$TRANSCRIPT"; }
record() { echo "--- $1 ---" >> "$TRANSCRIPT"; shift; "$@" | tee -a "$TRANSCRIPT"; echo >> "$TRANSCRIPT"; }

echo "amdxdna srcversion: $(cat /sys/module/amdxdna/srcversion 2>/dev/null)" | tee -a "$TRANSCRIPT"

log "== starting npu serve =="
"${NPU_LOCK_ARGV[@]}" "$BIN" --config "$CFG" serve --port "$PORT" > "$SMOKE_DIR/serve.log" 2>&1 &
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
# Compare only AFTER establishing both are real completions. Two identical error bodies are
# byte-identical too, and reporting that as PASS is a check that never reached its subject.
# Compare the COMPLETION, not the envelope. `id` and `created` are per-request identity fields, so
# whole-body equality can never hold -- the same mistake as hashing an xclbin whose UUID is stamped
# per build. And check a completion exists first: two identical error bodies are byte-identical too.
C1=$(printf '%s' "$R1" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || true)
C2=$(printf '%s' "$R2" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || true)
if [ -z "$C1" ]; then
  log "FAIL: determinism arm got no completion at all -- response was: $R1"
elif [ "$C1" = "$C2" ]; then log "PASS: temperature-0 completions identical"
else log "FAIL: two temperature-0 completions differ"; log "  R1: $C1"; log "  R2: $C2"; fi

record "unsupported n:2 -> expect 400" curl -s -o /dev/null -w "%{http_code}\n" \
  "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"hi"}],"n":2}'

log "== stopping server =="
kill "$SERVE_PID" 2>/dev/null
wait "$SERVE_PID" 2>/dev/null

log "== npu generate CLI (one-shot, streams to stdout) =="
"${NPU_LOCK_ARGV[@]}" "$BIN" --config "$CFG" generate "Say hello in one short sentence." --max-tokens 32 --temperature 0 \
  2>&1 | tee -a "$TRANSCRIPT"

log "== done, transcript at $TRANSCRIPT =="
