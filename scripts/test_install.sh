#!/usr/bin/env bash
# Verify the installed npu engine service works end-to-end: health, model list, and a real
# transcription of a sample clip through /v1/audio/transcriptions. Run after scripts/install.sh.
#
# (On a fresh machine the model artifacts under artifacts/ must already be generated; this checks the
# DEPLOYED models actually run on the NPU - the meaningful "do the models work" gate.)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${NPU_PORT:-11434}"
BASE="http://127.0.0.1:$PORT"
CLIP="${1:-$REPO/artifacts/wer_clips/ru_01.wav}"

echo "==> waiting for $BASE/healthz"
for i in $(seq 1 60); do
  if curl -fsS "$BASE/healthz" >/dev/null 2>&1; then break; fi
  sleep 1
  if [ "$i" = 60 ]; then echo "!! service not responding on :$PORT" >&2; exit 1; fi
done
echo "    health: $(curl -fsS "$BASE/healthz")"

echo "==> models:"
curl -fsS "$BASE/v1/models"; echo

echo "==> transcribing $CLIP"
[ -f "$CLIP" ] || { echo "!! sample clip not found: $CLIP (pass one as arg)" >&2; exit 1; }
resp="$(curl -fsS -F "file=@$CLIP" "$BASE/v1/audio/transcriptions")"
echo "    response: $resp"

# crude check: a non-empty "text" field
text="$(printf '%s' "$resp" | sed -n 's/.*"text" *: *"\([^"]*\)".*/\1/p')"
if [ -n "$text" ]; then
  echo "==> PASS - transcription returned: \"$text\""
else
  echo "!! FAIL - empty/absent text in response" >&2
  exit 1
fi
