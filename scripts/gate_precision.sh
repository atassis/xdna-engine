#!/usr/bin/env bash
# Quality gate for a precision plan, host-side, no device.
#
#   bash scripts/gate_precision.sh <plan> [TOKENS] [MAX_PPL_PCT]
#     plan          a preset name, inline JSON, or a path -- the SAME value PRECISION takes
#     TOKENS        paired positions (default 2000, the shape the decode_layer_dp gate used)
#     MAX_PPL_PCT   fail if the paired delta's LOWER CI bound exceeds this (default 1.0)
#
# WHY A GATE AT ALL. The shipped correctness gate is tests/refs/qwen3-0.6b/bf16_oracle.json: one
# prompt, eight free-running tokens, judged by argmax, with two steps at 0.0203 and 0.154 logit
# margins. It cannot see a format that shifts every logit slightly, and rel-L2 is a note rather
# than a blocker by standing policy -- so before this, a silent precision regression tripped
# nothing we run. Floor-vs-nearest-even rounding once cost 1.3x accuracy and passed every gate.
#
# READ plan_arms.py's header for what this eval CANNOT see -- notably the `kv` site, the core's
# own rounding mode, and trajectory drift. The run prints its own sensitivity; a null result is
# "smaller than that", never "zero".
#
# Needs .venv-export (torch + transformers), deliberately not .venv-iron. ~2 min at 2000 tokens.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
PLAN="${1:?usage: gate_precision.sh <plan> [TOKENS] [MAX_PPL_PCT]}"
TOKENS="${2:-2000}"
MAXPCT="${3:-1.0}"
VENV="${VENV_EXPORT:-$REPO/.venv-export}"
[ -x "$VENV/bin/python" ] || VENV="$WS/xdna-engine/.venv-export"
[ -x "$VENV/bin/python" ] || { echo "ERROR: no export venv (torch+transformers) at $VENV"; exit 1; }
QLAB="${QLAB_WORK:-/mnt/data/xdna/qlab}"
CORPUS="${PRECISION_GATE_CORPUS:-$QLAB/corpora/natural-prose.txt}"
[ -r "$CORPUS" ] || { echo "ERROR: no corpus at $CORPUS (hostlab/make_corpora.py builds them)"; exit 1; }
OUT="${PRECISION_GATE_OUT:-$QLAB/runs/precision-gate.json}"

exec "$VENV/bin/python" "$REPO/designs/decode_fused/hostlab/plan_arms.py" \
    --plan "$PLAN" --corpus "$CORPUS" --tokens "$TOKENS" \
    --max-ppl-pct "$MAXPCT" --out-json "$OUT"
