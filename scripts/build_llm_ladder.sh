#!/usr/bin/env bash
# Build a resident WINDOW LADDER for a decoder LLM: N decode ELFs at different attention windows
# over ONE KV allocation, which `scenarios/generate-*-ladder.toml` then names in `decode_ladder`.
#
#   bash scripts/build_llm_ladder.sh <spec> <alloc> <window>...
#   bash scripts/build_llm_ladder.sh qwen3-0.6b 4096 256 512 1024 1536 2048 4096
#
# Compile-only, no NPU needed. ~1-2 min per arm.
#
# WHY THE TWO ENV FLAGS BELOW ARE BOTH REQUIRED, and what breaks without each:
#
#   KV_ALLOC=<alloc>          sizes the KV cache for the CAPACITY while each arm computes over its
#                             own narrower WINDOW. Without it every arm allocates its own window's
#                             worth of cache and the arms hold different amounts of context.
#
#   LADDER_SCRATCH_ORDER=1    pins the persistent buffers (weights + kc/vc) to the FRONT of the
#                             scratch arena. Without it the window-dependent softmax scratch
#                             (`sc`/`sw`, Hq*S each) sits ahead of them and shifts everything after
#                             it: measured 2026-09-09, arms at window 256 and 512 over one
#                             allocation disagreed on 304 of 313 named offsets, and the engine's
#                             `check_shared_layout_agrees` correctly refuses to bind them to one
#                             arena. With it, all six arms agree on 310 persistent buffers exactly.
#
# Needs an IRON carrying `scratch_order` (iron/common/sequence.py). It is on integration-stack as
# of 2026-09-09; an older checkout fails with an unexpected-keyword TypeError naming it.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC="${1:?usage: build_llm_ladder.sh <spec> <alloc> <window>...}"
ALLOC="${2:?usage: build_llm_ladder.sh <spec> <alloc> <window>...}"
shift 2
[ "$#" -ge 1 ] || { echo "ERROR: name at least one window" >&2; exit 2; }

OUTBASE="${LADDER_OUT:-$REPO/artifacts/$SPEC/ladder$ALLOC}"
echo "[ladder] spec=$SPEC alloc=$ALLOC windows=$* -> $OUTBASE"

for W in "$@"; do
    [ "$W" -le "$ALLOC" ] || { echo "ERROR: window $W exceeds the allocation $ALLOC" >&2; exit 2; }
    OUT="$OUTBASE/w$W"
    if [ -f "$OUT/meta.json" ]; then
        echo "[ladder] w$W already built, skipping"
        continue
    fi
    echo "[ladder] building window=$W"
    # Sequential and CHECKED, not a parallel fan-out: aiecc is memory-hungry and gets OOM-killed
    # under pressure on this box, which surfaces as `Compilation failed with exit code 1` and reads
    # exactly like a capability limit. If an arm fails here, rebuild it alone before believing the
    # window is unbuildable.
    KV_ALLOC="$ALLOC" LADDER_SCRATCH_ORDER=1 GEN_EXTRA="--max-seq $W" \
        bash "$REPO/scripts/build_llm_decode.sh" "$SPEC" "" "$OUT"
    [ -f "$OUT/meta.json" ] || { echo "ERROR: w$W produced no meta.json" >&2; exit 1; }
done

echo "[ladder] verifying every arm agrees on the persistent arena layout"
python3 - "$OUTBASE" <<'PY'
import json, pathlib, sys
base = pathlib.Path(sys.argv[1])
arms = {}
for m in sorted(base.glob("w*/meta.json")):
    d = json.load(open(m))
    arms[d["dims"]["S"]] = d          # keyed by the artifact's OWN window, never the dir name
if len(arms) < 2:
    print(f"[ladder] {len(arms)} arm(s); nothing to cross-check"); sys.exit(0)
ws = sorted(arms)
ref = arms[ws[0]]
persistent = set(ref["weights"]) | set(ref["cache_buffers"])
bad = [(w, n) for w in ws[1:] for n in persistent
       if arms[w]["layout"].get(n) != ref["layout"].get(n)]
if bad:
    print(f"[ladder] FAIL: {len(bad)} persistent buffer(s) disagree, e.g. {bad[:3]}")
    print("[ladder] the arms cannot share an arena -- was LADDER_SCRATCH_ORDER=1 set for ALL of them?")
    sys.exit(1)
print(f"[ladder] OK: windows {ws} agree on {len(persistent)} persistent buffers")
print(f"[ladder] arena must be sized for the widest: {max(a['scratch_size'] for a in arms.values())} bytes")
PY
echo "[ladder] done"
