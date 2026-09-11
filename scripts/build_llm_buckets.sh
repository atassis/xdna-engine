#!/usr/bin/env bash
# Build resident WINDOW BUCKETS for a decoder LLM: N decode ELFs at different attention windows
# over ONE KV allocation, which `scenarios/generate-*-buckets.toml` then names in `decode_buckets`.
#
#   bash scripts/build_llm_buckets.sh <spec> <alloc> <window>...
#   bash scripts/build_llm_buckets.sh qwen3-0.6b 4096 256 512 1024 1536 2048 4096
#
# Compile-only, no NPU needed. ~1-2 min per bucket.
#
# WHY THE TWO ENV FLAGS BELOW ARE BOTH REQUIRED, and what breaks without each:
#
#   KV_ALLOC=<alloc>          sizes the KV cache for the CAPACITY while each bucket computes over
#                             its own narrower WINDOW. Without it every bucket allocates its own
#                             window's worth of cache and they hold different amounts of context.
#
#   BUCKET_SCRATCH_ORDER=1    pins the persistent buffers (weights + kc/vc) to the FRONT of the
#                             scratch arena. Without it the window-dependent softmax scratch
#                             (`sc`/`sw`, Hq*S each) sits ahead of them and shifts everything after
#                             it: measured 2026-09-09, buckets at window 256 and 512 over one
#                             allocation disagreed on 304 of 313 named offsets, and the engine's
#                             `check_shared_layout_agrees` correctly refuses to bind them to one
#                             arena. With it, all six buckets agree on 310 persistent buffers.
#
# Needs an IRON carrying `scratch_order` (iron/common/sequence.py). It is on integration-stack as
# of 2026-09-09; an older checkout fails with an unexpected-keyword TypeError naming it.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC="${1:?usage: build_llm_buckets.sh <spec> <alloc> <window>...}"
ALLOC="${2:?usage: build_llm_buckets.sh <spec> <alloc> <window>...}"
shift 2
[ "$#" -ge 1 ] || { echo "ERROR: name at least one window" >&2; exit 2; }

OUTBASE="${BUCKETS_OUT:-$REPO/artifacts/$SPEC/buckets$ALLOC}"
echo "[buckets] spec=$SPEC alloc=$ALLOC windows=$* -> $OUTBASE"

for W in "$@"; do
    [ "$W" -le "$ALLOC" ] || { echo "ERROR: window $W exceeds the allocation $ALLOC" >&2; exit 2; }
    OUT="$OUTBASE/w$W"
    if [ -f "$OUT/meta.json" ]; then
        echo "[buckets] w$W already built, skipping"
        continue
    fi
    echo "[buckets] building window=$W"
    # Sequential and CHECKED, not a parallel fan-out: aiecc is memory-hungry and gets OOM-killed
    # under pressure on this box, which surfaces as `Compilation failed with exit code 1` and reads
    # exactly like a capability limit. If a bucket fails here, rebuild it alone before believing
    # the window is unbuildable.
    KV_ALLOC="$ALLOC" BUCKET_SCRATCH_ORDER=1 GEN_EXTRA="--max-seq $W" \
        bash "$REPO/scripts/build_llm_decode.sh" "$SPEC" "" "$OUT"
    [ -f "$OUT/meta.json" ] || { echo "ERROR: w$W produced no meta.json" >&2; exit 1; }
done

echo "[buckets] verifying every bucket agrees on the persistent arena layout"
python3 - "$OUTBASE" <<'PY'
import json, pathlib, sys
base = pathlib.Path(sys.argv[1])
buckets = {}
for m in sorted(base.glob("w*/meta.json")):
    d = json.load(open(m))
    buckets[d["dims"]["S"]] = d       # keyed by the artifact's OWN window, never the dir name
if len(buckets) < 2:
    print(f"[buckets] {len(buckets)} bucket(s); nothing to cross-check"); sys.exit(0)
ws = sorted(buckets)
ref = buckets[ws[0]]
persistent = set(ref["weights"]) | set(ref["cache_buffers"])
bad = [(w, n) for w in ws[1:] for n in persistent
       if buckets[w]["layout"].get(n) != ref["layout"].get(n)]
if bad:
    print(f"[buckets] FAIL: {len(bad)} persistent buffer(s) disagree, e.g. {bad[:3]}")
    print("[buckets] they cannot share an arena -- was BUCKET_SCRATCH_ORDER=1 set for ALL of them?")
    sys.exit(1)
print(f"[buckets] OK: windows {ws} agree on {len(persistent)} persistent buffers")
print(f"[buckets] arena must be sized for the widest: {max(b['scratch_size'] for b in buckets.values())} bytes")
PY
echo "[buckets] done"
