#!/usr/bin/env bash
# Fetch the live state of the upstream Peano int->float->int miscompile PR/issue cluster
# (reviews, comments, inline review threads, reactions) across Xilinx/llvm-aie + Xilinx/mlir-aie.
#
# Purpose: a new session can run this ONE command to re-acquire full context on the upstream
# work without hand-querying each PR. Read-only (gh api graphql); no outward action.
#
# Usage:
#   bash scripts/peano_upstream_status.sh                 # full digest + most-recent-activity feed
#   bash scripts/peano_upstream_status.sh --since 2026-06-25T18:00   # flag items newer than <ISO>
#
# Resume launcher with the full narrative: internal notes
# Latest session record: internal notes
set -euo pipefail

SINCE=""
if [[ "${1:-}" == "--since" ]]; then SINCE="${2:-}"; fi

# The cluster. Edit these arrays as the work evolves (add a revert PR, the un-pin PR, etc.).
LLVM_AIE_ITEMS="1053 1054 1051 1056 1050"   # issue(root), fix PR, thomthehound issue, erwei revert issue, mludevid rework PR
MLIR_AIE_ITEMS="3221 3223 3219"             # pin PR (merged), durable pin PR, CI-breakage issue

emit_items() {
  local items="$1" n
  for n in $items; do
    cat <<EOF
    i${n}: issueOrPullRequest(number:${n}) {
      __typename
      ... on Issue { number title state updatedAt
        reactionGroups{content users{totalCount}}
        comments(last:8){nodes{author{login} createdAt body}} }
      ... on PullRequest { number title state updatedAt reviewDecision merged
        reactionGroups{content users{totalCount}}
        reviews(last:10){nodes{author{login} state submittedAt}}
        comments(last:8){nodes{author{login} createdAt body}}
        reviewThreads(last:25){nodes{path line isResolved comments(last:6){nodes{author{login} createdAt body}}}} }
    }
EOF
  done
}

QUERY="query {
  llvmaie: repository(owner:\"Xilinx\", name:\"llvm-aie\") {
$(emit_items "$LLVM_AIE_ITEMS")
  }
  mliraie: repository(owner:\"Xilinx\", name:\"mlir-aie\") {
$(emit_items "$MLIR_AIE_ITEMS")
  }
}"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
gh api graphql -f query="$QUERY" > "$TMP"

SINCE="$SINCE" python3 - "$TMP" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))["data"]
SINCE = os.environ.get("SINCE") or ""
def snip(b, n=200): return (b or "").replace("\n", " ").strip()[:n]
def rx(g):
    s = [f"{x['content']}x{x['users']['totalCount']}" for x in (g or []) if x['users']['totalCount'] > 0]
    return "  reactions[" + ", ".join(s) + "]" if s else ""
def newf(ts): return "  <== NEW" if SINCE and ts and ts >= SINCE else ""

feed = []  # (timestamp, one-line summary) across the whole cluster
def add(ts, line): feed.append((ts or "", line))

print("=" * 78)
print("UPSTREAM PEANO CLUSTER STATUS" + (f"  (flagging activity since {SINCE})" if SINCE else ""))
print("=" * 78)

for repo in ("llvmaie", "mliraie"):
    for key, it in d[repo].items():
        if not it:
            print(f"\n### {key}: (not found / no access)"); continue
        typ = it.get("__typename", "")
        st = it.get("state", "")
        extra = []
        if it.get("merged"): extra.append("MERGED")
        if it.get("reviewDecision"): extra.append(it["reviewDecision"])
        hdr = f"\n### {repo.replace('aie','-aie')} #{it['number']} [{typ} {st} {' '.join(extra)}] {it.get('title','')}"
        print(hdr + rx(it.get("reactionGroups")))
        print(f"  updated: {it.get('updatedAt','')}")
        for r in (it.get("reviews") or {}).get("nodes", []):
            ts = r.get("submittedAt", "")
            print(f"  REVIEW {r['author']['login']} {r['state']} {ts[:16]}{newf(ts)}")
            add(ts, f"#{it['number']} REVIEW {r['author']['login']} {r['state']}")
        for c in (it.get("comments") or {}).get("nodes", []):
            ts = c["createdAt"]
            print(f"  CMT {c['author']['login']} {ts[:16]}{newf(ts)}: {snip(c['body'])}")
            add(ts, f"#{it['number']} comment by {c['author']['login']}")
        for t in (it.get("reviewThreads") or {}).get("nodes", []):
            for c in t["comments"]["nodes"]:
                ts = c["createdAt"]
                tag = newf(ts)
                if tag or not SINCE:  # always show threads when no --since; else only new ones
                    print(f"  THREAD {t['path']}:{t['line']} {c['author']['login']} {ts[:16]}{tag}: {snip(c['body'],150)}")
                if tag: add(ts, f"#{it['number']} thread reply by {c['author']['login']} on {t['path']}")

print("\n" + "=" * 78)
print("MOST RECENT ACTIVITY (newest first)")
print("=" * 78)
for ts, line in sorted(feed, reverse=True)[:14]:
    print(f"  {ts[:16]}  {line}")
PY
