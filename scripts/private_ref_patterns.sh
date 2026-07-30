# private_ref_patterns.sh -- SINGLE SOURCE OF TRUTH for the "does this text point at
# private material" regexes. Sourced by:
#   - scripts/check_no_private_refs.sh   (working-tree file content, and --message mode
#     for arbitrary text such as a commit message)
#   - hooks/pre-push                     (via check_no_private_refs.sh --message, once
#     per commit in the range being pushed)
#
# Do NOT copy these patterns into another file. Two copies drift, and the drift is a
# silent hole (this is exactly the gap leak-gate-scan-commit-messages closed: the
# 2026-07-26 violation was a commit message, which no gate scanned at all). Grow the
# vocabulary here once and both the content path and the message path pick it up.
#
# Not a standalone script: no shebang, no `set -e` (would bind the sourcing shell).
# Callers already run under `set -euo pipefail`.

private_ref_patterns=(
  'xdna-engine-private'
  'docs/(handoffs|tasks|log|kb|reference|archive|research|superpowers)/'
  '(^|[^A-Za-z0-9_./-])(journal|strategy|internal)/'
  '(^|[^A-Za-z0-9_])(kb|tasks)\.sh([^A-Za-z0-9]|$)'
  '~/repositories/|xdna-engine-workspace|/home/[a-z]'
)
private_ref_regex="$(IFS='|'; echo "${private_ref_patterns[*]}")"

# Wikilink pattern -- the private KB cross-references its own notes as `[[slug]]`. ANY
# such token in the public tree both names a private note to an outside reader and is a
# dead link for them, so it is a leak on its own, independent of the path/name patterns
# above.
#
# Anchored on the slug shape (`[a-z0-9][a-z0-9-]*` hard against both brackets, no
# inner whitespace) specifically so it does NOT match:
#  - bash `[[ ... ]]` test syntax: bash requires whitespace right inside `[[`/`]]` for
#    it to parse as the test command at all (`[[foo]]` is a syntax error, not a test),
#    so a real test can never satisfy "non-space char touching both brackets".
#  - markdown `[[TOC]]`-style or link syntax that isn't this slug shape.
# Verified against a full sweep of this tree (2026-07-30) rather than assumed: the only
# two NON-slug matches of the slug-shape regex found were Cargo's TOML array-of-tables
# header (`[[bin]]` in Cargo.toml / rust/npu-runtime/src/config.rs's embedded TOML
# example, and prose describing it in bench/test_harness_smoke.py, install.sh) and one
# ONNX I/O shape example (`input_ids=[[last]]`, a nested-array literal, not a slug) in
# rust/npu-engine/src/asr/whisper.rs. Both are excluded by exact token, not by a broad
# heuristic (e.g. "require a hyphen") -- the private KB also has legitimate single-word
# slugs (`design`, `approaches`), so a hyphen-required pattern would silently stop
# catching a real single-word slug leak. Grow this allowlist only when a NEW verified
# non-KB `[[word]]` use shows up; do not add words defensively.
private_ref_wikilink_re='\[\[[a-z0-9][a-z0-9-]*\]\]'
private_ref_benign_wikilink_re='\[\[(bin|model|last)\]\]'
