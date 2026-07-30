#!/usr/bin/env bash
# check_no_private_refs.sh -- audit the PUBLIC xdna-engine tree for anything that
# points at private material (the private repo, its KB, its tooling, or a local
# dev-workspace path). The rule: private data lives in the private repo, and NO
# public file may LINK to it -- not by repo name, not by KB path, not by tooling
# name, not by absolute dev path.
#
# Usage:
#   scripts/check_no_private_refs.sh            # scan the whole tracked tree
#   scripts/check_no_private_refs.sh <files...> # scan just these files (hook mode)
#
# Exit 0 = clean; exit 1 = found a private reference (prints file:line).
# Wire it into the pre-push guard (hooks/pre-push) so a leak cannot be pushed.
set -euo pipefail

# Patterns that mean "this public file references private material".
#  - the private repo by name
#  - private-KB structural paths (the private journal's docs/ subtree + its dirs)
#  - private tooling scripts (kb.sh / tasks.sh live only in the private journal)
#  - a local dev-workspace absolute path (leaks the layout that houses the private repo)
patterns=(
  'xdna-engine-private'
  'docs/(handoffs|tasks|log|kb|reference|archive|research|superpowers)/'
  '(^|[^A-Za-z0-9_./-])(journal|strategy|internal)/'
  '(^|[^A-Za-z0-9_])(kb|tasks)\.sh([^A-Za-z0-9]|$)'
  '~/repositories/|xdna-engine-workspace|/home/[a-z]'
)
regex="$(IFS='|'; echo "${patterns[*]}")"

# Wikilink pattern -- the private KB's own cross-reference syntax is `[[slug]]`
# (see xdna-engine-private/journal/scripts/kb.sh). ANY such token in the public tree
# both names a private note to an outside reader and is a dead link for them, so it is
# a leak on its own, independent of the path/name patterns above.
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
wikilink_re='\[\[[a-z0-9][a-z0-9-]*\]\]'
benign_wikilink_re='\[\[(bin|model|last)\]\]'

# Files that are ALLOWED to name these patterns (the guards themselves + the ignore list).
allow='^(scripts/check_no_private_refs\.sh|hooks/pre-push|\.githooks-install\.md|\.gitignore)$'

cd "$(git rev-parse --show-toplevel)"

if [ "$#" -gt 0 ]; then
  files=()
  for f in "$@"; do [[ "$f" =~ $allow ]] || files+=("$f"); done
  [ "${#files[@]}" -eq 0 ] && exit 0
  hits="$(git grep -nIEi "$regex" -- "${files[@]}" 2>/dev/null || true)"
  wiki_hits="$(git grep -nIE "$wikilink_re" -- "${files[@]}" 2>/dev/null | grep -viE "$benign_wikilink_re" || true)"
else
  # whole tree, minus the allowed guard files and lockfiles
  hits="$(git grep -nIEi "$regex" -- . ':!*.lock' \
            ':!scripts/check_no_private_refs.sh' ':!hooks/pre-push' \
            ':!.githooks-install.md' ':!.gitignore' 2>/dev/null || true)"
  wiki_hits="$(git grep -nIE "$wikilink_re" -- . ':!*.lock' \
            ':!scripts/check_no_private_refs.sh' ':!hooks/pre-push' \
            ':!.githooks-install.md' ':!.gitignore' 2>/dev/null | grep -viE "$benign_wikilink_re" || true)"
fi

if [ -n "$wiki_hits" ]; then
  hits="$(printf '%s\n%s' "$hits" "$wiki_hits" | sed '/^$/d')"
fi

if [ -n "$hits" ]; then
  echo "PRIVATE REFERENCE(S) found in the public tree -- move the data private and unlink:" >&2
  printf '%s\n' "$hits" >&2
  echo "" >&2
  echo "Rule: never link a public file to private material. Genericize the reference" >&2
  echo "(describe the fact in-tree) or drop it. See hooks/pre-push." >&2
  exit 1
fi
exit 0
