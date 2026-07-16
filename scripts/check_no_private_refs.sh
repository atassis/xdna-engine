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

# Files that are ALLOWED to name these patterns (the guards themselves + the ignore list).
allow='^(scripts/check_no_private_refs\.sh|hooks/pre-push|\.githooks-install\.md|\.gitignore)$'

cd "$(git rev-parse --show-toplevel)"

if [ "$#" -gt 0 ]; then
  files=()
  for f in "$@"; do [[ "$f" =~ $allow ]] || files+=("$f"); done
  [ "${#files[@]}" -eq 0 ] && exit 0
  hits="$(git grep -nIEi "$regex" -- "${files[@]}" 2>/dev/null || true)"
else
  # whole tree, minus the allowed guard files and lockfiles
  hits="$(git grep -nIEi "$regex" -- . ':!*.lock' \
            ':!scripts/check_no_private_refs.sh' ':!hooks/pre-push' \
            ':!.githooks-install.md' ':!.gitignore' 2>/dev/null || true)"
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
