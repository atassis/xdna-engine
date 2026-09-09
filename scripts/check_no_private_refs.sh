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
#   scripts/check_no_private_refs.sh --message  # scan stdin as free text (a commit
#                                                # message, not a tracked file) --
#                                                # used by hooks/pre-push per commit.
#
# Exit 0 = clean; exit 1 = found a private reference (prints file:line, or the
# matching line(s) of the text in --message mode).
#
# `git grep --untracked` is load-bearing, not a nicety: plain `git grep` searches only
# TRACKED files, so a brand-new public file could carry a private KB wikilink and this
# script would exit 0 on it -- a false all-clear at exactly the moment you would run it
# (before staging). Caught 2026-08-18 on an untracked scripts/*.py. Standard exclusions
# still apply, so .gitignore'd paths stay out.
# Wire it into the pre-push guard (hooks/pre-push) so a leak cannot be pushed.
set -euo pipefail

# Pattern definitions live in ONE place (scripts/private_ref_patterns.sh) so the
# content path (this script) and the commit-message path (this script's --message
# mode, called from hooks/pre-push) can never drift apart.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=private_ref_patterns.sh
source "$script_dir/private_ref_patterns.sh"
patterns=("${private_ref_patterns[@]}")
regex="$private_ref_regex"
wikilink_re="$private_ref_wikilink_re"
benign_wikilink_re="$private_ref_benign_wikilink_re"

# --message mode: scan stdin (a commit message, not a working-tree file) against the
# exact same $regex / $wikilink_re / $benign_wikilink_re as the file-content path
# below. No git grep here -- there is no file, no tree, no line-numbered blob, just
# free text -- so plain `grep` stands in for `git grep`.
if [ "${1:-}" = "--message" ]; then
  text="$(cat)"
  hits="$(printf '%s\n' "$text" | grep -EIi "$regex" || true)"
  wiki_hits="$(printf '%s\n' "$text" | grep -EI "$wikilink_re" | grep -viE "$benign_wikilink_re" || true)"
  if [ -n "$wiki_hits" ]; then
    hits="$(printf '%s\n%s' "$hits" "$wiki_hits" | sed '/^$/d')"
  fi
  if [ -n "$hits" ]; then
    echo "PRIVATE REFERENCE(S) found in commit message text:" >&2
    printf '%s\n' "$hits" >&2
    exit 1
  fi
  exit 0
fi

# Files that are ALLOWED to name these patterns (the guards themselves + the ignore list).
allow='^(scripts/check_no_private_refs\.sh|scripts/private_ref_patterns\.sh|hooks/pre-push|hooks/pre-push-fork|\.githooks-install\.md|\.gitignore)$'

cd "$(git rev-parse --show-toplevel)"

# --rev <committish>: scan THAT TREE instead of the working tree, and drop --untracked
# (a commit has no untracked files). This is what the pre-push tree guard wants: the
# question there is "does the tree I am PUSHING carry a private reference", and answering
# it from the working tree makes one session's UNSAVED edit block another session's push
# of unrelated commits. Measured 2026-09-09: two pushes were refused because a concurrent
# session had an uncommitted `[[slug]]` in gen_llm_decode.py, while the pushed tree was
# clean. The working-tree default stays for every other caller -- catching a brand-new
# unstaged file is exactly why --untracked is load-bearing above.
REV=""
if [ "${1:-}" = "--rev" ]; then REV="${2:?--rev needs a committish}"; shift 2; fi
if [ -n "$REV" ]; then SCAN=("$REV"); UNTRACKED=(); else SCAN=(); UNTRACKED=(--untracked); fi

if [ "$#" -gt 0 ]; then
  files=()
  for f in "$@"; do [[ "$f" =~ $allow ]] || files+=("$f"); done
  [ "${#files[@]}" -eq 0 ] && exit 0
  hits="$(git grep "${UNTRACKED[@]}" -nIEi "$regex" "${SCAN[@]}" -- "${files[@]}" 2>/dev/null || true)"
  wiki_hits="$(git grep "${UNTRACKED[@]}" -nIE "$wikilink_re" "${SCAN[@]}" -- "${files[@]}" 2>/dev/null | grep -viE "$benign_wikilink_re" || true)"
else
  # whole tree, minus the allowed guard files and rust/Cargo.lock, whose generated
  # `[[package]]` headers have the wikilink shape. Exclude that ONE path, never *.lock:
  # toolchain.lock is prose and cites the KB.
  hits="$(git grep "${UNTRACKED[@]}" -nIEi "$regex" "${SCAN[@]}" -- . ':!rust/Cargo.lock' \
            ':!scripts/check_no_private_refs.sh' ':!scripts/private_ref_patterns.sh' \
            ':!hooks/pre-push' ':!hooks/pre-push-fork' ':!.githooks-install.md' ':!.gitignore' 2>/dev/null || true)"
  wiki_hits="$(git grep "${UNTRACKED[@]}" -nIE "$wikilink_re" "${SCAN[@]}" -- . ':!rust/Cargo.lock' \
            ':!scripts/check_no_private_refs.sh' ':!scripts/private_ref_patterns.sh' \
            ':!hooks/pre-push' ':!hooks/pre-push-fork' ':!.githooks-install.md' ':!.gitignore' 2>/dev/null | grep -viE "$benign_wikilink_re" || true)"
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
