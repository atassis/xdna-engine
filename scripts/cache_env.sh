#!/usr/bin/env bash
# cache_env.sh -- single relocatable anchor for the build CACHE (toolchain instances,
# fetched MLIR distro, ccache, worktrees, goldens). Mirrors amd_paths.sh but for
# regenerable BUILD ARTIFACTS, which live INSIDE THE REPO (not ~/.cache and not the
# umbrella workspace) so a bare `git clone` of this repo is buildable on its own --
# nothing surprising left in the system, and no sibling checkout implied.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/cache_env.sh"
#   ... use "$XDNA_CACHE"/{mlir-distro,instances,ccache,goldens,...}
#
# Two ways to point it elsewhere, and the SECOND is how this box shares one 30 GB cache
# across the umbrella workspace instead of duplicating it per checkout:
#   1. export XDNA_CACHE=... (e.g. a fast scratch disk), or
#   2. make <repo>/.cache a symlink -- it is gitignored, so it never ships, and it needs
#      no cooperation from any caller: scripts, agent sessions, systemd units and cron all
#      pick it up. An exported var has to be set in every one of those, which is the
#      failure toolchain.lock's IRON_FORK_COMMIT comment documents ("resolved to whatever
#      the shared checkout happened to be on, and NOTHING recorded that").
# CAUTION with (2): `rm -rf .cache/` WITH the trailing slash follows the link and wipes the
# shared cache. Without it you only remove the link.

XDNA_WS="${XDNA_WS:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
export XDNA_WS
XDNA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
# A linked WORKTREE inherits the main checkout's cache. `.cache` is gitignored, so `git worktree
# add` never creates the symlink (2) above describes -- and the `mkdir -p` below would then mint a
# private EMPTY cache, whose visible failure is "MLIR distro not provisioned" and whose invisible
# one is a full toolchain rebuild into a directory nothing else will ever find. A worktree's `.git`
# is a FILE naming an admin dir under the main checkout, which is what makes the main root
# recoverable here.
if [ -z "${XDNA_CACHE:-}" ] && [ ! -e "$XDNA_REPO/.cache" ] && [ -f "$XDNA_REPO/.git" ]; then
  _ce_main="$(git -C "$XDNA_REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
  [ -n "$_ce_main" ] && [ -d "${_ce_main%/.git}/.cache" ] && XDNA_CACHE="${_ce_main%/.git}/.cache"
  unset _ce_main
fi
export XDNA_CACHE="${XDNA_CACHE:-$XDNA_REPO/.cache}"
mkdir -p "$XDNA_CACHE" 2>/dev/null || true
