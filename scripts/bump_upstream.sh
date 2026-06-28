#!/usr/bin/env bash
# Bump our mlir-aie fork integration branch (atassis/mlir-aie:xdna2-asr) onto the latest upstream:
# rebase our 14-patch stack onto Xilinx/mlir-aie, DROP commits that merged upstream (they apply empty),
# and surface conflicts as "rebase this PR". This is how our PR queue stays actualized as PRs land.
#
#   scripts/bump_upstream.sh                 # DRY-RUN (default): preview in a throwaway worktree, no changes
#   scripts/bump_upstream.sh --onto <ref>    # rebase target (default: upstream/main)
#   scripts/bump_upstream.sh --apply         # do it for real: rebase xdna2-asr -> rebuild instance + CPU
#                                            #   smoke -> re-pin toolchain.lock ONLY if green; else abort
#
# Re-pinning toolchain.lock changes the lock hash -> toolchain_up.sh builds a fresh instance from the
# rebased commit; the old instance stays cached. After --apply: device-re-validate, push the
# rebased branch, and delete per-PR branches whose commit dropped (merged upstream).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/toolchain.lock"; set +a
SUB="$REPO/mlir-aie"

APPLY=0; ONTO="upstream/main"
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1 ;;
    --onto)  ONTO="${2:?--onto needs a ref}"; shift ;;
    *) echo "usage: bump_upstream.sh [--apply] [--onto <upstream-ref>]" >&2; exit 2 ;;
  esac; shift
done

# Remotes + fetch.
git -C "$SUB" remote get-url upstream >/dev/null 2>&1 || git -C "$SUB" remote add upstream https://github.com/Xilinx/mlir-aie
git -C "$SUB" remote get-url fork     >/dev/null 2>&1 || git -C "$SUB" remote add fork "${MLIR_AIE_FORK_URL:-https://github.com/atassis/mlir-aie}"
echo "[bump] fetching upstream + fork ..." >&2
git -C "$SUB" fetch -q upstream
git -C "$SUB" cat-file -e "${MLIR_AIE_FORK_COMMIT}^{commit}" 2>/dev/null || git -C "$SUB" fetch -q fork xdna2-asr

BASE="$(git -C "$SUB" merge-base "$MLIR_AIE_FORK_COMMIT" "$ONTO")"
NEWUP="$(git -C "$SUB" rev-parse "$ONTO")"
N_OURS="$(git -C "$SUB" rev-list --count "$BASE..$MLIR_AIE_FORK_COMMIT")"
echo "[bump] our branch tip : $(git -C "$SUB" rev-parse --short "$MLIR_AIE_FORK_COMMIT")  ($N_OURS commits over base)"
echo "[bump] current base   : $(git -C "$SUB" rev-parse --short "$BASE")"
echo "[bump] target ($ONTO): $(git -C "$SUB" rev-parse --short "$NEWUP")"
if [ "$BASE" = "$NEWUP" ]; then echo "[bump] already on the latest $ONTO -- nothing to bump."; exit 0; fi

# Rebase in a THROWAWAY worktree so xdna2-asr (and the live submodule checkout) are untouched until --apply.
WT="$(mktemp -d "${TMPDIR:-/tmp}/bump-wt.XXXXXX")"
cleanup() { git -C "$SUB" worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"; git -C "$SUB" worktree prune 2>/dev/null || true; }
trap cleanup EXIT
git -C "$SUB" worktree add -q --detach "$WT" "$MLIR_AIE_FORK_COMMIT"

echo "[bump] rebasing our $N_OURS commits onto $(git -C "$SUB" rev-parse --short "$NEWUP") ..." >&2
if git -C "$WT" -c rerere.enabled=false rebase --onto "$NEWUP" "$BASE" 2>/tmp/bump-rebase.err; then
  NEWTIP="$(git -C "$WT" rev-parse HEAD)"
  N_NEW="$(git -C "$SUB" rev-list --count "$NEWUP..$NEWTIP")"
  N_DROPPED=$((N_OURS - N_NEW))
  echo "[bump] REBASE CLEAN: $N_NEW commits kept, $N_DROPPED dropped (merged upstream)."
  echo "[bump] --- kept commits (the live PR queue) ---"
  git -C "$WT" log --oneline "$NEWUP..$NEWTIP" | sed 's/^/[bump]   /'
  if [ "$N_DROPPED" -gt 0 ]; then
    echo "[bump] --- dropped (now upstream; delete their per-PR branches) ---"
    comm -23 \
      <(git -C "$SUB" log --format='%s' "$BASE..$MLIR_AIE_FORK_COMMIT" | sort) \
      <(git -C "$WT" log --format='%s' "$NEWUP..$NEWTIP" | sort) | sed 's/^/[bump]   DROPPED: /'
  fi
  if [ "$APPLY" -eq 1 ]; then
    echo "[bump] --apply: moving xdna2-asr to the rebased tip + re-pinning toolchain.lock ..." >&2
    git -C "$SUB" branch -f xdna2-asr "$NEWTIP"
    sed -i "s|^MLIR_AIE_FORK_COMMIT=.*|MLIR_AIE_FORK_COMMIT=$NEWTIP   # rebased onto $(git -C "$SUB" rev-parse --short "$NEWUP")|" "$REPO/toolchain.lock"
    echo "[bump] re-pinned. NEXT: scripts/toolchain_smoke.sh (CPU gate) -> if green, device-revalidate + push xdna2-asr + delete the dropped per-PR branches. If the smoke FAILS, revert toolchain.lock."
  else
    echo "[bump] DRY-RUN: no changes made. Re-run with --apply to move xdna2-asr + re-pin (then smoke + revalidate)."
  fi
else
  CONFLICT="$(git -C "$WT" rev-parse --short REBASE_HEAD 2>/dev/null || echo '?')"
  echo "[bump] CONFLICT -- the rebase stopped (upstream moved under one of our patches)."
  echo "[bump] conflicting commit: $(git -C "$WT" log --oneline -1 REBASE_HEAD 2>/dev/null || echo "$CONFLICT")"
  echo "[bump] => that patch needs a manual rebase (\"rebase this PR\"). Resolve in a real worktree of xdna2-asr."
  git -C "$WT" rebase --abort 2>/dev/null || true
  exit 1
fi
