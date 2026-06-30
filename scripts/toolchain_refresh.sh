#!/usr/bin/env bash
# toolchain_refresh.sh -- the "stay on latest upstream" doctor.
#
# Owner principle (internal notes): consume LATEST upstream work;
# never sit on months-stale pins. A pin only stays fresh if you actively BUMP it. This script makes
# staleness VISIBLE and prints the exact next-step commands -- it does NOT auto-rebase or force-push,
# because (a) rebase conflicts need human judgment (upstream removes things we depend on -- placer,
# makefile-common kernel-build) and (b) adopting needs the device-revalidation gate. Report first, act
# deliberately.
#
# Usage: scripts/toolchain_refresh.sh            # report drift for all forks + the pinned wheels
#        REFRESH_FETCH=0 scripts/toolchain_refresh.sh   # skip network fetch (use already-fetched refs)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "$REPO/toolchain.lock"; set +a

MLIR_AIE="$REPO/mlir-aie"
IRON="${IRON:-~/repositories/ns/amd/IRON}"
MLIR_AIR="${MLIR_AIR:-~/mlir-air}"
FETCH="${REFRESH_FETCH:-1}"

hr(){ printf '%s\n' "------------------------------------------------------------------------"; }

# Report drift for one fork checkout.
#   $1 dir   $2 our-branch   $3 upstream-ref   $4 label
report_fork() {
  local dir="$1" branch="$2" up="$3" label="$4"
  hr; echo "### $label   ($dir)"
  git -C "$dir" rev-parse --git-dir >/dev/null 2>&1 || { echo "  MISSING checkout"; return; }
  [ "$FETCH" = 1 ] && git -C "$dir" fetch -q --all 2>/dev/null || true
  local tip uptip mb behind
  tip="$(git -C "$dir" rev-parse --short "$branch" 2>/dev/null || echo '?')"
  uptip="$(git -C "$dir" rev-parse --short "$up" 2>/dev/null || echo '?')"
  echo "  our $branch = $tip    upstream $up = $uptip"
  if git -C "$dir" rev-parse "$up" >/dev/null 2>&1 && git -C "$dir" rev-parse "$branch" >/dev/null 2>&1; then
    behind="$(git -C "$dir" rev-list --count "$branch..$up" 2>/dev/null || echo '?')"
    mb="$(git -C "$dir" merge-base "$branch" "$up" 2>/dev/null | cut -c1-9)"
    echo "  upstream commits not yet in our branch: $behind   (merge-base $mb)"
    # Which of OUR commits are already merged upstream (subject-match against the upstream gap) = DROP list.
    echo "  our commits that look MERGED upstream (drop on next rebase):"
    local found=0
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      local sub; sub="$(echo "$line" | cut -d' ' -f2-)"
      # match by PR-number tag or a strong subject prefix in the upstream gap
      if git -C "$dir" log --oneline "$branch..$up" 2>/dev/null | grep -qiF "$(echo "$sub" | sed -E 's/\.patch.*//; s/:.*//' | cut -c1-30)"; then
        echo "      - $line"; found=1
      fi
    done < <(git -C "$dir" log --oneline "$up..$branch" 2>/dev/null)
    [ "$found" = 0 ] && echo "      (none auto-detected -- still verify by hand: upstream may have squashed)"
    echo "  REBASE (after deciding keepers): git -C $dir rebase --onto $up $mb $branch   # skip merged/empty commits"
  fi
}

echo "TOOLCHAIN REFRESH -- staleness report ($(date -u +%Y-%m-%dT%H:%MZ 2>/dev/null || echo now))"
echo "Lock pins:  MLIR_AIE_FORK_COMMIT=${MLIR_AIE_FORK_COMMIT:0:12}  PEANO_DIST=$PEANO_DIST  NANOBIND=$NANOBIND"

report_fork "$MLIR_AIE" xdna2-asr upstream/main "mlir-aie (Xilinx)"
report_fork "$IRON"     xdna2-asr origin/devel  "amd/IRON"
report_fork "$MLIR_AIR" main      origin/main    "mlir-air (Xilinx, PRs live on branches)"

hr
echo "### Peano / llvm-aie wheel pin"
echo "  pinned: $PEANO_DIST"
WHEEL_DIR="$REPO/.venv-iron/lib/python3.14/site-packages"
if [ -d "$WHEEL_DIR" ]; then
  installed="$(ls -d "$WHEEL_DIR"/llvm_aie* 2>/dev/null | head -1 | xargs -r basename)"
  echo "  installed in .venv-iron: ${installed:-<none>}"
fi
echo "  latest nightly: see https://github.com/Xilinx/llvm-aie/releases (bump PEANO_DIST + re-smoke if newer & needed)"

hr
echo "### Open PR ledger (verify movement before rebasing -- merged PRs must be dropped)"
echo "  gh search prs --author @me --state merged --limit 30 --json repository,number,title"
echo "  gh search prs --author @me --state open   --limit 30 --json repository,number,title"

hr
cat <<'NEXT'
NEXT STEPS (deliberate, gated):
  1. Decide keepers vs drops per fork (drop anything merged; re-express patches that conflict with merged work).
  2. Rebase each fork branch (commands above). Resolve conflicts BY HAND -- upstream may have removed things we
     depend on (e.g. aie.iron.placers, makefile-common kernel-build); those need a port, not a mechanical pick.
  3. Re-pin toolchain.lock MLIR_AIE_FORK_COMMIT to the new mlir-aie tip; bump PEANO_DIST if a newer nightly is needed.
  4. CPU gate:    scripts/toolchain_smoke.sh        (MUST pass -- exit 0)
  5. DEVICE gate (NPU window, single-tenant): re-validate WER/occupancy before force-pushing forks or shipping.
     Only after BOTH gates pass: force-push fork branches (force-with-lease; keep presync-* rollback tags).
NEXT
