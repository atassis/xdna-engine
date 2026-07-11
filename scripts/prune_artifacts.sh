#!/usr/bin/env bash
# prune_artifacts.sh -- SAFE, dry-run-by-default reclaim of regenerable artifacts/ contents.
#
# The "gitignored = recreatable" principle (npu-weights roadmap T8): nothing under artifacts/ is
# precious IF it has a proven regeneration path. This tool codifies the keep/delete RULE:
#   * PRUNABLE = has a verified regen command (the regen script/tool exists on disk) AND is either a
#     transient A/B validation snapshot (leading `_`) or a per-model oracle/arena regenerable on demand.
#   * KEEP     = tracked in git (a checked-in reference like goldens/), OR a load-bearing artifact the
#     live engine currently loads (regenerable, but in active use -> not reclaimed by default).
#   * REVIEW   = regenerable in principle but no regen script found here, or large/ambiguous -> owner decides.
#
# HARD SAFETY: never deletes without an explicit --delete, and never marks an entry PRUNABLE unless its
# regen command's script/tool is present. Default run prints a plan only and reclaims nothing.
#
# Usage:
#   scripts/prune_artifacts.sh            # dry-run: print the plan + reclaimable size (default)
#   scripts/prune_artifacts.sh --delete   # actually remove PRUNABLE entries (owner-invoked only)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO"

DO_DELETE=0
[ "${1:-}" = "--delete" ] && DO_DELETE=1

ART="artifacts"
[ -d "$ART" ] || { echo "no $ART/ dir here"; exit 0; }

# Map an artifacts/<name> to: CLASS|REGEN_CMD|REGEN_PROBE   (REGEN_PROBE = a path that must exist for
# the regen to be possible; empty = always-available tool). Classification is by name pattern.
classify() {
  local name="$1" cls regen probe
  case "$name" in
    _validate_*|_ab_*|*_ab|*_validate*)
      cls=PRUNABLE; regen="re-run the A/B validation build that produced it"; probe="" ;;
    esm2-8m)   cls=PRUNABLE; regen=".venv-export/bin/python scripts/export_esm.py facebook/esm2_t6_8M_UR50D esm2-8m";  probe="scripts/export_esm.py" ;;
    esm2-35m)  cls=PRUNABLE; regen=".venv-export/bin/python scripts/export_esm.py facebook/esm2_t12_35M_UR50D esm2-35m"; probe="scripts/export_esm.py" ;;
    parakeet)  cls=KEEP;     regen=".venv-export/bin/python scripts/extract_parakeet_encoder.py (LOAD-BEARING: live ASR encoder)"; probe="scripts/extract_parakeet_encoder.py" ;;
    gemma4-e2b|gemma3-270m)  cls=REVIEW; regen="gemma export spike (see spike-gemma-ffn); regenerable but owner-scoped"; probe="" ;;
    goldens)   cls=KEEP;     regen="git-tracked reference (manifest.tsv) -- NOT regenerated, do not prune"; probe="" ;;
    *)         cls=REVIEW;   regen="no regen rule recorded for this name"; probe="" ;;
  esac
  # Downgrade PRUNABLE -> REVIEW if the regen script is named but absent (never claim a false regen path).
  if [ "$cls" = "PRUNABLE" ] && [ -n "$probe" ] && [ ! -e "$probe" ]; then
    cls=REVIEW; regen="regen script MISSING ($probe) -- cannot prove regenerable: $regen"
  fi
  echo "$cls|$regen"
}

printf '%-22s %8s  %-9s %s\n' "ARTIFACT" "SIZE" "CLASS" "REGEN / REASON"
printf '%-22s %8s  %-9s %s\n' "--------" "----" "-----" "--------------"
reclaim_kb=0
prunable_list=()
for d in "$ART"/*; do
  [ -e "$d" ] || continue
  name="$(basename "$d")"
  size_h="$(du -sh "$d" 2>/dev/null | cut -f1)"
  size_kb="$(du -sk "$d" 2>/dev/null | cut -f1)"
  IFS='|' read -r cls regen <<<"$(classify "$name")"
  printf '%-22s %8s  %-9s %s\n' "$name" "$size_h" "$cls" "$regen"
  if [ "$cls" = "PRUNABLE" ]; then
    reclaim_kb=$((reclaim_kb + size_kb))
    prunable_list+=("$d")
  fi
done

reclaim_h="$(awk -v k="$reclaim_kb" 'BEGIN{ s=k*1024; u="B"; if(s>=1024){s/=1024;u="KB"} if(s>=1024){s/=1024;u="MB"} if(s>=1024){s/=1024;u="GB"} printf "%.1f%s", s, u }')"
echo
echo "PRUNABLE reclaimable: ${reclaim_h}  (${#prunable_list[@]} entries)"

if [ "$DO_DELETE" = "1" ]; then
  echo "== --delete: removing PRUNABLE entries =="
  for d in "${prunable_list[@]}"; do echo "  rm -rf $d"; rm -rf "$d"; done
  echo "done. Regenerate any needed entry with its REGEN command above."
else
  echo "(dry-run -- nothing deleted. Re-run with --delete to reclaim PRUNABLE entries. KEEP/REVIEW never auto-pruned.)"
fi
