#!/usr/bin/env bash
# Score tanh_subreg_clobber.ll against every Peano build we have, plus the two flags
# that make it correct. Device-free: this is a codegen defect, so compiling is enough.
#
#   route_b_kernels/probes/tanh_subreg_clobber.sh [extra llc flags...]
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ll="$here/tanh_subreg_clobber.ll"
check="$here/check_vtanh_clobber.py"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

score() {                       # score <llc> <label> [flags...]
  local llc="$1" label="$2"; shift 2
  [ -x "$llc" ] || { echo "  $label -- no llc"; return; }
  "$llc" -O2 -mtriple=aie2p-none-unknown-elf "$@" "$ll" -o "$tmp/out.S" 2>/dev/null
  [ -s "$tmp/out.S" ] || { echo "  $label -- llc FAILED"; return; }
  printf '  %-46s %s\n' "$label" "$(python3 "$check" "$tmp/out.S" x | tail -1 | sed 's/^ *//')"
}

for d in "$here"/../../../.cache/peano-local/*/; do
  score "$d/bin/llc" "$(basename "$d")"
done
active="${PEANO_INSTALL_DIR:-}"
if [ -n "$active" ]; then
  echo "active pin, with the flags that change the verdict:"
  score "$active/bin/llc" "baseline"
  score "$active/bin/llc" "-enable-subreg-liveness=false" -enable-subreg-liveness=false
  score "$active/bin/llc" "-join-liveintervals=false"     -join-liveintervals=false
  score "$active/bin/llc" "-verify-machineinstrs"         -verify-machineinstrs
fi
