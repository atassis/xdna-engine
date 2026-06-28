#!/usr/bin/env bash
# Point .venv-iron at a toolchain instance's python (off the wheel), reversibly, via the aie.pth redirect.
#   toolchain_wire.sh on   -> wire to the current lock's instance (builds it if needed)
#   toolchain_wire.sh off  -> restore the wheel
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTH="$REPO/.venv-iron/lib/python3.14/site-packages/aie.pth"
BAK="$PTH.wheel-bak"
case "${1:-}" in
  on)
    INST="$("$REPO/scripts/toolchain_up.sh")"
    [ -f "$BAK" ] || cp "$PTH" "$BAK"
    printf '%s\n' "$INST/python" > "$PTH"
    echo "[wire] aie.pth -> $INST/python" ;;
  off)
    [ -f "$BAK" ] && mv "$BAK" "$PTH" && echo "[wire] restored wheel aie.pth" || echo "[wire] no backup; already wheel" ;;
  *) echo "usage: toolchain_wire.sh on|off" >&2; exit 2 ;;
esac
