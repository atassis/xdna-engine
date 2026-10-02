#!/usr/bin/env bash
# Bind .venv-iron's aie.pth to the selected toolchain instance.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/lib/fork_python_binding.sh"
case "${1:-}" in
  on)
    INST="$("$REPO/scripts/toolchain_up.sh")"
    bind_fork_python "$REPO/.venv-iron/bin/python" "$INST"
    echo "[wire] aie.pth -> $INST/python" ;;
  off)
    echo "[wire] ERROR: wheel Python fallback is unsupported; re-run scripts/toolchain_up.sh" >&2
    exit 1 ;;
  *) echo "usage: toolchain_wire.sh on" >&2; exit 2 ;;
esac
