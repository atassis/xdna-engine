#!/usr/bin/env bash
# Run a designs/warp/kernel/*.py device script under the shared NPU lock. Copy of
# designs/fsr1/kernel/run.sh (same rail, different worktree).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"          # wt-npu-warp
WS="$(cd "$REPO/.." && pwd)"                  # the umbrella workspace dir this repo is checked out under
PROG="${1:?usage: run.sh <verify_xxx.py>}"

LOCK="${NPU_LOCK_SH:-}"
[ -n "$LOCK" ] && [ -x "$LOCK" ] || { echo "run.sh: NPU_LOCK_SH unset or not executable: '$LOCK'. Point it at the workspace's shared NPU lock helper." >&2; exit 1; }

INST="$("$REPO/scripts/toolchain_up.sh")"
[ -n "$INST" ] || { echo "run.sh: empty instance dir from toolchain_up.sh" >&2; exit 1; }

VENV="${BRICK_VENV:-}"
if [ -z "$VENV" ]; then
  for c in "$REPO/.venv-iron" "$WS"/*/.venv-iron; do
    [ -x "$c/bin/python" ] && { VENV="$c"; break; }
  done
fi
[ -n "$VENV" ] || { echo "run.sh: no .venv-iron found; set BRICK_VENV" >&2; exit 1; }

echo "[run.sh] instance $INST" >&2
echo "[run.sh] venv     $VENV" >&2

export NPU_WAIT_S="${NPU_WAIT_S:-1800}"
exec "$LOCK" queue -- env \
  PATH="$VENV/bin:$VENV/cc-shim:$PATH" \
  PYTHONPATH="$INST/python:${PYTHONPATH:-}" \
  AIECC_PATH="$INST/bin/aiecc" \
  PEANO_INSTALL_DIR="$VENV/lib/python3.14/site-packages/llvm-aie" \
  XRT_INC_DIR=/usr/include \
  XRT_LIB_DIR=/usr/lib \
  CUDA_VISIBLE_DEVICES="" \
  bash -c "cd '$HERE' && exec python -u '$PROG'"
