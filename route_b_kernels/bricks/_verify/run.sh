#!/usr/bin/env bash
# Run a brick device-verify harness under the NPU lock with the wired IRON toolchain.
# Usage: ./run.sh verify_xxx.py
#
# Two things have to come from two DIFFERENT places, and conflating them is what broke this:
#
#   * the toolchain INSTANCE must come from THIS worktree's toolchain.lock. The instance dir is
#     content-addressed on that file, so borrowing a sibling's env silently runs the bricks
#     against the sibling's pin. That is what used to happen here (hardcoded wt-conveyor-dev),
#     and when that sibling's pin went stale/unbuildable, toolchain_up.sh failed, `set -e` killed
#     the run BEFORE python, and the harness printed nothing -- it read as an empty run rather
#     than a broken toolchain.
#
#   * the VENV has to be borrowed, because not every worktree has one (this one does not). It
#     carries the python deps (ml_dtypes for the bf16 host dtypes) and the Peano install.
#
# So: instance from here, venv from wherever one exists. Override either with
# BRICK_VENV=/path/to/.venv-iron if you need a specific one.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
PROG="${1:?usage: run.sh <verify_xxx.py>}"

# The NPU is single-tenant, so device runs must be serialised across every checkout on the
# box. That lock helper coordinates more than this repo and therefore lives outside it: point
# NPU_LOCK_SH at it. Contract: `<lock> queue -- <cmd...>` runs <cmd> holding the device lock,
# waiting up to NPU_WAIT_S, and exits 75 if the device stays busy (caller should DEFER, not
# force). Without it we still run, but unserialised -- fine on an idle box, not for timing.
LOCK="${NPU_LOCK_SH:-}"
if [ -z "$LOCK" ] || [ ! -x "$LOCK" ]; then
  echo "run.sh: NPU_LOCK_SH unset or not executable -- running WITHOUT the device lock." >&2
  echo "        Serialise manually if anything else may touch the NPU." >&2
  LOCK=""
fi

# Instance: this worktree's pin. Let failures speak -- do not redirect to /dev/null.
INST="$("$REPO/scripts/toolchain_up.sh")"
[ -n "$INST" ] || { echo "run.sh: empty instance dir from toolchain_up.sh" >&2; exit 1; }

# Venv: first one that actually exists, unless told otherwise.
VENV="${BRICK_VENV:-}"
if [ -z "$VENV" ]; then
  for c in "$REPO/.venv-iron" "$WS"/*/.venv-iron; do
    [ -x "$c/bin/python" ] && { VENV="$c"; break; }
  done
fi
[ -n "$VENV" ] || { echo "run.sh: no .venv-iron found; set BRICK_VENV" >&2; exit 1; }

echo "[run.sh] instance $INST" >&2
echo "[run.sh] venv     $VENV" >&2

export NPU_WAIT_S="${NPU_WAIT_S:-600}"
# With a lock helper: run under it. Without: run directly (warned above).
exec ${LOCK:+"$LOCK" queue --} env \
  PATH="$VENV/bin:$VENV/cc-shim:$PATH" \
  PYTHONPATH="$INST/python:${PYTHONPATH:-}" \
  AIECC_PATH="$INST/bin/aiecc" \
  PEANO_INSTALL_DIR="$VENV/lib/python3.14/site-packages/llvm-aie" \
  XRT_INC_DIR=/usr/include \
  XRT_LIB_DIR=/usr/lib \
  CUDA_VISIBLE_DEVICES="" \
  bash -c "cd '$HERE' && exec python -u '$PROG'"
