#!/usr/bin/env bash
# resident_forward build environment: PYTHONPATH/PATH for rf_build.py (aiecc + IRON).
#
# IRON_DIR is REQUIRED (no default): a silently-wrong IRON tree changes the generated MLIR/ELF
# bytes, so recipes/rf48C.sh checks it against a specific pinned commit rather than accepting
# whatever amd_paths.sh's IRON_DIR floor happens to resolve to.
# RF_INST defaults to the toolchain instance scripts/toolchain_up.sh resolves for toolchain.lock.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${IRON_DIR:?set IRON_DIR to an IRON checkout on the pinned commit (see recipes/rf48C.sh)}"
source "$REPO/scripts/amd_paths.sh"
RF_INST="${RF_INST:-$("$REPO/scripts/toolchain_up.sh")}"
export RF_INST IRON_DIR
export PYTHONPATH="$IRON_DIR:$RF_INST/build/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$RF_INST/bin:$AIEBU_ASM_DIR:$PATH"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
PY="$VENV_IRON/bin/python"
[ -x "$PY" ] || PY=python3
export PY
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
