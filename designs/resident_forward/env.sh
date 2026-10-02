#!/usr/bin/env bash
# resident_forward build environment: PYTHONPATH/PATH for rf_build.py (aiecc + IRON).
#
# RF_INST defaults to the toolchain instance scripts/toolchain_up.sh resolves for toolchain.lock.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$REPO/scripts/amd_paths.sh"
iron_require_source
RF_INST="${RF_INST:-$("$REPO/scripts/toolchain_up.sh")}"
export RF_INST IRON_DIR
export PYTHONPATH="$IRON_DIR:$RF_INST/build/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$RF_INST/bin:$AIEBU_ASM_DIR:$PATH"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
PY="$VENV_IRON/bin/python"
[ -x "$PY" ] || PY=python3
export PY
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
