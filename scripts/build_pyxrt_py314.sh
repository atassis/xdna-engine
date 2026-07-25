#!/usr/bin/env bash
# Build a py3.14 pyxrt with the ctrl-scratchpad binding and install it into .venv-iron.
#
# WHY: the Arch `xrt 2.21.75` package ships a py3.14 pyxrt.so built WITHOUT
# `xrt::run::get_ctrl_scratchpad_bo` (the C++ symbol IS in the installed
# libxrt_coreutil, and the binding IS in the XRT source + upstream -- only the
# packaged py3.14 .so omits it). IRON's ParameterScratchpad (per-token decode
# kv_off/sm_mask writes) calls `run.get_ctrl_scratchpad_bo()`, so the stale
# binding blocks all scratchpad-parameter decode from Python. We own the
# toolchain: rebuild the one pybind module against the installed XRT 2.21.75
# headers/lib. Reproducible so the .so is never a lost hand-build.
#
# Usage: bash scripts/build_pyxrt_py314.sh   (idempotent; overwrites the venv .so)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
VENV_IRON="${VENV_IRON:-$REPO/.venv-iron}"
XRT_SRC="${XRT_SRC:-$WS/XRT-src}"
SRC="$XRT_SRC/src/python/pybind11/src/pyxrt.cpp"
SITE="$VENV_IRON/lib/python3.14/site-packages"
OUT="$SITE/pyxrt.cpython-314-x86_64-linux-gnu.so"
# pybind11 headers: reuse the torch-bundled set already in .venv-iron (py3.14).
PYBIND_INC="$SITE/torch/include"

[ -f "$SRC" ] || { echo "ERROR: pyxrt.cpp not at $SRC"; exit 1; }
[ -f "$PYBIND_INC/pybind11/pybind11.h" ] || { echo "ERROR: pybind11 headers not at $PYBIND_INC (needs torch in .venv-iron)"; exit 1; }
grep -q "get_ctrl_scratchpad_bo" /usr/include/xrt/xrt_kernel.h || { echo "ERROR: installed XRT headers lack get_ctrl_scratchpad_bo -- XRT too old"; exit 1; }

echo "[build_pyxrt] compiling $SRC -> $OUT (against system XRT $(pacman -Q xrt 2>/dev/null | awk '{print $2}'))"
g++ -O2 -shared -fPIC -std=c++17 -fvisibility=hidden \
  $(python3.14-config --includes) \
  -I"$PYBIND_INC" \
  -I/usr/include \
  "$SRC" \
  -o "$OUT" \
  -L/usr/lib -lxrt_coreutil

"$VENV_IRON/bin/python" -c "import pyxrt; assert hasattr(pyxrt.run,'get_ctrl_scratchpad_bo'), 'binding missing'; print('[build_pyxrt] OK:', pyxrt.__file__)"
