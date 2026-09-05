#!/usr/bin/env bash
# Compile a brick .cc for aie2p WITHOUT touching the device. Usage: ./compile_check.sh <file.cc>...
#
# Why this exists: the NPU is single-tenant, so only one thing can gate on device at a time, but
# most kernel mistakes (template instantiation, aie_api overload resolution, arity) are compile-time
# and need no device at all. Authoring can therefore run wide while device verification stays serial.
#
# The flags are NOT invented here -- they are the instance's own PEANOWRAP2P_FLAGS
# (src/programming_examples/makefile-common:34). The one non-obvious member is
# -D__AIE_API_AIE_ADF_HPP__: the target predefines __AIENGINE__, which makes aie.hpp pull in
# aie_adf.hpp, which needs adf.h -- a Vitis header the Peano-only instance does not ship. Defining
# the include guard suppresses that branch. Without it every brick fails on "'adf.h' file not found",
# which reads as a broken toolchain rather than a missing -D.
#
# A clean compile is NECESSARY, NOT SUFFICIENT: it says nothing about spill/aliasing codegen, which
# is what actually broke sin and rope_lut. Device gate still decides.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

INST="$("$REPO/scripts/toolchain_up.sh")"
[ -n "$INST" ] || { echo "compile_check: empty instance dir from toolchain_up.sh" >&2; exit 1; }

# Peano lives in a venv, same borrowing rule as run.sh (not every worktree has one).
PE="${PEANO_INSTALL_DIR:-}"
if [ -z "$PE" ]; then
  WS="$(cd "$REPO/.." && pwd)"
  for c in "$REPO/.venv-iron" "$WS"/*/.venv-iron; do
    for p in "$c"/lib/python*/site-packages/llvm-aie; do
      [ -x "$p/bin/clang++" ] && { PE="$p"; break 2; }
    done
  done
fi
[ -x "$PE/bin/clang++" ] || { echo "compile_check: no Peano clang++; set PEANO_INSTALL_DIR" >&2; exit 1; }

INC="$INST/src/third_party/aie_api/include"
[ -f "$INC/aie_api/aie.hpp" ] || INC="$INST/include"

rc=0
for f in "$@"; do
  printf '%-46s ' "$f"
  if err="$("$PE/bin/clang++" --target=aie2p-none-unknown-elf -O2 -std=c++20 \
        -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ -I"$INC" -c "$f" -o /dev/null 2>&1)"; then
    echo "COMPILES"
  else
    echo "FAIL"; printf '%s\n' "$err" | head -20; rc=1
  fi
done
exit $rc
