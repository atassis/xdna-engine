#!/usr/bin/env bash
# Pre-build the resident relpos-MHA block (STEP=8) for one or more encoder frame counts T,
# installing each into artifacts/relpos/T<T>/{final.xclbin,insts.bin} where npu.rs loads it.
# The kernel bakes RELPOS_T, so there is one xclbin per T. TQ=8 KB=43 (must match npu.rs).
#
# Usage:  scripts/relpos_prebuild.sh <T> [<T> ...]
# Needs the FORK toolchain env (sourced internally). Serializes on the shared toolchain.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
TQ=8; KB=43
EX=mlir-aie/programming_examples/ml/relpos_mha

setup_env() { source scripts/iron_env.sh; }
setup_env
python3 -c "import aie.iron" 2>/dev/null || { echo "[prebuild] iron env not green"; exit 2; }
scripts/sync_kernels.sh >/dev/null 2>&1

for T in "$@"; do
  out="artifacts/relpos/T${T}"
  if [ -f "$out/final.xclbin" ] && [ -f "$out/insts.bin" ]; then
    echo "[prebuild] T=$T already present -> $out"; continue
  fi
  echo "[prebuild] building STEP=8 T=$T TQ=$TQ KB=$KB ..."
  ( cd "$EX" && make clean >/dev/null 2>&1; \
    make NPU2=1 STEP=8 T="$T" TQ="$TQ" KB="$KB" >/dev/null 2>&1 )
  if [ ! -f "$EX/build/final.xclbin" ]; then
    echo "[prebuild] FAILED T=$T (no final.xclbin)"; exit 1
  fi
  mkdir -p "$out"
  cp "$EX/build/final.xclbin" "$out/final.xclbin"
  cp "$EX/build/insts.bin"    "$out/insts.bin"
  echo "[prebuild] installed T=$T -> $out"
done
echo "[prebuild] done."
