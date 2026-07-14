#!/usr/bin/env bash
# STEP-C: build the SINGLE resident relpos-MHA xclbin (STEP=8, T=RELPOS_BUILT_T=172) and its
# template instruction stream into artifacts/relpos/single/{final.xclbin,insts.bin}. npu.rs loads
# this ONCE (resident) and PATCHES the t_active word of the insts per clip -- one xclbin serves any
# clip T <= 172, zero per-clip build. TQ=8 KB=43 (must match npu.rs).
#
# Usage:  scripts/relpos_prebuild.sh          (builds the single T=172 xclbin)
# Needs the FORK toolchain env (sourced internally). Serializes on the shared toolchain.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
BUILT_T=172; TQ=8; KB=43
EX=mlir-aie/programming_examples/ml/relpos_mha
out="artifacts/relpos/single"

setup_env() { source scripts/iron_env.sh; }
setup_env
python3 -c "import aie.iron" 2>/dev/null || { echo "[prebuild] iron env not green"; exit 2; }
scripts/sync_kernels.sh >/dev/null 2>&1

if [ -f "$out/final.xclbin" ] && [ -f "$out/insts.bin" ] && [ -z "${FORCE:-}" ]; then
  echo "[prebuild] single xclbin already present -> $out (FORCE=1 to rebuild)"; exit 0
fi
echo "[prebuild] building the single STEP=8 T=$BUILT_T TQ=$TQ KB=$KB xclbin (runtime t_active) ..."
( cd "$EX" && make clean >/dev/null 2>&1; \
  make NPU2=1 STEP=8 T="$BUILT_T" TQ="$TQ" KB="$KB" TACTIVE="$BUILT_T" >/dev/null 2>&1 )
if [ ! -f "$EX/build/final.xclbin" ]; then
  echo "[prebuild] FAILED (no final.xclbin)"; exit 1
fi
mkdir -p "$out"
cp "$EX/build/final.xclbin" "$out/final.xclbin"
cp "$EX/build/insts.bin"    "$out/insts.bin"
echo "[prebuild] installed single xclbin + template insts -> $out"
