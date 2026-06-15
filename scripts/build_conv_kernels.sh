#!/usr/bin/env bash
# Build the per-channel-band M-stationary conv xclbins for Phase-2c (ResNet-18 on NPU).
# Each: M=512 (one M-tile = m*4*8), K=768 (one K-split chunk), N=Cout band. CPU-only (open Peano).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
bash scripts/sync_kernels.sh >/dev/null
WA=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
cd "$WA"
for N in 64 128 256 512; do
  echo "== mstat 512x768x$N =="
  make -f Makefile.mstat NPU2=1 M=512 K=768 N=$N m=16 k=32 n=32 n_aie_cols=8 \
    build/final_mstat_512x768x${N}_16x32x32_8c.xclbin
done
ls -1 build/final_mstat_512x768x*_16x32x32_8c.xclbin
