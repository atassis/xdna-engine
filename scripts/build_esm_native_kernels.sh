#!/usr/bin/env bash
# Build the NATIVE (real-K) whole-array xclbins for ESM-2 (research/comparison path; see
# internal notes + internal notes). Idempotent; CPU-only.
# Reusable template for onboarding ANY new-width model natively: pick K=real width, N=round up to the
# tiling multiple (tile_n * n_aie_cols = 256 for the 32x32x32 8-col tile), one (K,N) per matmul type.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
MMW=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array

build_shape() { # K N
  local K=$1 N=$2
  rm -f "$MMW/build/mm_32x32x32.o" "$MMW/build/aie_512x${K}x${N}_32x32x32_8c.mlir"
  echo "== native $K x $N =="
  make -C "$MMW" NPU2=1 M=512 K="$K" N="$N" m=32 k=32 n=32 \
    dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 \
    "build/final_512x${K}x${N}_32x32x32_8c.xclbin"
}

# 8M (hidden 320, ff 1280): proj 320x512, ffn1 320x1280, ffn2 1280x512
build_shape 320 512
build_shape 320 1280
build_shape 1280 512
# 35M (hidden 480, ff 1920): proj 480x512, ffn1 480x2048, ffn2 1920x512
build_shape 480 512
build_shape 480 2048
build_shape 1920 512

echo "ESM native xclbins built. Verify: ./rust/target/debug/verify_esm scenarios/esm2-8m-native.toml"
