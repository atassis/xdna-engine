#!/usr/bin/env bash
# Build the on-chip single-query MHA-decode xclbin (M1 Task 1; CPU-only, no NPU).
#
# ONE xclbin (seq=448): the tile count is FIXED at n_tiles=ceil(448/64)=7 and the real
# per-tile key count is a RUNTIME value (host writes an int32 into each tile's header), so
# the SAME xclbin serves every cache length S<=448 with no zero-pad softmax poison. The
# probe (rust/npu-asr/src/bin/mha_decode_probe.rs) feeds fixed-seed random q/K/V at several
# S and compares ctx vs the host `attend_one` reference.
#
# Output (in the ml/mha_decode build sandbox):
#   final_mha_decode_448.xclbin   insts_mha_decode_448.txt
#
# Usage:  scripts/build_mha_decode.sh          (builds the single seq=448 xclbin)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
bash scripts/sync_kernels.sh

MHA=mlir-aie/programming_examples/ml/mha_decode
S=448  # fixed: names the single runtime-S xclbin; tile count is ceil(448/64)=7.

echo "== mha_decode: runtime-S xclbin seq=${S} (streaming/flash single-query MHA, 12 heads x 64, bf16 in / f32 ctx) =="
# Regenerate the MLIR fresh; kernel .o depends only on TKV.
rm -f "$MHA/build/aie_mha_decode_${S}.mlir"
make -C "$MHA" -f Makefile.mha NPU2=1 seq="$S" "build/final_mha_decode_${S}.xclbin"
echo "Built: $MHA/build/final_mha_decode_${S}.xclbin"
echo "       $MHA/build/insts_mha_decode_${S}.txt"
