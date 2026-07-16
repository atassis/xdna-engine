#!/usr/bin/env bash
# Build the 8-head relpos-MHA CONVEYOR xclbin (STEP scores(relpos) -> softmax -> ctx, 8 heads x 3
# tiles, ONE dispatch) into artifacts/conveyor/single/{final.xclbin,insts.bin}. Mirrors
# relpos_prebuild.sh. npu.rs::relpos_mha_conveyor loads this once (resident) behind
# PARAKEET_CONVEYOR_MHA=1.
#
# Real Parakeet dims: TQ=8, DK=128, T=172 padded to 176 (a VL(16) multiple), N_QT=22, H=8.
# Validated recipe (H=8 rel-L2 4.69e-3 @ 3.96 ms): grouped-MemTile split of q AND k + acquire-once
# resident weights + ctx JOIN, v per-head direct (3 MemTile ops/group, GJ=4). The conveyor generator
# + kernel live in the mlir-aie fork (branch conveyor-proto-real-dims), example dir
# programming_examples/basic/conveyor_proto (conveyor_attn.cc + conveyor_attn_iron.py).
#
# Usage:  scripts/conveyor_prebuild.sh          (builds the 8-head T=176 xclbin)
# Needs the FORK toolchain env (sourced internally). Serializes on the shared toolchain + NPU.
# DEVICE HYGIENE FIRST: systemctl --user stop npu-asr npu-vox ; fuser /dev/accel/accel0 (must be clear).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# 8-head conveyor build dims (MUST match npu.rs CONV_* and conveyor_attn_iron.py).
TQ=8; T=176; DK=128; NQT=22; HEADS=8
INV_SCALE=0.08838835   # 1/sqrt(128)
EX=mlir-aie/programming_examples/basic/conveyor_proto
out="artifacts/conveyor/single"

source scripts/iron_env.sh
python3 -c "import aie.iron" 2>/dev/null || { echo "[conveyor-prebuild] iron env not green"; exit 2; }
scripts/sync_kernels.sh >/dev/null 2>&1 || true

if [ -f "$out/final.xclbin" ] && [ -f "$out/insts.bin" ] && [ -z "${FORCE:-}" ]; then
  echo "[conveyor-prebuild] xclbin already present -> $out (FORCE=1 to rebuild)"; exit 0
fi
echo "[conveyor-prebuild] building 8-head conveyor (VARIANT=attn RELPOS=1, TQ=$TQ T=$T DK=$DK NQT=$NQT H=$HEADS) ..."
# The IRON generator reads ATTN_* from env; the kernel .cc reads the same dims via -D (KFLAGS).
# Keep the two in sync or the belt layout and the kernel loop bounds diverge.
( cd "$EX" && make clean >/dev/null 2>&1; \
  ATTN_TQ=$TQ ATTN_T=$T ATTN_DK=$DK ATTN_NQT=$NQT ATTN_HEADS=$HEADS \
  make NPU2=1 VARIANT=attn RELPOS=1 \
       ATTN_TQ=$TQ ATTN_T=$T ATTN_DK=$DK ATTN_NQT=$NQT ATTN_HEADS=$HEADS \
       KFLAGS="-DATTN_TQ=$TQ -DATTN_T=$T -DATTN_DK=$DK -DATTN_SCALE=${INV_SCALE}f" >/dev/null 2>&1 )
if [ ! -f "$EX/build/final.xclbin" ]; then
  echo "[conveyor-prebuild] FAILED (no final.xclbin) -- check the fork branch (conveyor-proto-real-dims) + KFLAGS"; exit 1
fi
mkdir -p "$out"
cp "$EX/build/final.xclbin" "$out/final.xclbin"
cp "$EX/build/insts.bin"    "$out/insts.bin"
echo "[conveyor-prebuild] installed 8-head conveyor xclbin + insts -> $out"
