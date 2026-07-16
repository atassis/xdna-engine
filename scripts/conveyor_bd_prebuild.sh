#!/usr/bin/env bash
# Build the H=4 BD-ON-CHIP conveyor xclbin (4-stage column BD->scores->softmax->ctx per head, BD computed
# on-chip -> NO host BD precompute) into artifacts/conveyor_bd/single/{final.xclbin,insts.bin}. Shipped as
# H=4x2 (dispatch twice for 8 heads). Real Parakeet dims TQ=8 T=176 DK=128 N_QT=22 P=351 BD_KB=39 H=4.
# Device hygiene first: systemctl --user stop npu-asr npu-vox ; fuser /dev/accel/accel0 (clear).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
TQ=8; T=176; DK=128; NQT=22; BD_KB=39; H=4; INV=0.08838835
EX=mlir-aie/programming_examples/basic/conveyor_proto
out="artifacts/conveyor_bd/single"

source scripts/iron_env.sh
python3 -c "import aie.iron" 2>/dev/null || { echo "[bd-prebuild] iron env not green"; exit 2; }
scripts/sync_kernels.sh >/dev/null 2>&1 || true

if [ -f "$out/final.xclbin" ] && [ -f "$out/insts.bin" ] && [ -z "${FORCE:-}" ]; then
  echo "[bd-prebuild] present -> $out (FORCE=1 to rebuild)"; exit 0
fi
echo "[bd-prebuild] building H=$H BD-onchip conveyor (T=$T N_QT=$NQT BD_KB=$BD_KB) ..."
( cd "$EX" && make clean >/dev/null 2>&1; \
  ATTN_TQ=$TQ ATTN_T=$T ATTN_DK=$DK ATTN_NQT=$NQT ATTN_HEADS=$H BD_KB=$BD_KB \
  make NPU2=1 VARIANT=attn BDON=1 ATTN_TQ=$TQ ATTN_T=$T ATTN_DK=$DK ATTN_NQT=$NQT ATTN_HEADS=$H BD_KB=$BD_KB \
       KFLAGS="-DATTN_TQ=$TQ -DATTN_T=$T -DATTN_DK=$DK -DATTN_NQT=$NQT -DBD_KB=$BD_KB -DATTN_SCALE=${INV}f" >/dev/null 2>&1 )
if [ ! -f "$EX/build/final.xclbin" ]; then
  echo "[bd-prebuild] FAILED (no final.xclbin) -- check the BD-onchip generator + KFLAGS"; exit 1
fi
mkdir -p "$out"; cp "$EX/build/final.xclbin" "$out/final.xclbin"; cp "$EX/build/insts.bin" "$out/insts.bin"
echo "[bd-prebuild] installed H=$H BD-onchip xclbin -> $out"
