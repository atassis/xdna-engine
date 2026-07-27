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
# Copy-forward the STAGED BD-onchip sources (route_b_kernels/conveyor_proto = source of truth, per
# sync_kernels.sh discipline) into the mlir-aie build sandbox. These carry the t_active RTP key-mask
# (stage_scores_relpos_bd_mask + bd_emit_bake_ta) + the p-resident hook. Idempotent overlay; never edit
# the sandbox copy directly. Skip if the staged dir is absent (fall back to whatever the fork branch has).
SRC=route_b_kernels/conveyor_proto
if [ -d "$SRC" ]; then
  mkdir -p "$EX"
  cp "$SRC/conveyor_attn.cc" "$SRC/conveyor_attn_iron.py" "$SRC/Makefile" "$SRC/run_bd_onchip.py" "$EX/"
  echo "[bd-prebuild] synced staged BD-onchip sources -> $EX"
fi
# MASK=1 wires the t_active RTP key-mask (required before any WER gate). Set MASK=0 to build the
# unmasked full-length variant (matches the standalone T==BUILT_T rel-L2 gate only).
MASK="${MASK:-1}"
echo "[bd-prebuild] building H=$H BD-onchip conveyor (T=$T N_QT=$NQT BD_KB=$BD_KB MASK=$MASK) ..."
( cd "$EX" && make clean >/dev/null 2>&1; \
  ATTN_TQ=$TQ ATTN_T=$T ATTN_DK=$DK ATTN_NQT=$NQT ATTN_HEADS=$H BD_KB=$BD_KB \
  make NPU2=1 VARIANT=attn BDON=1 MASK=$MASK ATTN_TQ=$TQ ATTN_T=$T ATTN_DK=$DK ATTN_NQT=$NQT ATTN_HEADS=$H BD_KB=$BD_KB \
       KFLAGS="-DATTN_TQ=$TQ -DATTN_T=$T -DATTN_DK=$DK -DATTN_NQT=$NQT -DBD_KB=$BD_KB -DATTN_SCALE=${INV}f" >/dev/null 2>&1 )
if [ ! -f "$EX/build/final.xclbin" ]; then
  echo "[bd-prebuild] FAILED (no final.xclbin) -- check the BD-onchip generator + KFLAGS"; exit 1
fi
mkdir -p "$out"; cp "$EX/build/final.xclbin" "$out/final.xclbin"; cp "$EX/build/insts.bin" "$out/insts.bin"
echo "[bd-prebuild] installed H=$H BD-onchip xclbin -> $out"
