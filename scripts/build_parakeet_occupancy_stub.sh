#!/usr/bin/env bash
# Build the DATA_MOVEMENT_ONLY stub xclbin twin of the Parakeet resident kernel
# (CPU-only; no NPU). For the per-op occupancy harness (brick #8).
#
# The stub xclbin is byte-identical to the production resident kernel in EVERYTHING
# (objectFIFO dataflow, BD chains, locks, DMA access patterns) EXCEPT the in-core
# matmul body, which is elided (route_b_kernels/occupancy/mm_movement_stub.cc). The
# A/B latency diff (full - stub) isolates per-dispatch COMPUTE from movement+stall.
#
# Recipe (swap-and-restore so the production tree is never left modified):
#   1. ensure production resident xclbins exist (scripts/build_parakeet_kernels.sh)
#   2. compile the stub object over build/mm_<tile>.o
#   3. relink final_512x1024x4096_<tile>_8c.xclbin  (now the STUB)  -> save as _STUB
#   4. restore: rebuild production from the real mm.cc (build_parakeet_kernels.sh)
# A trap restores production on any exit.
#
# Usage:
#   scripts/build_parakeet_occupancy_stub.sh            # fast tile 64x32x128 (default)
#   TILE=32x32x32 scripts/build_parakeet_occupancy_stub.sh   # native bf16 tile
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
MMW=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
MAD="$(python3 -c 'from aie.utils.config import root_path; print(root_path())')"
TILE="${TILE:-64x32x128}"

case "$TILE" in
  64x32x128) m=64; k=32; n=128; FAST="emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1";
             MMDEF="-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 -DBFP16_IREE";;
  32x32x32)  m=32; k=32; n=32;  FAST=""; MMDEF="";;
  *) echo "unknown TILE=$TILE (want 64x32x128 or 32x32x32)"; exit 2;;
esac

FINAL="$MMW/build/final_512x1024x4096_${TILE}_8c.xclbin"
STUB_OUT="$MMW/build/final_512x1024x4096_${TILE}_8c_STUB.xclbin"
MMO="$MMW/build/mm_${m}x${k}x${n}.o"

restore_production() {
  echo "[stub] restoring production resident kernel ($TILE) ..."
  rm -f "$MMO"
  # rebuild only the chosen tile's production xclbins (real mm.cc)
  for NN in 1024 2048 4096; do
    WA_C_DEPTH=1 make -C "$MMW" AIECC_JOBS="${AIECC_JOBS:-0}" NPU2=1 M=512 K=1024 N="$NN" \
      m=$m k=$k n=$n dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 $FAST \
      "build/final_512x1024x${NN}_${TILE}_8c.xclbin" >/dev/null
  done
}
trap restore_production EXIT

# 1. production must exist (the FULL twin the harness loads).
if [[ ! -f "$FINAL" ]]; then
  echo "[stub] production $FINAL missing -- building it first ..."
  scripts/build_parakeet_kernels.sh
fi

# 2. compile the movement-only stub OVER the production matmul object.
echo "[stub] compiling mm_movement_stub.cc -> $MMO ($TILE) ..."
"$PEANO_INSTALL_DIR/bin/clang++" -O2 -std=c++20 --target=aie2p-none-unknown-elf -DNDEBUG \
  -I "$MAD/include" -Dbf16_f32_ONLY -DDIM_M=$m -DDIM_K=$k -DDIM_N=$n -DVECTORIZED_ONLY $MMDEF \
  -c route_b_kernels/occupancy/mm_movement_stub.cc -o "$MMO"
touch "$MMO"  # newer than mm.cc so make won't rebuild from the real source

# 3. relink the N=4096 resident xclbin with the stub object, save as _STUB.
echo "[stub] relinking resident xclbin with the stub object ..."
rm -f "$FINAL"
WA_C_DEPTH=1 make -C "$MMW" AIECC_JOBS="${AIECC_JOBS:-0}" NPU2=1 M=512 K=1024 N=4096 \
  m=$m k=$k n=$n dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 $FAST \
  "build/final_512x1024x4096_${TILE}_8c.xclbin"
cp -f "$FINAL" "$STUB_OUT"
echo "[stub] wrote $STUB_OUT"
# trap restores the production FINAL on exit.
echo "[stub] done (production restored on exit)."
