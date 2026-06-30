#!/usr/bin/env bash
# Build the extra sweep points for the DMA-occupancy regression (CPU-only; no NPU, but
# TOUCHES the shared amd/IRON checkout -> serialize, run in a coordinated window).
#
# The 3-point N-sweep on the resident K=1024 xclbin already gives a clean 2-param fit
# (~115us fixed + ~59 GB/s effective DMA). This builds the points that TIGHTEN it:
#   (A) N-sweep fill-in: N=3072 golden + insts on the EXISTING resident xclbin (cheap;
#       insts only, no new xclbin) -> 4 N points for a better-conditioned fit.
#   (B) K-sweep / tile-sweep: new FULL+STUB xclbins at a different bytes-per-transfer
#       ratio -> breaks the N-sweep collinearity so the 3-param fit can SEPARATE pure
#       per-byte DMA (c2) from per-transfer BD/lock overhead (c1). Heavier (full builds).
#
# After building, run (in the SAME window, NPU free):
#   scripts/parakeet_dma_occupancy_harness.py --sweep-N 1024 2048 3072 4096
# or wrap with the service-stop/fuser/restart discipline of run_parakeet_occupancy.sh.
#
# Usage:
#   scripts/build_parakeet_dma_sweep.sh            # (A) N-sweep fill-in only (default, cheap)
#   SWEEP=K scripts/build_parakeet_dma_sweep.sh    # (A) + (B) K-sweep tile points (heavier)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
MMW=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
TILE="${TILE:-64x32x128}"; m=64; k=32; n=128
OUTDIR=artifacts/parakeet/occupancy; mkdir -p "$OUTDIR"

gen_golden() {  # $1=M $2=K $3=N  -- bf16 A,B + f32 ref (reuses the occupancy golden logic)
  .venv-iron/bin/python - "$1" "$2" "$3" "$OUTDIR" <<'PY'
import sys, os, numpy as np
from ml_dtypes import bfloat16
M,K,N,outdir = int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
p=os.path.join(outdir,f"golden_{M}x{K}x{N}.npz")
if os.path.exists(p): print(f"[golden] {p} exists, skip"); sys.exit()
rng=np.random.RandomState(0)
A=rng.uniform(-1,1,size=(M,K)).astype(bfloat16); B=rng.uniform(-1,1,size=(K,N)).astype(bfloat16)
ref=A.astype(np.float32)@B.astype(np.float32)
np.savez(p, A=A.view(np.uint16), B=B.view(np.uint16), ref=ref); print(f"[golden] wrote {p}")
PY
}

build_point() {  # $1=M $2=K $3=N  -- FAST BFP16 tile, produces final xclbin + insts
  local M=$1 K=$2 N=$3
  echo "== build FAST BFP16 ${M}x${K}x${N} ${TILE} 8c =="
  WA_C_DEPTH=1 make -C "$MMW" AIECC_JOBS="${AIECC_JOBS:-0}" NPU2=1 M="$M" K="$K" N="$N" m=$m k=$k n=$n \
     dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 \
     emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 \
     "build/final_${M}x${K}x${N}_${TILE}_8c.xclbin"
}

echo "### (A) N-sweep fill-in: N=3072 golden + insts (resident K=1024 xclbin) ###"
gen_golden 512 1024 3072
# N=3072 needs its instruction stream; the build emits insts_512x1024x3072 alongside the
# xclbin (the harness dispatches it on the existing resident N=4096 fast xclbin).
build_point 512 1024 3072

if [[ "${SWEEP:-}" == "K" ]]; then
  echo "### (B) K-sweep tile points: vary bytes-per-transfer to split DMA from BD/lock ###"
  # K=512 and K=2048 at fixed N keep the same OUTPUT tile structure but halve/double the
  # input k-blocks -> bytes-per-output-tile changes, breaking the N-sweep collinearity.
  for KK in 512 2048; do
    for NN in 2048 4096; do
      gen_golden 512 "$KK" "$NN"
      build_point 512 "$KK" "$NN"
    done
  done
  echo "NOTE: K-sweep points also need their own FULL+STUB twins -- relink each via"
  echo "      scripts/build_parakeet_occupancy_stub.sh after building the FULL set."
fi
echo "Done. Run: scripts/parakeet_dma_occupancy_harness.py --sweep-N 1024 2048 3072 4096"
