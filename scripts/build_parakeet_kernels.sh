#!/usr/bin/env bash
# Build the Parakeet NPU matmul xclbins (CPU-only; no NPU). Phase 3 / production engine.
#
# The zero-switch resident design (npu-parakeet src/npu.rs) runs the WHOLE encoder on ONE
# resident K=1024 whole_array xclbin (the N=4096 build), dispatching per-N instruction streams
# (N=1024/2048/4096) on it — zero hw-context switches. ff.l2's K=4096 is K-split into 4× K=1024.
# Two tiles are built so the resident kernel is selectable at runtime:
#   - FAST BFP16_IREE  64x32x128  (default; ~2× native; lossier — WER-lossless but rel~0.49/24 blocks)
#   - NATIVE bf16      32x32x32   (NPU_NATIVE=1; accurate, rel~5e-2 < 0.08 bar)
# n=128 / n=32 both divide 1024/2048/4096. The resident xclbin is the N=4096 one of the chosen tile;
# the N=1024/2048 builds are only for their instruction streams (run on the resident kernel).
#
# The mlir-aie toolchain (.venv-iron + Peano + patched submodule) lives in the MAIN worktree; run
# from there (owner-approved 2026-06-13). Outputs land in main's gitignored whole_array/build/.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
source scripts/kernel_sandbox.sh
MMW=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
ensure_fresh_sandbox "$MMW/build"           # wipe old-pin xclbins/objects on a toolchain change
bash scripts/sync_kernels.sh >/dev/null   # copy whole_array_iron.py + Makefile.resident into the sandbox

# The blessed toolchain instance rewrote matrix_multiplication/makefile-common to an @iron.jit flow
# (stock whole_array.py) that (a) names insts .bin -- the engine reads insts_*.txt -- and (b) has no
# WA_C_DEPTH knob, so the wide fast tile (64x32x128) overflows L1. So drive the MLIR-emitting
# whole_array_iron.py through Makefile.resident (route_b_override .txt-insts + WA_C_DEPTH flow).
MK="-f Makefile.resident"

# --- FAST BFP16_IREE, tile 64x32x128 (default resident) ---
rm -f $MMW/build/mm_64x32x128.o
for N in 1024 2048 4096; do
  rm -f $MMW/build/aie_512x1024x${N}_64x32x128_8c.mlir
  echo "== FAST BFP16 512x1024x${N} 64x32x128 8c =="
  WA_C_DEPTH=1 make $MK -C $MMW AIECC_JOBS="${AIECC_JOBS:-0}" NPU2=1 M=512 K=1024 N="$N" m=64 k=32 n=128 \
     dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 \
     emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 \
     build/final_512x1024x${N}_64x32x128_8c.xclbin
done

# --- NATIVE bf16, tile 32x32x32 (accurate; NPU_NATIVE=1) ---
rm -f $MMW/build/mm_32x32x32.o
for N in 1024 2048 4096; do
  rm -f $MMW/build/aie_512x1024x${N}_32x32x32_8c.mlir
  echo "== NATIVE bf16 512x1024x${N} 32x32x32 8c =="
  make $MK -C $MMW AIECC_JOBS="${AIECC_JOBS:-0}" NPU2=1 M=512 K=1024 N="$N" m=32 k=32 n=32 dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 \
       build/final_512x1024x${N}_32x32x32_8c.xclbin
done
echo "Built Parakeet resident xclbins + per-N instruction streams (fast + native tiles)."
