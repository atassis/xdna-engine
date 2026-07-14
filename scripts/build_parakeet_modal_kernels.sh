#!/usr/bin/env bash
# Build the Parakeet MODAL resident xclbin + per-N instruction streams (A1: ff_act on-chip).
#
# The zero-switch resident design (npu-parakeet src/npu.rs) runs the WHOLE encoder on ONE
# resident K=1024 whole_array xclbin, dispatching per-N instruction streams on it. This script
# builds the MODAL variant of that xclbin: a fused f32-out epilogue whose RTP (baked per inst
# stream) selects silu(1) vs identity(0). The N=4096 stream (fc1 / ff.l1 only) is baked SILU;
# N=1024/2048 (every other GEMM) are baked IDENTITY (numerically the plain matmul). This moves
# the FFN SiLU activation (`ff_act`, the #1 host-compute lever) onto the NPU with ZERO extra
# hw-context switches (one xclbin, mode lives in the runtime inst stream).
#
# Fast BFP16_IREE tile 64x32x128 only (matches the shipped resident). Bias is NOT folded
# (Parakeet fc1 has no host bias), so no K-augmentation -- the modal xclbin stays K=1024.
#
# CPU-only (no NPU). Toolchain (.venv-iron + Peano + patched submodule) lives in the MAIN
# worktree; outputs land in the gitignored whole_array/build/ (same dir as the plain resident).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
source scripts/kernel_sandbox.sh
MMW=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
ensure_fresh_sandbox "$MMW/build"           # wipe old-pin xclbins/objects on a toolchain change
bash scripts/sync_kernels.sh >/dev/null     # copies Makefile.modal + whole_array_modal_iron.py + mm_silu_epilogue.cc

MK="-f Makefile.modal"
COMMON="NPU2=1 M=512 K=1024 m=64 k=32 n=128 dtype_in=bf16 dtype_out=f32 n_aie_cols=8 use_iron=1 \
        emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1"

# --- SILU mode, N=4096 : the resident xclbin + the fc1 (ff.l1) instruction stream ---
rm -f $MMW/build/aie_512x1024x4096_64x32x128_8c_modalsilu.mlir
echo "== MODAL-SILU 512x1024x4096 64x32x128 8c (resident xclbin + fc1 stream) =="
WA_C_DEPTH=1 make $MK -C $MMW AIECC_JOBS="${AIECC_JOBS:-0}" $COMMON N=4096 \
   build/final_512x1024x4096_64x32x128_8c_modalsilu.xclbin

# --- IDENTITY mode, N=1024 and N=2048 : every other encoder GEMM's instruction stream ---
for N in 1024 2048; do
  rm -f $MMW/build/aie_512x1024x${N}_64x32x128_8c_modalid.mlir
  echo "== MODAL-ID 512x1024x${N} 64x32x128 8c (identity stream) =="
  WA_C_DEPTH=1 make $MK -C $MMW AIECC_JOBS="${AIECC_JOBS:-0}" $COMMON N="$N" no_silu=1 \
     build/final_512x1024x${N}_64x32x128_8c_modalid.xclbin
done

echo "Built Parakeet MODAL resident xclbin (silu@4096) + identity insts (1024/2048)."
ls -la $MMW/build/final_512x1024x4096_64x32x128_8c_modalsilu.xclbin \
       $MMW/build/insts_512x1024x4096_64x32x128_8c_modalsilu.txt \
       $MMW/build/insts_512x1024x1024_64x32x128_8c_modalid.txt \
       $MMW/build/insts_512x1024x2048_64x32x128_8c_modalid.txt

# --- RESIDENT-LN SEAM (DEFAULT LN->fc1 on-NPU): ctxLN(normalize-only) + affine_cast(gamma,beta) at
#     PAD_M x KRES = 512 x 1024. Loaded co-resident by npu.rs::resident_ln; the encoder defaults to
#     the device-side LN->fc1 seam when these are present (opt out: PARAKEET_RESIDENT_FF=0). ---
LNML=mlir-aie/programming_examples/ml/layernorm
LNDIR=artifacts/parakeet/ln
mkdir -p "$LNDIR"
echo "== RESIDENT-LN: ctxLN + affine_cast (+ plain cast) 512x1024 =="
make -C $LNML -f Makefile.ctxln      NPU2=1 rows=512 cols=1024 build/final_ctxln_512x1024.xclbin
make -C $LNML -f Makefile.affinecast NPU2=1 rows=512 cols=1024 build/final_affcast_512x1024.xclbin
make -C $LNML -f Makefile.cast       NPU2=1 rows=512 cols=1024 build/final_cast_512x1024.xclbin
for tag in ctxln_512x1024 affcast_512x1024 cast_512x1024; do
  cp "$LNML/build/final_${tag}.xclbin" "$LNML/build/insts_${tag}.txt" "$LNDIR/"
done
echo "Built + staged resident-LN seam xclbins -> $LNDIR"
ls -la $LNDIR/
