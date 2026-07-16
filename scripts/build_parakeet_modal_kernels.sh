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
# RESIDENT-CONV: GLU gate (conv-module step 2). a*sigmoid(g) over pw1's [T,2D] -> [T,D], cols=D=1024.
echo "== RESIDENT-CONV: GLU 512x1024 =="
make -C $LNML -f Makefile.glu        NPU2=1 rows=512 cols=1024 build/final_glu_512x1024.xclbin
# RESIDENT-FFN: f32 accumulate-add brick (fc2 on-device K-split accum). out[t,c]=a[t,c]+b[t,c], f32.
# Sums the DFF/KRES fc2 partials into ONE device BO so the FFN output stays device-resident
# (deletes the host Array2 accumulation; bit-identical to the host sequential f32 K-split). rows=PAD_M, cols=KRES.
echo "== RESIDENT-FFN: acc_add 512x1024 =="
make -C $LNML -f Makefile.accadd     NPU2=1 rows=512 cols=1024 build/final_accadd_512x1024.xclbin
# RESIDENT-BLOCK: scaled residual-add (whole-block fusion residual). out[t,c]=a[t,c]+scale*b[t,c], f32.
# s050 (scale=0.5) = the Macaron FFN residual x+0.5*ff. scale is baked (2-input-DMA limit -> one xclbin
# per scale); the full-residual s100 (scale=1.0) is built when the MHSA/conv seams need it. rows=PAD_M, cols=KRES.
echo "== RESIDENT-BLOCK: residual_add s050 (x+0.5*ff) 512x1024 =="
make -C $LNML -f Makefile.resadd     NPU2=1 rows=512 cols=1024 scale=0.5 stag=050 build/final_resadd_512x1024_s050.xclbin
# RESIDENT-CONV: post-dwconv SiLU brick (conv-module step 4). silu(x) over [C,T]=[1024,400] f32 -> f32.
# SEPARATE single-op-loop brick (NOT a dwconv epilogue -- the fused epilogue miscompiles alternate
# channels on this toolchain; see dwconv-fused-epilogue-alt-channel-miscompile). rows=C, cols=T.
echo "== RESIDENT-CONV: SiLU brick 1024x400 =="
make -C $LNML -f Makefile.silu2      NPU2=1 rows=1024 cols=400 build/final_silu_1024x400.xclbin
# full FFN fc1->fc2 device-side: cast@DFF (fc1 f32 [T,4096] -> bf16) + the K=4096 fc2 resident (identity)
echo "== RESIDENT-FFN: cast@4096 + K=4096 fc2 (identity) =="
make -C $LNML -f Makefile.cast       NPU2=1 rows=512 cols=4096 build/final_cast_512x4096.xclbin
WA_C_DEPTH=1 make -C $MMW -f Makefile.modal NPU2=1 M=512 K=4096 N=1024 m=64 k=32 n=128 n_aie_cols=8 \
  emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 no_silu=1 build/final_512x4096x1024_64x32x128_8c_modalid.xclbin
for tag in ctxln_512x1024 affcast_512x1024 cast_512x1024 cast_512x4096 glu_512x1024 accadd_512x1024 resadd_512x1024_s050 silu_1024x400; do
  cp "$LNML/build/final_${tag}.xclbin" "$LNML/build/insts_${tag}.txt" "$LNDIR/"
done
cp "$MMW/build/final_512x4096x1024_64x32x128_8c_modalid.xclbin" "$MMW/build/insts_512x4096x1024_64x32x128_8c_modalid.txt" "$LNDIR/"
# RESIDENT-CONV: depthwise conv1d (conv-module step 3), k9/C=1024/T=400, 8 columns. Builds the VECTORIZED
# aligned-load + aie::shuffle_down_fill FIR (dwconv1d_shift) -- correct + fast (~7x the scalar). It avoids
# both toolchain traps the naive brick hits (unaligned-L1-load snap; aie::sliding_mul_ops bf16 inf/nan --
# see the step-3 dwconv investigation). Scalar fallback available with -DDWCONV_SCALAR=1. Distinct insts.bin ABI.
echo "== RESIDENT-CONV: dwconv1d k9 (vectorized aligned+shuffle) 1024x400 =="
DWML=mlir-aie/programming_examples/ml/dwconv1d
make -C $DWML NPU2=1 cols=8 build/final.xclbin
cp "$DWML/build/final.xclbin" "$LNDIR/final_dwconv_1024x400.xclbin"
cp "$DWML/build/insts.bin"    "$LNDIR/insts_dwconv_1024x400.txt"
# RESIDENT-CONV: FUSED dwconv->SiLU in ONE xclbin (conv step 3+4). Collapses the separate
# dwconv + silu xclbins into a single resident hw-context (dwconv f32-out core -> on-chip f32 fifo ->
# silu core), so the on-NPU post-dwconv SiLU costs no extra hw-context switch / host round-trip.
echo "== RESIDENT-CONV: FUSED dwconv+silu 1024x400 (one xclbin) =="
make -C $DWML -f Makefile.dwsilu NPU2=1 cols=8 build/final_dwconv_silu_1024x400.xclbin
cp "$DWML/build/final_dwconv_silu_1024x400.xclbin" "$LNDIR/final_dwconv_silu_1024x400.xclbin"
cp "$DWML/build/insts_dwconv_silu_1024x400.txt"    "$LNDIR/insts_dwconv_silu_1024x400.txt"
# RESIDENT-CONV: TIME-MAJOR fused dwconv->SiLU (conv step 3b). [T,D] layout -> consumes GLU's [T,D]
# directly + emits pw2's [T,D] directly, DISSOLVING both host transposes (no shuffle / cross-column
# DMA -> no n-D-DMA hang). Same two-core pipeline; when present the Rust conv path prefers it.
echo "== RESIDENT-CONV: TIME-MAJOR fused dwconv+silu 1024x400 (transposes dissolved) =="
make -C $DWML -f Makefile.dwsilu_t NPU2=1 cols=8 build/final_dwconv_silu_t_1024x400.xclbin
cp "$DWML/build/final_dwconv_silu_t_1024x400.xclbin" "$LNDIR/final_dwconv_silu_t_1024x400.xclbin"
cp "$DWML/build/insts_dwconv_silu_t_1024x400.txt"    "$LNDIR/insts_dwconv_silu_t_1024x400.txt"
echo "Built + staged resident FFN + conv xclbins (LN->fc1 seam + fc1->fc2 + GLU + dwconv + fused dwconv-silu [+ time-major]) -> $LNDIR"
ls -la $LNDIR/
