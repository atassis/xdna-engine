#!/usr/bin/env bash
# Build the K=768 GELU resident-rail xclbin set (BERT / Whisper-small / ESM-2 encoder FFN).
#
# PREP ARTIFACT -- NOT YET DEVICE-VALIDATED. This script stages the build recipe for the
# (K=768, DFF<=3072, act=GELU, residual-scale=1.0) resident rail that the three vanilla
# encoders share. Run it in the MAIN worktree (the toolchain -- .venv-iron + Peano + patched
# submodule -- lives there, same as build_parakeet_modal_kernels.sh). It builds ONLY on the
# CPU (aiecc place+route); it runs nothing on the NPU. Numerical validation (rel-L2 vs host
# truth) happens in the device session -- see the turnkey task doc referenced at the bottom.
#
# WHY these shapes (see the scout report generalization.md and the GELU-kernel read):
#   * Whisper-small d_model=768 ffn=3072, BERT-base hidden=768 intermediate=3072, ESM-2 padded
#     to K=768 by ctx2::round_stream -- ONE (K=768, N=3072, GELU) FFN rail serves all three.
#   * GELU epilogue = mm_gelu_epilogue_f32o (route_b_kernels/aie_kernels/mm_silu_epilogue.cc):
#     TANH-APPROX gelu, tanh(sqrt(2/pi)*(x+0.044715 x^3)), computed in bf16. Selected by the
#     modal RTP mode baked per instruction stream: rtp[0]=2 (gelu), set by the generator's
#     --gelu flag (whole_array_modal_iron.py, mode_val=2). One xclbin can also host the id
#     (rtp[0]=0) stream. NOTE: transformers-default BERT/Whisper use EXACT (erf) GELU, so this
#     tanh-approx + bf16 introduces an activation delta -- GATE on rel-L2 vs host truth, not WER.
#   * bias: gelu(A@B+bias) needs the bias INSIDE the activation, so fc1 folds its bias via the
#     modal K-augmentation (K_real=768 + one k=32 block = K_aug=800; this is exactly why
#     Makefile.modal defaults K=800). fc2 uses the identity epilogue, so its bias can be added
#     on the host AFTER the collapse (no K-aug); K stays 3072.
#   * fc2 is the K-collapse modal (like Parakeet's fc2_k4096) sized to K=3072, N=768. N=768 is
#     NOT a multiple of n*n_aie_cols=128*8=1024, so fc2 uses the single-core tile n=96
#     (768 = 96*8, one N-block; 64x32x96 satisfies m%r,k%s,n%t and (m*n)%16). fc1 keeps n=128.
#
# Opt-out / reuse: the same rail serves Whisper-small's encoder EXCEPT Whisper's encoder length
# is a fixed T=1500 > PAD_M=512, so Whisper needs a SECOND set at rows=1536 (a larger, slower
# build on the SAME kernel source). BERT short seqs (<=512) ride PAD_M=512 unchanged -- build
# BERT first (this script), reuse the K=768 kernels for Whisper by re-running with PAD_M=1536.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
source scripts/iron_env.sh
source scripts/kernel_sandbox.sh
MMW=mlir-aie/programming_examples/basic/matrix_multiplication/whole_array
ensure_fresh_sandbox "$MMW/build"           # wipe old-pin xclbins/objects on a toolchain change
bash scripts/sync_kernels.sh >/dev/null     # copies Makefile.modal + whole_array_modal_iron.py + mm_silu_epilogue.cc + ctx_ln/*

PAD_M="${PAD_M:-512}"                        # BERT: 512 (short seq). Whisper-small encoder: re-run with PAD_M=1536.
MK="-f Makefile.modal"
OUTDIR=artifacts/k768_gelu_rail
mkdir -p "$OUTDIR"

# --- fc1: modal GELU, K_aug=800 (768 real + 32 bias-fold block), N=3072, bfp16 fast tile 64x32x128 ---
# out = gelu(x @ W1 + b1), f32 out, tanh-approx gelu on chip (rtp[0]=2).
echo "== K768 fc1 GELU  ${PAD_M}x800x3072 64x32x128 8c (modalgelu) =="
rm -f "$MMW/build/aie_${PAD_M}x800x3072_64x32x128_8c_modalgelu.mlir"
WA_C_DEPTH=1 make $MK -C "$MMW" AIECC_JOBS="${AIECC_JOBS:-0}" \
  NPU2=1 M="$PAD_M" K=800 N=3072 m=64 k=32 n=128 n_aie_cols=8 use_iron=1 \
  emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 gelu=1 \
  "build/final_${PAD_M}x800x3072_64x32x128_8c_modalgelu.xclbin"

# --- fc2: K-collapse identity modal, K=3072, N=768, tile 64x32x96 (768 = 96*8, one N-block) ---
# out = h_bf16 @ W2  (identity epilogue, f32 out); host adds b2 after (bias outside identity is fine).
echo "== K768 fc2 collapse  ${PAD_M}x3072x768 64x32x96 8c (modalid) =="
rm -f "$MMW/build/aie_${PAD_M}x3072x768_64x32x96_8c_modalid.mlir"
WA_C_DEPTH=1 make $MK -C "$MMW" AIECC_JOBS="${AIECC_JOBS:-0}" \
  NPU2=1 M="$PAD_M" K=3072 N=768 m=64 k=32 n=96 n_aie_cols=8 use_iron=1 \
  emulate_bfloat16_mmul_with_bfp16=1 bfp16_iree=1 no_silu=1 \
  "build/final_${PAD_M}x3072x768_64x32x96_8c_modalid.xclbin"

# --- casts: f32 stream -> bf16 for each modal GEMM's A input ---
#   cast@768  : LN(x)  [T,768]  f32 -> bf16   (fc1 A input)
#   cast@3072 : gelu   [T,3072] f32 -> bf16   (fc2 A input)
LNML=mlir-aie/programming_examples/ml/layernorm
echo "== K768 casts  ${PAD_M}x768 + ${PAD_M}x3072 =="
make -C "$LNML" -f Makefile.cast NPU2=1 rows="$PAD_M" cols=768  "build/final_cast_${PAD_M}x768.xclbin"
make -C "$LNML" -f Makefile.cast NPU2=1 rows="$PAD_M" cols=3072 "build/final_cast_${PAD_M}x3072.xclbin"

# --- residual add, scale=1.0 (vanilla transformer full residual; s100 already exists at cols=1024) ---
echo "== K768 residual_add s100  ${PAD_M}x768 =="
make -C "$LNML" -f Makefile.resadd NPU2=1 rows="$PAD_M" cols=768 scale=1.0 stag=100 \
  "build/final_resadd_${PAD_M}x768_s100.xclbin"

# --- OPTIONAL (fuller seam, only when LN moves on-chip too): device LN bricks at cols=768. ---
# For the FIRST BERT advance the trailing post-norm LN can stay on host (cheap); the FFN
# cast->fc1(gelu)->cast->fc2 chain is the device advance. Uncomment to also stage on-chip LN:
#   make -C "$LNML" -f Makefile.ctxln      NPU2=1 rows="$PAD_M" cols=768 "build/final_ctxln_${PAD_M}x768.xclbin"
#   make -C "$LNML" -f Makefile.affinecast NPU2=1 rows="$PAD_M" cols=768 "build/final_affcast_${PAD_M}x768.xclbin"

# --- stage artifacts + instruction streams next to the Parakeet rail's layout ---
for tag in \
  "final_${PAD_M}x800x3072_64x32x128_8c_modalgelu" \
  "final_${PAD_M}x3072x768_64x32x96_8c_modalid"; do
  cp "$MMW/build/${tag}.xclbin" "$MMW/build/insts_${tag#final_}.txt" "$OUTDIR/" 2>/dev/null || true
done
for tag in "cast_${PAD_M}x768" "cast_${PAD_M}x3072" "resadd_${PAD_M}x768_s100"; do
  cp "$LNML/build/final_${tag}.xclbin" "$LNML/build/insts_${tag}.txt" "$OUTDIR/" 2>/dev/null || true
done

echo "Staged K=768 GELU resident-rail xclbins -> $OUTDIR"
ls -la "$OUTDIR/" || true
echo
echo "NEXT (device session): validate rel-L2 vs host truth, then wire under BertEncoder."
echo "Turnkey: xdna-engine-private/journal/docs/handoffs/active/2026-07-17-k768-gelu-rail-device.md"
