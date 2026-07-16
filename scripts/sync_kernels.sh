#!/usr/bin/env bash
# Copy our canonical custom kernels/designs FORWARD into the mlir-aie build sandbox.
#
# DIRECTION MATTERS: route_b_kernels/ (tracked, real source files) is the SINGLE
# SOURCE OF TRUTH. mlir-aie/ is a gitignored, disposable build sandbox. This copies
# repo -> sandbox (one-directional), so there is no drift: you ALWAYS edit
# route_b_kernels/, NEVER the mlir-aie copy (it's just a build input, recreated here).
# Real files (not symlinks) so mlir-aie's relative-path Makefiles/includes work.
# Idempotent; called by setup_route_b.sh and build_kernels.sh.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
RB=route_b_kernels
# Target mlir-aie root: default = the submodule (back-compat); pass an arg to overlay a clean fork-branch
# checkout instead (Phase-2 policy B -- kernels stay route_b-authored, synced into the build source).
AIEROOT="${1:-mlir-aie}"
PE=$AIEROOT/programming_examples
MM=$PE/basic/matrix_multiplication
K=$AIEROOT/aie_kernels/aie2p

[ -d "$AIEROOT" ] || { echo "$AIEROOT not present — run scripts/setup_route_b.sh first" >&2; exit 1; }
mkdir -p "$PE/ml/dwconv1d" "$PE/ml/softmax400" "$PE/ml/layernorm" "$PE/ml/relpos_mha"

# dwconv1d k=5 (docs/08) — last missing Conformer primitive
cp "$RB/dwconv1d/dwconv1d.cc" "$K/dwconv1d.cc"
cp "$RB/dwconv1d/dwconv1d.py" "$PE/ml/dwconv1d/dwconv1d.py"
cp "$RB/dwconv1d/Makefile"    "$PE/ml/dwconv1d/Makefile"
# FUSED dwconv->SiLU in one xclbin (conv step 3+4): dwconv f32-out core -> on-chip
# f32 fifo -> silu core (two separate cores, immune to the alt-channel miscompile). Needs both
# dwconv1d.cc (above) and silu_brick.cc (synced below) in $K.
cp "$RB/dwconv1d/dwconv_silu_iron.py" "$PE/ml/dwconv1d/dwconv_silu_iron.py"
cp "$RB/dwconv1d/Makefile.dwsilu"     "$PE/ml/dwconv1d/Makefile.dwsilu"
# TIME-MAJOR fused dwconv->SiLU (conv step 3b, transpose-dissolving): [T,D] dwconv (dwconv1d_k9_tmajor)
# -> on-chip f32 fifo -> silu core. Consumes GLU's [T,D] directly + emits pw2's [T,D] directly, so BOTH
# host transposes are dissolved (no shuffle, no cross-column DMA -> no n-D-DMA hang). Same dwconv1d.cc +
# silu_brick.cc objects as the channel-major fused brick.
cp "$RB/dwconv1d/dwconv_silu_tmajor_iron.py" "$PE/ml/dwconv1d/dwconv_silu_tmajor_iron.py"
cp "$RB/dwconv1d/Makefile.dwsilu_t"          "$PE/ml/dwconv1d/Makefile.dwsilu_t"
# on-chip COMPUTE-tile transpose brick (conv step 3b enabler): element transpose in-core
# (transpose_tile.cc), shim DMA does only contiguous read + unit-inner-stride scatter --
# AVOIDS the transposing n-D DMA that hangs when co-resident (blocker npu.rs:740).
cp "$RB/aie_kernels/transpose_tile.cc"  "$K/transpose_tile.cc"
cp "$RB/dwconv1d/transpose_iron.py"     "$PE/ml/dwconv1d/transpose_iron.py"
cp "$RB/dwconv1d/Makefile.transpose"    "$PE/ml/dwconv1d/Makefile.transpose"
# fused bias+SiLU / narrow epilogue kernel (docs/10)
cp "$RB/aie_kernels/mm_silu_epilogue.cc" "$K/mm_silu_epilogue.cc"
# softmax-400 (pad->416) example
cp "$RB/softmax400/softmax400.py" "$PE/ml/softmax400/softmax400.py"
cp "$RB/softmax400/Makefile"      "$PE/ml/softmax400/Makefile"
# relpos MHA scores->softmax STANDALONE step-1 kernel (rel_shift strided-read +
# vectorized-exp2 softmax, no matmul) -- de-risks the two hard rel-pos bricks.
cp "$RB/relpos_mha/relpos_mha.cc"                    "$K/relpos_mha.cc"
cp "$RB/relpos_mha/relpos_scores_softmax_iron.py"    "$PE/ml/relpos_mha/relpos_scores_softmax_iron.py"
# relpos MHA STEP-2 composed kernel: on-chip AC=qu@k^T matmul feeding the resident
# L1 f32 score tile -> scores->softmax (first resident-block test). make STEP=2.
cp "$RB/relpos_mha/relpos_ac_scores_softmax_iron.py" "$PE/ml/relpos_mha/relpos_ac_scores_softmax_iron.py"
# relpos MHA STEP-3 composed kernel: BOTH AC=qu@k^T and BD=qv@p^T on-chip (resident).
cp "$RB/relpos_mha/relpos_qkp_scores_softmax_iron.py" "$PE/ml/relpos_mha/relpos_qkp_scores_softmax_iron.py"
cp "$RB/relpos_mha/relpos_ctx_iron.py" "$PE/ml/relpos_mha/relpos_ctx_iron.py"
cp "$RB/relpos_mha/relpos_full_iron.py" "$PE/ml/relpos_mha/relpos_full_iron.py"
# relpos MHA STEP-6: ROW-TILED, MemTile-staged block (T up to 172). k/p/V staged in
# L2, T query rows tiled by TQ with the GLOBAL-index rel_shift. make STEP=6 T=172.
cp "$RB/relpos_mha/relpos_rowtiled_iron.py" "$PE/ml/relpos_mha/relpos_rowtiled_iron.py"
cp "$RB/relpos_mha/relpos_kpvstream_iron.py" "$PE/ml/relpos_mha/relpos_kpvstream_iron.py"
cp "$RB/relpos_mha/relpos_rowtiled_stream_iron.py" "$PE/ml/relpos_mha/relpos_rowtiled_stream_iron.py"
cp "$RB/relpos_mha/probe_floor_iron.py"              "$PE/ml/relpos_mha/probe_floor_iron.py"
cp "$RB/relpos_mha/Makefile"                         "$PE/ml/relpos_mha/Makefile"
# plain resident whole_array matmul (no epilogue) -- MLIR-emitting generator +
# Makefile.resident (route_b_override .txt-insts + WA_C_DEPTH flow) for the Parakeet
# resident encoder tiles and the thin-M decode GEMV (build_parakeet/decode_kernels.sh).
cp "$RB/whole_array_fused/whole_array_iron.py"      "$MM/whole_array/whole_array_iron.py"
cp "$RB/whole_array_fused/Makefile.resident"        "$MM/whole_array/Makefile.resident"
# whole_array fused matmul+epilogue design
cp "$RB/whole_array_fused/whole_array_silu_iron.py" "$MM/whole_array/whole_array_silu_iron.py"
cp "$RB/whole_array_fused/Makefile.silu"            "$MM/whole_array/Makefile.silu"
cp "$RB/whole_array_fused/whole_array_modal_iron.py" "$MM/whole_array/whole_array_modal_iron.py"
cp "$RB/whole_array_fused/Makefile.modal"            "$MM/whole_array/Makefile.modal"
# L3 — int8 modal: on-chip i32->f32 dequant epilogue (internal notes)
cp "$RB/whole_array_fused/whole_array_modal_int8_iron.py" "$MM/whole_array/whole_array_modal_int8_iron.py"
cp "$RB/whole_array_fused/Makefile.modal.int8"           "$MM/whole_array/Makefile.modal.int8"
# single_core fused GEMM->GEMM (on-chip intermediate) design
cp "$RB/ffn_gemm2/ffn_gemm2_iron.py" "$MM/single_core/ffn_gemm2_iron.py"
cp "$RB/ffn_gemm2/Makefile.ffn"      "$MM/single_core/Makefile.ffn"
# M-stationary GEMM probe (internal notes; KILLED but kept reproducible) — bin/mstat_probe.rs
cp "$RB/m_stationary/m_stationary_iron.py" "$MM/whole_array/m_stationary_iron.py"
cp "$RB/m_stationary/Makefile.mstat"       "$MM/whole_array/Makefile.mstat"
# M-stationary GEMM + fused LayerNorm epilogue (Phase 1.2 spike) — bin/mstat_ln_probe.rs
cp "$RB/aie_kernels/mm_ln_epilogue.cc"         "$K/mm_ln_epilogue.cc"
cp "$RB/m_stationary/m_stationary_ln_iron.py"  "$MM/whole_array/m_stationary_ln_iron.py"
cp "$RB/m_stationary/Makefile.mstatln"         "$MM/whole_array/Makefile.mstatln"
# ctxLN — encoder LayerNorm on the NPU (Step D, internal notes): f32 two-pass kernel + design
cp "$RB/aie_kernels/ln_2pass.cc"     "$K/ln_2pass.cc"
cp "$RB/ctx_ln/ctx_ln_iron.py"       "$PE/ml/layernorm/ctx_ln_iron.py"
cp "$RB/ctx_ln/Makefile.ctxln"       "$PE/ml/layernorm/Makefile.ctxln"
# device-side f32->bf16 cast (resident-rails seam primitive)
cp "$RB/aie_kernels/cast_f32_bf16.cc" "$K/cast_f32_bf16.cc"
cp "$RB/ctx_ln/cast_f32_bf16_iron.py" "$PE/ml/layernorm/cast_f32_bf16_iron.py"
cp "$RB/ctx_ln/Makefile.cast"         "$PE/ml/layernorm/Makefile.cast"
# device-side affine + f32->bf16 cast (resident-rails LN affine seam)
cp "$RB/aie_kernels/affine_cast.cc"   "$K/affine_cast.cc"
cp "$RB/ctx_ln/affine_cast_iron.py"   "$PE/ml/layernorm/affine_cast_iron.py"
cp "$RB/ctx_ln/Makefile.affinecast"   "$PE/ml/layernorm/Makefile.affinecast"
cp "$RB/ctx_ln/deint_cast_iron.py"   "$PE/ml/layernorm/deint_cast_iron.py"
cp "$RB/ctx_ln/Makefile.deint"        "$PE/ml/layernorm/Makefile.deint"
# device-side GLU (conv-module gate step): a*sigmoid(g) over pw1's [T,2D] -> [T,D]
cp "$RB/aie_kernels/glu.cc"           "$K/glu.cc"
cp "$RB/ctx_ln/glu_iron.py"           "$PE/ml/layernorm/glu_iron.py"
cp "$RB/ctx_ln/Makefile.glu"          "$PE/ml/layernorm/Makefile.glu"
# device-side f32 accumulate-add (resident-FFN fc2 on-device K-split accum): out = a + b, f32
cp "$RB/aie_kernels/acc_add.cc"       "$K/acc_add.cc"
cp "$RB/ctx_ln/acc_add_iron.py"       "$PE/ml/layernorm/acc_add_iron.py"
cp "$RB/ctx_ln/Makefile.accadd"       "$PE/ml/layernorm/Makefile.accadd"
# post-dwconv SiLU brick (conv step 4) -- SEPARATE single-op-loop brick (immune to the
# fused-epilogue per-channel-loop miscompile; see dwconv-fused-epilogue-alt-channel-miscompile).
cp "$RB/ctx_ln/silu_brick.cc"         "$K/silu_brick.cc"
cp "$RB/ctx_ln/silu_iron.py"          "$PE/ml/layernorm/silu_iron.py"
cp "$RB/ctx_ln/Makefile.silu2"        "$PE/ml/layernorm/Makefile.silu2"
# mha_decode — on-chip single-query MHA for the Whisper decoder (M1 Task 0): kernel + design
mkdir -p "$PE/ml/mha_decode"
cp "$RB/mha_decode/mha_decode.cc"      "$K/mha_decode.cc"
cp "$RB/mha_decode/mha_decode_iron.py" "$PE/ml/mha_decode/mha_decode_iron.py"
cp "$RB/mha_decode/Makefile.mha"       "$PE/ml/mha_decode/Makefile.mha"

echo "synced route_b_kernels/ -> mlir-aie build sandbox (edit route_b_kernels/, never mlir-aie/)"
