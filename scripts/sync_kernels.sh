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
# fused bias+SiLU / narrow epilogue kernel (docs/10)
cp "$RB/aie_kernels/mm_silu_epilogue.cc" "$K/mm_silu_epilogue.cc"
# softmax-400 (pad->416) example
cp "$RB/softmax400/softmax400.py" "$PE/ml/softmax400/softmax400.py"
cp "$RB/softmax400/Makefile"      "$PE/ml/softmax400/Makefile"
# relpos MHA scores->softmax STANDALONE step-1 kernel (rel_shift strided-read +
# vectorized-exp2 softmax, no matmul) -- de-risks the two hard rel-pos bricks.
cp "$RB/relpos_mha/relpos_mha.cc"                    "$K/relpos_mha.cc"
cp "$RB/relpos_mha/relpos_scores_softmax_iron.py"    "$PE/ml/relpos_mha/relpos_scores_softmax_iron.py"
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
# mha_decode — on-chip single-query MHA for the Whisper decoder (M1 Task 0): kernel + design
mkdir -p "$PE/ml/mha_decode"
cp "$RB/mha_decode/mha_decode.cc"      "$K/mha_decode.cc"
cp "$RB/mha_decode/mha_decode_iron.py" "$PE/ml/mha_decode/mha_decode_iron.py"
cp "$RB/mha_decode/Makefile.mha"       "$PE/ml/mha_decode/Makefile.mha"

echo "synced route_b_kernels/ -> mlir-aie build sandbox (edit route_b_kernels/, never mlir-aie/)"
