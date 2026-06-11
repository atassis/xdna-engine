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
PE=mlir-aie/programming_examples
MM=$PE/basic/matrix_multiplication
K=mlir-aie/aie_kernels/aie2p

[ -d mlir-aie ] || { echo "mlir-aie not present — run scripts/setup_route_b.sh first" >&2; exit 1; }
mkdir -p "$PE/ml/dwconv1d" "$PE/ml/softmax400"

# dwconv1d k=5 (docs/08) — last missing Conformer primitive
cp "$RB/dwconv1d/dwconv1d.cc" "$K/dwconv1d.cc"
cp "$RB/dwconv1d/dwconv1d.py" "$PE/ml/dwconv1d/dwconv1d.py"
cp "$RB/dwconv1d/Makefile"    "$PE/ml/dwconv1d/Makefile"
# fused bias+SiLU / narrow epilogue kernel (docs/10)
cp "$RB/aie_kernels/mm_silu_epilogue.cc" "$K/mm_silu_epilogue.cc"
# softmax-400 (pad->416) example
cp "$RB/softmax400/softmax400.py" "$PE/ml/softmax400/softmax400.py"
cp "$RB/softmax400/Makefile"      "$PE/ml/softmax400/Makefile"
# whole_array fused matmul+epilogue design
cp "$RB/whole_array_fused/whole_array_silu_iron.py" "$MM/whole_array/whole_array_silu_iron.py"
cp "$RB/whole_array_fused/Makefile.silu"            "$MM/whole_array/Makefile.silu"
# single_core fused GEMM->GEMM (on-chip intermediate) design
cp "$RB/ffn_gemm2/ffn_gemm2_iron.py" "$MM/single_core/ffn_gemm2_iron.py"
cp "$RB/ffn_gemm2/Makefile.ffn"      "$MM/single_core/Makefile.ffn"

echo "synced route_b_kernels/ -> mlir-aie build sandbox (edit route_b_kernels/, never mlir-aie/)"
