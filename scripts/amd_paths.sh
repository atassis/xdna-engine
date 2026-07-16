#!/usr/bin/env bash
# amd_paths.sh -- single relocatable anchor for the AMD/Xilinx upstream checkouts.
#
# SOURCE this from any script that needs the IRON / XRT / mlir-air checkouts. It
# derives the umbrella-workspace root from THIS file's own location (the upstream
# checkouts live as siblings of the engine repo under the workspace), so the whole
# tree is RELOCATABLE -- no hardcoded $HOME/absolute paths. Every var is overridable
# from the environment (export IRON_DIR=... before sourcing to point elsewhere).
#
#   source "$(dirname "${BASH_SOURCE[0]}")/amd_paths.sh"
#   ... use "$IRON_DIR" / "$XRT_SRC_DIR" / "$MLIR_AIR_DIR" / "$AIEBU_ASM_DIR"
#
# Layout it assumes:  <workspace>/{xdna-engine/scripts/amd_paths.sh, IRON, XRT-src, mlir-air, ...}
#                     (upstream checkouts are flat siblings of the engine repo at the workspace root)

# workspace root = parent of the engine repo (this file lives in <engine>/scripts/)
XDNA_WS="${XDNA_WS:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}"
export XDNA_WS

export IRON_DIR="${IRON_DIR:-$XDNA_WS/IRON}"
export XRT_SRC_DIR="${XRT_SRC_DIR:-$XDNA_WS/XRT-src}"
export AIEBU_ASM_DIR="${AIEBU_ASM_DIR:-$XRT_SRC_DIR/src/runtime_src/core/common/aiebu/build/Release/src/cpp/utils/asm}"

# mlir-air / llvm-aie are NOT defaulted here: in setup_amd_toolchains.sh an EMPTY
# MLIR_AIR_DIR/LLVM_AIE_DIR is the "do not patch this repo" gate. Their canonical
# location (when you do opt in) is $XDNA_WS/{mlir-air,llvm-aie}.
