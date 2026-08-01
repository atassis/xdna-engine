# Source this to build/run mlir-aie examples on this CachyOS box.
#   source scripts/iron_env.sh
# Then in an example dir:  make NPU2=1            (build xclbin, CPU)
#                          make NPU2=1 <name>.exe (build host harness, CPU)
#                          make NPU2=1 run        (run on NPU — stop flm-asr.service first)
# Self-derive the repo root. This file is SOURCED, so $0 is the shell, not the script:
# use BASH_SOURCE under bash and fall back to $0 under zsh (zsh sets $0 to the sourced path).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PATH="$REPO/.venv-iron/bin:$REPO/.venv-iron/cc-shim:$PATH"   # aiecc + gcc-13 shims
INST="$("$REPO/scripts/toolchain_up.sh")" || { echo "iron_env: toolchain_up.sh failed" >&2; return 1; }
[ -n "$INST" ] || { echo "iron_env: empty instance dir from toolchain_up.sh" >&2; return 1; }
export PYTHONPATH="$INST/python:${PYTHONPATH:-}"   # aie resolves to the fork instance (place-tiles), not the wheel
export AIECC_PATH="$INST/bin/aiecc"
export MLIR_AIE_INSTANCE="$INST"   # kernel .cc sources resolve here, at the pin -- see route_b_override.mk
export PEANO_INSTALL_DIR="$REPO/.venv-iron/lib/python3.14/site-packages/llvm-aie"
# Arch xrt cmake export is broken (missing static .a); point common.cmake at the shared .so
export XRT_INC_DIR=/usr/include
export XRT_LIB_DIR=/usr/lib
# aiebu-asm assembles the control-code ELF (the `insts.elf` edge of aiecc). It ships with XRT, not
# with mlir-aie, and a distro XRT package may not install it -- when it is missing, aiecc runs the
# whole pipeline and only then dies with "tool 'aiebu-asm' not found in search paths or PATH", which
# reads like a toolchain bug rather than a missing dependency. Point AIEBU_ASM_DIR at the directory
# holding it (an XRT build tree's .../aiebu/build/Release/src/cpp/utils/asm works); otherwise try the
# standard XRT install. Silent when it is already on PATH.
if ! command -v aiebu-asm >/dev/null 2>&1; then
  for _d in "${AIEBU_ASM_DIR:-}" "${XILINX_XRT:-/opt/xilinx/xrt}/bin"; do
    [ -n "$_d" ] && [ -x "$_d/aiebu-asm" ] && { export PATH="$_d:$PATH"; break; }
  done
  unset _d
fi
# HuggingFace: make the shared big-partition cache explicit (default ~/.cache/huggingface is symlinked
# to /mnt/data). Respects an existing override; guarantees every HF flow reuses the one cache, no dup.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
