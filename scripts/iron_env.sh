# Source this to build/run mlir-aie examples on this CachyOS box.
#   source scripts/iron_env.sh
# Then in an example dir:  make NPU2=1            (build xclbin, CPU)
#                          make NPU2=1 <name>.exe (build host harness, CPU)
#                          make NPU2=1 run        (run on NPU — stop flm-asr.service first)
# Self-derive the repo root. This file is SOURCED, so $0 is the shell, not the script:
# use BASH_SOURCE under bash and fall back to $0 under zsh (zsh sets $0 to the sourced path).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PATH="$REPO/.venv-iron/bin:$REPO/.venv-iron/cc-shim:$PATH"   # aiecc + gcc-13 shims
export PEANO_INSTALL_DIR="$REPO/.venv-iron/lib/python3.14/site-packages/llvm-aie"
# Arch xrt cmake export is broken (missing static .a); point common.cmake at the shared .so
export XRT_INC_DIR=/usr/include
export XRT_LIB_DIR=/usr/lib
