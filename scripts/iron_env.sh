# Source this to build/run mlir-aie examples on this CachyOS box.
#   source scripts/iron_env.sh
# Then in an example dir:  make NPU2=1            (build xclbin, CPU)
#                          make NPU2=1 <name>.exe (build host harness, CPU)
#                          make NPU2=1 run        (run on NPU — stop flm-asr.service first)
REPO="$REPO"
export PATH="$REPO/.venv-iron/bin:$REPO/.venv-iron/cc-shim:$PATH"   # aiecc + gcc-13 shims
export PEANO_INSTALL_DIR="$REPO/.venv-iron/lib/python3.14/site-packages/llvm-aie"
# Arch xrt cmake export is broken (missing static .a); point common.cmake at the shared .so
export XRT_INC_DIR=/usr/include
export XRT_LIB_DIR=/usr/lib
