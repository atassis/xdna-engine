#!/usr/bin/env bash
# Build the aie2p cascade-accessor device test to an xclbin + host exe.
#
# Validates aie_api/cascade.hpp (aie::cascade_out / aie::cascade_in_i32) on real
# aie2p silicon: 3 cores in a cascade chain (kernel1 -> kernel2 -> kernel3),
# connected by aie.cascade_flow, seed 14 -> +100 -> +100, host expects buf[5]==214.
#
# RUN THIS IN A COORDINATED DEVICE WINDOW: it sources iron_env.sh, which uses the
# shared IRON toolchain instance the decode session also uses. Do the build while
# that session is paused (or serialize), same as any shared-toolchain work.
#
# Prereq: the fork aie_api with cascade.hpp must be on the kernel include path
# (FORK_AIE_API below); that is the atassis/aie_api feat/adf-free-cascade tree.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../../../../.." && pwd)"        # workspace root
FORK_AIE_API="$WS/aie_api-fork/include"          # cascade.hpp lives here
MLIR_AIE="$WS/mlir-aie"
TEST_LIB="$MLIR_AIE/runtime_lib/test_lib"

echo "== [0/3] environment =="
source "$WS/xdna-engine/scripts/iron_env.sh"     # sets PEANO_INSTALL_DIR, AIECC_PATH, PATH
CXX="$PEANO_INSTALL_DIR/bin/clang++"
AIECC="${AIECC_PATH:-aiecc.py}"
echo "  peano: $CXX"
echo "  aiecc: $AIECC"

echo "== [1/3] compile the 3 cascade kernels (Peano, aie2p, fork aie_api) =="
for k in kernel1 kernel2 kernel3; do
  "$CXX" --target=aie2p-none-unknown-elf -std=c++2b -Wno-parentheses -Wno-attributes \
         -O2 -I"$FORK_AIE_API" -c "$HERE/$k.cc" -o "$HERE/$k.o"
  echo "  built $k.o"
done

echo "== [2/3] aiecc -> cascade.xclbin + insts.bin (peano backend) =="
# NOTE (blind build): the exact backend flag may need tuning in the device
# session. run.lit uses %aiecc %backend_flags with the "peano" feature. If this
# invocation is rejected, compare against mlir-aie/test/npu-xrt/cascade_flows/run.lit
# and the live aiecc --help for the current peano flag.
cd "$HERE"
# --no-xchesscc: compile AND link with Peano (disables xbridge/chess). Without
# it aiecc falls back to the chess toolchain (chess-llvm-link), which is absent.
"$AIECC" --no-aiesim --no-xchesscc --peano "$PEANO_INSTALL_DIR" \
  --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
  --xclbin-name=cascade.xclbin --npu-insts-name=insts.bin \
  cascade.mlir

echo "== [3/3] build the host test (XRT) =="
"${HOST_CXX:-clang++}" "$HERE/test.cpp" "$TEST_LIB/test_utils.cpp" -o "$HERE/test.exe" \
  -std=c++17 -Wall \
  -I"$TEST_LIB" -I"${XRT_INC_DIR:-/usr/include}" \
  -L"${XRT_LIB_DIR:-/usr/lib}" \
  -lxrt_coreutil -luuid -lstdc++ -ldl -lrt -lpthread

echo "== done: cascade.xclbin, insts.bin, test.exe =="
echo "   run with ./run.sh  (needs the NPU free -- see README)"
